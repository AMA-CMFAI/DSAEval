import os
import re
import json
import time
import logging
import base64
import ast
import nbformat
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, APIError
from typing import Dict, Any, Optional
# =================Configuration=================

# 1. API 设置
API_KEY = "xxx" # replace by your api Key
API_BASE = "xxx" # replace by your api base URL,for example: https://api.openai.com/xxx
EVAL_MODEL_NAME = "xxx" # evaluation model name,forexample anthropic/claude-haiku-4.5

# 2. range settings
EVAL_START_INDEX = 0        # start index (inclusive)
EVAL_BATCH_SIZE = None   # end_index = start_index + batch_size; None indicates all remaining

INNER_CONCURRENCY = 10     # model parrallelism per process

# 3. file path settings
LOGS_ROOT_DIR = "xxx"          # logs path root dir
GROUND_TRUTH_PATH = "xxx.json" # ground truth JSON file path
IMAGE_BASE_DIR = "xxx"         # base dir for reference images
OUTPUT_ROOT_DIR = "xxx"        # output dir for evaluation results

# 4. model list

TARGET_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v3.2-exp",
    "google/gemini-3-pro",
    "minimax/minimax-m2",
    "mistralai/ministral-14b-2512",
    "mimo-v2-flash",
    "x-ai/grok-4.1-fast",
    "z-ai/glm-4.6v",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "openai/gpt-oss-120b",
    "openai/gpt-5.2",
    "openai/gpt-5-nano" 
]

# =========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception:
        return None

def extract_reference_images(item):
    figure_str = item.get("figure")
    if not figure_str or not isinstance(figure_str, str):
        return []
    
    image_ids = re.findall(r'<image_id:(\d+)>', figure_str)
    if not image_ids:
        return []

    notebook_name = item.get("notebook", "")
    folder_name = notebook_name.replace(".ipynb", "") if notebook_name else "unknown"
    
    encoded_images = []
    for img_id in image_ids:
        img_path = os.path.join(IMAGE_BASE_DIR, folder_name, f"{img_id}.png")
        if not os.path.exists(img_path):
             img_path = os.path.join(IMAGE_BASE_DIR, folder_name, f"{img_id}.svg")
        
        if os.path.exists(img_path):
            base64_img = encode_image(img_path)
            if base64_img:
                encoded_images.append(base64_img)
    return encoded_images

def extract_predicted_images(pred_text):
    delimiter = "==========[Finished, Generated Files]=========="
    if delimiter not in pred_text:
        return pred_text.strip(), []
    
    parts = pred_text.split(delimiter)
    text_content = parts[0].strip()
    figures_part = parts[1].strip()
    
    image_paths = []
    if figures_part:
        try:
            parsed_list = ast.literal_eval(figures_part)
            if isinstance(parsed_list, list):
                image_paths = parsed_list
        except Exception:
            image_paths = [p.strip() for p in figures_part.replace("[","").replace("]","").split(",")]

    encoded_images = []
    valid_img_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    for path in image_paths:
        path = path.strip()
        if not path: continue
        if not path.lower().endswith(valid_img_exts): continue

        old_prefix = "/tmp/chat_session"
        new_prefix = "/srv/share/dsa_eval_chat_session"
        if path.startswith(old_prefix):
            real_path = path.replace(old_prefix, new_prefix, 1)
        else:
            real_path = path

        base64_img = encode_image(real_path)
        if base64_img:
            encoded_images.append(base64_img)
            
    return text_content, encoded_images

def extract_id_from_filename(filename):
    
    match = re.search(r'_(\d+)_final_report\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def clean_json_string(s: str) -> str:
    pattern = r'```json\s*(\s*[\S\s]*?)\s*```'
    match = re.search(pattern, s, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return s.strip()

def parse_eval_response(raw_output: str) -> Dict[str, Any]:
    """
    Parse the evaluation response from the model.
    """
    try:
        # 第一步：清理可能存在的 ```json ... ``` 包裹
        cleaned = clean_json_string(raw_output)

        # 第二步：尝试解析为 JSON
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}")
        return {
            "error": "JSON Decode Failed",
            "raw_output": raw_output,
            "parsed_content": cleaned
        }
    except Exception as e:
        logging.error(f"Unexpected error during parsing: {e}")
        return {
            "error": "Parsing Error",
            "raw_output": raw_output,
            "exception": str(e)
        }



import json
import nbformat
import logging

def notebook_to_eval_inputs(notebook_path: str):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # clean markdown cells
        if 'cells' in data and isinstance(data['cells'], list):
            for cell in data['cells']:
                if cell.get('cell_type') == 'markdown':
                    cell.pop('execution_count', None)
                    cell.pop('outputs', None)

        cleaned_json_string = json.dumps(data)
        notebook = nbformat.reads(cleaned_json_string, as_version=4)
        
        if not notebook.cells:
            return None, 0.0, False
        remaining_cells = notebook.cells[1:]
        code_blocks = []
        total_code_count = 0
        error_code_count = 0
        is_last_code_cell_success = True 
        
        for cell in remaining_cells:
            if cell.cell_type == 'code':
                total_code_count += 1
                
                # streamline outputs to check for errors
                has_error = False
                for o in cell.outputs:
                    # check for explicit error output
                    if o.output_type == 'error':
                        has_error = True
                        break
                    # check stream outputs for error indicators
                    if o.output_type == 'stream':
                        text = "".join(o.get('text', []))
                        if "[Error]" in text or "Traceback" in text:
                            has_error = True
                            break
                
                if has_error:
                    error_code_count += 1
                
                is_last_code_cell_success = not has_error

                code_blocks.append({
                    "step": total_code_count,
                    "code": cell.source,
                    "execution_status": "error" if has_error else "success"
                })

        code_error_rate = (error_code_count / total_code_count) if total_code_count > 0 else 0.0
        
        predicted_code_json = {
            "total_steps": total_code_count,
            "implementation_details": code_blocks
        }

        return predicted_code_json, code_error_rate, is_last_code_cell_success

    except Exception as e:
        # logging.error(f"Error parsing notebook: {e}")
        return None, 0.0, False


# -----------------------------
# Evaluation Prompt
# -----------------------------
def evaluate_answer_multimodal(problem, standard_answer, predicted_code, predicted_reasoning_answer, 
                               ref_images, pred_images, client, model):
    
    EVAL_PROMPT = """You are a data science evaluation assistant. Your task is to objectively evaluate the quality of a generated solution for the given [PROBLEM].

### Evaluation Principles:
1. **Core Objective:** Did the model answer the [PROBLEM] correctly?
2. **The [STANDARD_ANSWER] represents just one possible way to solve the problem. The [PREDICTED_REASONING_ANSWER] may adopt a different but equally valid or even superior approach, such as using a linear model instead of a neural network, reporting a statistical summary rather than a visualization, or employing a different training–test split.
3. **Visuals:** You may receive "Reference Figures" and "Predicted Figures". 
   - **DO NOT PENALIZE** if the predicted answer has NO figures, *UNLESS* the [PROBLEM] explicitly asks to "plot", "visualize", or "graph".
   - If the problem does not explicitly ask for a plot, a text/numerical analysis is  valid even if the reference answer used a plot.
   - If both have figures, compare them to ensure the predicted figure shows correct data distrubtion, trends and insights(ignoring style differences).
4. **Code-Answer Consistency :** You must verify if the final answer/conclusion in [predicted_reasoning_answer] is actually derived from [PREDICTED_CODE]. 
   - If the text reports a specific value or insight that contradicts the code's logic or is not produced by the code, it is a **hallucination**.
   - If some code cells contain errors but are followed by subsequent cells that fix these issues and successfully produce the result, it is NOT a hallucination.
### Evaluation Dimensions :

* **Consistency (true or false):** Focus on the alignment between [PREDICTED_CODE] and [predicted_reasoning_answer].
    - **true**: The narrative accurately reflects the code's final output. 
    - **false**: The narrative claims results that the code did not produce, or the code's unresolved errors make the reported conclusion impossible.
    

* **ReasoningProcess:** Focus on the conceptual soundness and validity of the logic or method.
    -Score 0-2 (Poor): The reasoning contains fundamental errors, is completely irrelevant to the problem, or violates core statistical/mathematical principles (e.g., using a classification model to solve a regression problem without reasonable transformation logic, confusing causal relationships with correlational relationships).​
    -Score 3-4 (Weak): The direction of reasoning is relevant to the problem, but there are serious logical flaws or critical assumption errors; the core method is completely inappropriate and cannot achieve the problem's objectives (e.g., using simple linear regression to model highly non-linear data without any data transformation, ignoring key variables leading to complete model failure).​
    -Score 5-6 (Fair): The reasoning partially aligns with correct principles, but there are obvious logical defects, unreasonable assumptions, or suboptimal method selection. Although it can initially approach the target, the reliability of the results is low (e.g., using an appropriate model type but failing to handle multicollinearity, incomplete feature engineering limiting model performance).​
    -Score 7-8 (Good): The reasoning is conceptually clear and logically rigorous on the whole; the method selection is appropriate and efficient, with only minor flaws or room for optimization that does not affect the validity of core conclusions (e.g., reasonable model selection and parameter settings, but failing to try better feature combinations, or having room for improvement in cross-validation strategies).​
    -Score 9-10 (Excellent): The reasoning is conceptually clear and logically fully valid; the method selection is accurate and efficient, perfectly adapting to the problem's requirements (even if different from the reference solution, it has equivalent or better rationality), with no obvious room for optimization.

* **CodeSteps:** Focus on the correctness and completeness of the provided [PREDICTED_CODE].
    * 0-2 (Poor): Codes are missing, disordered, or mathematically incorrect; they fail to connect to the final result.
    * 3-4 (Weak): Codes contain significant errors or omissions in key steps, leading to incorrect intermediate results that cannot support the final conclusion.
    * 5-6 (Fair): Several intermediate codes are correct, but others contain minor errors, unclear reasoning, or incomplete implementation. The overall direction is reasonable, but consistency or precision is lacking.
    * 7-8 (Good): Most intermediate codes are correct and coherent, with only minor mistakes or areas for clearer explanation. They generally lead logically toward the final result.
    * 9-10 (Excellent): All major intermediate steps or computational procedures are correct, coherent, and lead logically toward the final result.


* **FinalResults General Principles**
    The final result must be relevant to the stated question or task.
    Differences from the reference or expected answer are acceptable if the alternative result is justified, methodologically sound, and produces a comparable or better outcome.
    
    **Evaluation depends on the problem type:**

    (1) **Quantitative tasks** (e.g., MSE, RMSE, Accuracy, R²):
        Judge correctness based on whether the reported value is reasonable and consistent with the metric’s directionality. The plausibility of the numerical range and internal consistency are important.
        * **Metrics Choice:** Assess if chosen metrics are appropriate and relevant compared to the reference.
        * **Metrics Quantity:** Assess if core required metrics are included and if valuable supplementary metrics are provided.

    (2) **Qualitative tasks** (e.g., causal conclusion, hypothesis decision):
        Judge whether the conclusion logically follows from prior reasoning, is factually sound, and addresses the question appropriately.

    (3) **Visualizations** (Triggered ONLY when both Reference and Predicted answers contain figures):
        Evaluate the predicted figure(s) as a key component of the final result using these four dimensions:
        * **Alignment (Faithfulness):** Does the plot accurately reflect the data and variables requested? Are axis labels/titles consistent with the finding?
        * **Effectiveness (Best Practices):** Is the chart type appropriate (e.g., using a bar chart for categorical comparison vs. a line chart for trends)? Does it highlight insights effectively without misleading encodings?
        * **Readability (Clarity):** Is text legible? Is there overlapping clutter? Is the layout clean?
        * **Aesthetics:** Is the design professional and harmonious (e.g., color palette, balance)?

    **Holistic Scoring (Integrating Text, and Visuals):**

    * **Score 0-2 (Poor):** The result is objectively incorrect, nonsensical, or irrelevant.
        (Quantitative) Uses completely inappropriate metrics.
        (Qualitative) Provides meaningless text or a "hallucination."
        **(Visual)** The figure plots the wrong variables, contradicts the textual conclusion, or is completely illegible/broken.

    * **Score 3-4 (Weak):** The result is plausible but objectively incorrect or contains demonstrable flaws.
        (Quantitative) Metrics are miscalculated or core metrics are omitted.
        (Qualitative) The conclusion contradicts evidence or involves severe logical leaps.
        **(Visual)** The figure exists but uses an incorrect chart type (e.g., pie chart for complex time series), has severe overplotting making it unreadable, or lacks essential labels (axes/legend) rendering it useless.

    * **Score 5-6 (Fair - Baseline):** The result is correct and achieves the same level of quality as the standard answer.
        (Quantitative) Uses correct metrics with correct calculations.
        (Qualitative) The conclusion is sound and supported by evidence.
        **(Visual)** The figure is functionally correct (Alignment & Effectiveness are good). It accurately represents the data and supports the conclusion, though it may lack aesthetic polish or advanced styling compared to the reference.

    * **Score 7-8 (Good):** The result is objectively correct and demonstrably superior to the standard answer.
        (Quantitative) Corrects a minor flaw in reference or adds valuable supplementary metrics.
        (Qualitative) Provides more insightful analysis.
        **(Visual)** The figure is superior in **Readability** or **Effectiveness**. For example, it adds trendlines where the reference didn't, resolves overlapping data points better, or uses a clearer color scheme that enhances interpretation.

    * **Score 9-10 (Excellent):** The result is objectively correct and significantly superior to the standard answer.
        (Quantitative) Uses a much better methodology leading to significantly better metrics.
        (Qualitative) Provides high-precision, deep insights missed by the reference.
        **(Visual)** The figure is publication-quality (**Aesthetics** & **Effectiveness**). It provides a deeper layer of insight (e.g., interactive elements, multi-panel insights, perfect annotations) that significantly aids in understanding the data better than the reference's static or simple plot.

[PROBLEM]: {problem}

[STANDARD_ANSWER]: 
{standard_answer}

[PREDICTED_CODE] 
{predicted_code}

[predicted_reasoning_answer]
{predicted_reasoning_answer}

# Output Format
Return **only** the following JSON, without any additional text or code
{{
  "Analysis": A brief explanation  of why these scores were given,
  "ReasoningProcess": <score>,
  "CodeSteps": <score>,
  "FinalResults": <score>，
  "Consistency": <true or false>
}}

"""
    
    formatted_prompt = EVAL_PROMPT.format(
        problem=problem,
        standard_answer=standard_answer,
        predicted_code=predicted_code,
        predicted_reasoning_answer=predicted_reasoning_answer
    )

    content = [{"type": "text", "text": formatted_prompt}]
    if ref_images:
        content.append({"type": "text", "text": "\n\n--- STANDARD REFERENCE FIGURES (For Context Only) ---"})
        for img in ref_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}", "detail": "auto"}  
            })
    if pred_images:
        content.append({"type": "text", "text": "\n\n--- PREDICTED FIGURES (To Evaluate) ---"})
        for img in pred_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}", "detail": "auto"} 
            })
    else:
        content.append({"type": "text", "text": "\n\n--- PREDICTED FIGURES ---\n[No figures generated by the model.]"})

    messages = [{"role": "system", "content": "You are an objective evaluator."}, {"role": "user", "content": content}]
    
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1
            )
            raw_content = response.choices[0].message.content

            try:
                parsed_json = parse_eval_response(raw_content)
                return json.dumps(parsed_json)  
            except Exception as e:
                logging.error(f"Final parse failed: {e}")
                return json.dumps({"error": "Final Parse Failed", "raw_output": raw_content})
          
        except Exception as e:
            last_error = str(e)
            logging.error(f"Attempt {i+1} failed for model {model}: {e}")
            if "context_length_exceeded" in str(e).lower():
                return json.dumps({"error": "Context Length Exceeded", "raw_output": str(e)})
            time.sleep(2**i)
            
    # if run out of retries
    logging.error(f"All retries failed. Last error: {last_error}")
    return json.dumps({"error": "API Call Failed", "last_error": last_error})



def process_single_task(gt_item, report_path, code_path, client, safe_model_name):
    """
    process a single task given ground truth item, report path, and code path
    """
    try:
        q_id = gt_item.get('id')
        data_type = gt_item.get('data_type', '')
        domain = gt_item.get('domain', '')  
        task_type = gt_item.get('task_type', '')

        # 1. read predicted text from report file
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        clean_pred_text = report_content.strip()

        # 2.  read generated images from log file
        pred_imgs = []
        log_path = report_path.replace("_final_report.txt", "_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    image_paths = log_data.get("generated_files", [])
                    
                    valid_img_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
                    for path in image_paths:
                        path = path.strip()
                        if not path or not path.lower().endswith(valid_img_exts): 
                            continue
                        
                        # /tmp/chat_session -> /srv/share/dsa_eval_chat_session
                        old_prefix = "/tmp/chat_session"
                        new_prefix = "/srv/share/dsa_eval_chat_session"
                        real_path = path.replace(old_prefix, new_prefix, 1) if path.startswith(old_prefix) else path
                        
                        base64_img = encode_image(real_path)
                        if base64_img:
                            pred_imgs.append(base64_img)
            except Exception as e:
                logging.warning(f"Error parsing log file {log_path}: {e}")

        # 3. scrape code JSON from notebook
        predicted_code_json = None
        code_error_rate=0.0
        final_code_cell_success=None
        if code_path and os.path.exists(code_path):
            predicted_code_json,code_error_rate,final_code_cell_success = notebook_to_eval_inputs(code_path)

        problem = gt_item.get("question", "")
        std_full = f"Reasoning:\n{gt_item.get('reasoning', '')}\n\nAnswer:\n{gt_item.get('answer', '')}"
        
        # scrape reference images
        ref_imgs = extract_reference_images(gt_item)

        # 4. evaluate 
        eval_raw = evaluate_answer_multimodal(
            problem=problem,
            standard_answer=std_full,
            predicted_code=predicted_code_json,         
            predicted_reasoning_answer=clean_pred_text, 
            ref_images=ref_imgs,
            pred_images=pred_imgs,                    
            client=client,
            model=EVAL_MODEL_NAME
        )
        
        try:
            eval_clean = clean_json_string(eval_raw)
            eval_json = json.loads(eval_clean)
        except json.JSONDecodeError:
            eval_json = {"error": "JSON Decode Failed", "raw_output": eval_raw}

        filename = os.path.basename(report_path)
        
        return {
            "id": q_id,
            "data_type": data_type,
            "domain": domain,
            "task_type": task_type,
            "file_name": filename,
            "full_path": report_path,
            "code_path": code_path, 
            "model": safe_model_name,
            "has_ref_figure": len(ref_imgs) > 0,
            "has_pred_figure": len(pred_imgs) > 0,
            "code_error_rate": code_error_rate,
            "final_code_cell_success": final_code_cell_success,
            "scores": eval_json
        }
    except Exception as e:
        logging.error(f"Error processing task ID {gt_item.get('id')}: {e}")
        return None



def process_model_directory_by_gt(model_rel_path, target_gt_items):
  
    input_base_dir = os.path.join(LOGS_ROOT_DIR, model_rel_path)
    safe_model_name = model_rel_path.replace("/", "_")
    output_dir = os.path.join(OUTPUT_ROOT_DIR, safe_model_name)
    
    if not os.path.exists(input_base_dir):
        return f"{safe_model_name}: Dir not found: {input_base_dir}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    end_index = EVAL_START_INDEX + len(target_gt_items)
    output_filename = f"evaluation_summary_{EVAL_START_INDEX}_{end_index}.json"
    output_file_path = os.path.join(output_dir, output_filename)

    # initialize Client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    # 1. construct id to file mapping
    
    id_to_file_pair = {} # {id: (report_path, code_path)}
    logging.info(f"[{safe_model_name}] Scanning directory recursively: {input_base_dir}")
    
    try:
        report_count = 0
        for root, dirs, files in os.walk(input_base_dir):
            for file in files:
                if file.endswith("_final_report.txt"):
                    fid = extract_id_from_filename(file)
                    if fid is not None:
                        report_full_path = os.path.join(root, file)
                        # macthing code file
                        code_file_name = file.replace("_final_report.txt", "_code.ipynb")
                        code_full_path = os.path.join(root, code_file_name)
                        
                        if os.path.exists(code_full_path):
                            id_to_file_pair[fid] = (report_full_path, code_full_path)
                        else:
                            id_to_file_pair[fid] = (report_full_path, None)
                        report_count += 1
        
        logging.info(f"[{safe_model_name}] Scanned {report_count} reports, mapped {len(id_to_file_pair)} unique IDs.")
        
    except Exception as e:
        logging.error(f"Failed to scan directory {input_base_dir}: {e}")
        return f"{safe_model_name}: Scan failed"

    if len(id_to_file_pair) == 0:
         return f"{safe_model_name}: Found 0 reports after recursive scan."

    results = []
    
    # 2. macth & process
    with ThreadPoolExecutor(max_workers=INNER_CONCURRENCY) as executor:
        futures = {}
        for gt_item in target_gt_items:
            q_id = gt_item.get('id')
            
            if q_id in id_to_file_pair:
                report_path, code_path = id_to_file_pair[q_id] 
                # file_path = id_to_file_path[q_id]
                fut = executor.submit(process_single_task, gt_item, report_path, code_path, client, safe_model_name)
                futures[fut] = q_id
        
        if not futures:
            return f"{safe_model_name}: No matching files found for the range."

        # tqdm monitoring
        pbar = tqdm(total=len(futures), desc=f"Eval {safe_model_name}", position=0, leave=True)
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            pbar.update(1)
        pbar.close()

    # 3. save results to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return f"{safe_model_name}: Processed {len(results)} items. Saved to {output_filename}"
# -----------------------------

# -----------------------------
def load_ground_truth_list(json_path):
    """
    load the full ground truth list from a JSON file
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 确保数据是列表
        if isinstance(data, list):
            return data
        else:
            logging.error("Ground truth file is not a JSON list.")
            return []
    except Exception as e:
        logging.error(f"load_ground_truth_list failed: {e}")
        return []

def main():
    # load full ground truth list
    full_gt_list = load_ground_truth_list(GROUND_TRUTH_PATH)
    if not full_gt_list: 
        print("Error: No ground truth data found.")
        return

    total_items = len(full_gt_list)
    print(f"Total Ground Truth Items: {total_items}")

    # slice the target GT items based on start index and batch size
    start = EVAL_START_INDEX
    if EVAL_BATCH_SIZE is None:
        end = total_items
    else:
        end = min(start + EVAL_BATCH_SIZE, total_items)

    if start >= total_items:
        print(f"Start index {start} is out of bounds (Total: {total_items}). Exiting.")
        return

    target_gt_items = full_gt_list[start:end]
    print(f"Target Evaluation Range: Index {start} to {end} (Count: {len(target_gt_items)})")
    preview_ids = [item.get('id') for item in target_gt_items[:5]]
    print(f"Preview IDs in this batch: {preview_ids} ...")
    with ThreadPoolExecutor(max_workers=min(100, len(TARGET_MODELS))) as executor:
        futures = []
        for model_rel_path in TARGET_MODELS:

            future = executor.submit(process_model_directory_by_gt, model_rel_path, target_gt_items)
            futures.append(future)
        
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()