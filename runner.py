import json
from tools import *
import os
from cache import Cache
from datetime import datetime
# from agent_sandbox import Sandbox
import tempfile
import shutil
import os
import argparse
import json
from datetime import datetime
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
from prompts import *
from openai import OpenAI
from tools import *
from data_agent import DataScienceAgent
from utils import *
from jupyter_sandbox import JupyterSandbox
import ast

def run_workflow(args):
    with open(args.qra_path) as f:
        qra_json = json.load(f)
    workflow_json = []
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    target_start = 0
    target_end = float('inf')

    found_start_point = True

    if args.id_range and not args.target_ids_json:
        try:
            parts = args.id_range.split(":")

            if parts[0].strip():
                target_start = int(parts[0])
                if target_start > 0:
                    found_start_point = False

            if len(parts) > 1 and parts[1].strip():
                target_end = int(parts[1])

            print(f"🎯 ID Range Active: Start Trigger={target_start}, End Limit={target_end}")

        except ValueError:
            print("❌ Error: id_range format invalid.")
            return []

    specific_id_set = set()

    if args.target_ids_json:
        if os.path.exists(args.target_ids_json):
            try:
                print(f"📂 Loading target IDs from: {args.target_ids_json}")
                with open(args.target_ids_json, 'r') as f:
                    loaded_ids = json.load(f)

                specific_id_set = {int(x) for x in loaded_ids}

                print(f"🎯 Targeted Execution Mode: {len(specific_id_set)} tasks loaded.")
            except Exception as e:
                print(f"❌ Error loading JSON ID list: {e}")
                return []
        else:
            print(f"❌ Error: File not found: {args.target_ids_json}")
            return []

    for dataset_group in qra_json:
        if not dataset_group: continue

        group_ids = [int(item["id"]) for item in dataset_group]
        if specific_id_set:

            if not specific_id_set.intersection(set(group_ids)):
                continue

        elif not found_start_point:
            if target_start not in group_ids:
                continue

        dataset_path = dataset_group[0].get("dataset105")
        folder_name = dataset_group[0]["folder_name"]

        cache = Cache()
        working_path = cache.create_session_path()
        print(f"\n\n🌍 === New Dataset Session: {folder_name} ===")

        try:
            with JupyterSandbox(args.time_out) as sb:
                DSAgent_PROMPT = DSAgent_PROMPT_VLM if args.is_multimodal else DSAgent_PROMPT_LLM

                agent = DataScienceAgent(
                    client=client,
                    model=args.model,
                    system_prompt=DSAgent_PROMPT.format(data_path=dataset_path, working_path=working_path),
                    max_iterations=args.max_it,
                    is_multimodal=args.is_multimodal
                )

                # =======================================================
                # data loop：running task by task for each question in dataset
                # =======================================================
                for j in dataset_group:
                    case_id = str(j["id"])
                    query = j['question']
                    case_id_int = int(case_id)

                    if specific_id_set:
                        if case_id_int not in specific_id_set:
                            continue

                    else:
                        if case_id_int >= target_end:
                            continue
                        if not found_start_point:
                            if case_id_int == target_start:
                                print(f"🚀 Found Start ID: {case_id_int}")
                                found_start_point = True
                            else:
                                continue

                    log_name = f"{args.model}_{folder_name}_{case_id}".replace("/", "_")
                    log_path = os.path.join(args.log_path, args.model, folder_name, log_name)
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)

                    notebook_path = log_path + "_code.ipynb"

                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"\n👉[{current_time}] Running Task ID: {case_id}")

                    start_time = time.time()

                    try:
                        agent.reset_notebook_buffer()
                        log_json, final_report = agent.run(
                            user_input=query,
                            working_path=working_path,
                            jupyter_sandbox=sb,
                            verbose=True
                        )

                        duration = time.time() - start_time
                        agent.save_notebook(notebook_path)

                        log_json.update({
                            "id": case_id,
                            "folder_name": folder_name,
                            "log_path": log_path,
                            "session_path": os.path.join(args.session_path, working_path),
                            "notebook_path": notebook_path,
                            "duration": round(duration, 2)
                        })
                        workflow_json.append(log_json)

                        with open(log_path + "_log.json", "w") as f:
                            json.dump(log_json, f, ensure_ascii=False, indent=2)

                        with open(log_path + "_final_report.txt", "w") as f:
                            f.write(final_report)

                        print(f"✅ Task {case_id} finished")

                    except Exception as e:
                        error_msg = f"Error in Task {case_id}: {repr(e)}"
                        print(f"❌ {error_msg}")

                        workflow_json.append({
                            "id": case_id,
                            "finished": False,
                            "program issues": True,
                            "folder_name": folder_name,
                            "log_path": log_path,
                            "error": error_msg
                        })
                        with open(log_path + "_error.txt", "w") as f:
                            f.write(error_msg)

                        # if the kernel is dead, closing all tasks in this dataset.
                        if "Kernel died" in str(e) or "Timeout" in str(e):
                            break
                        else:
                            continue

        except Exception as e:
            print(f"⛔ Critical Error in Dataset {folder_name}: {e}")
            continue

    print("\n=== All Workflows Finished ===")

    date_str = datetime.now().strftime("%Y%m%d")
    wf_filename = f"{args.model}_{date_str}_all.json".replace("/", "_")
    save_path = os.path.join(args.log_path, wf_filename)

    try:
        with open(save_path, "w") as f:
            json.dump(workflow_json, f, indent=2, ensure_ascii=False)
        print(f"📄 Full workflow log saved to: {save_path}")
    except Exception as e:
        print(f"Error saving workflow json: {e}")

    return workflow_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all tasks by data science agent"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-pro",
        help="Model name to run the agent with (default: openai/gpt-5-nano)"
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxx",
        help="API key for authentication (default: sk-xxx)"
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for the model API (default: https://openrouter.ai/api/v1)"
    )

    parser.add_argument(
        "--qra_path",
        type=str,
        default="/srv/share/dsaeval_small.json",
        help="Path to the QRA json file for evaluation (default: /srv/share/dsaeval_small.json)"
    )


    parser.add_argument(
        "--max_it",
        type=int,
        default=20,
        help="Maximum number of iterations (default: 20)"
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="/srv/share/DSA_Eval_logs/",
        help="Path to the log file (default: /srv/share/agent_logs)"
    )

    parser.add_argument(
        "--session_path",
        type=str,
        default="/srv/share/dsa_eval_chat_session",
        help="Session path of chat (mounted with docker), figures, models that be saved. (default: /srv/share/dsa_eval_chat_session)"
    )

    parser.add_argument(
        "--time_out",
        type=int,
        default=6000,
        help="Time out for each step."
    )


    parser.add_argument(
        "--id_range",
        type=str,
        help="Range of IDs to run, e.g. 0:10, 5:, :20",
        required=False
    )

    parser.add_argument(
        "--target_ids_json",
        type=str,
        required=False,
        help="Path to a JSON file containing a list of task IDs to run (e.g. 'retry_ids.json')"
    )

    parser.add_argument('--is_multimodal', action='store_true', help="Is the LM multimodal?")

    args = parser.parse_args()
    print(args)
    run_workflow(args)
