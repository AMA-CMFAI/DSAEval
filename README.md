# DSAEval

Evaluating Data Science Agents on a Wide Range of Real-World Data Science Problems.

## Installation

```bash
# Create virtual environment
python -m venv dsaeval_env
source dsaeval_env/bin/activate  # Linux/Mac
# dsaeval_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name dsa_eval --display-name "Python (DSA Eval)"
```

## Usage

### Basic Command

```bash
python runner.py \
    --model "anthropic/claude-sonnet-4-20250514" \
    --api_key "your-api-key" \
    --base_url "https://openrouter.ai/api/v1" \
    --qra_path "./dsaeval_small.json" \
    --log_path "./logs" \
    --session_path "./sessions"
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name | `google/gemini-3-pro` |
| `--api_key` | API key | `sk-xxx` |
| `--base_url` | API base URL | `https://openrouter.ai/api/v1` |
| `--qra_path` | Path to evaluation JSON | Required |
| `--max_it` | Max iterations | 20 |
| `--log_path` | Log output directory | `/srv/share/DSA_Eval_logs/` |
| `--session_path` | Session output directory | `/srv/share/dsa_eval_chat_session` |
| `--time_out` | Timeout in seconds | 6000 |
| `--is_multimodal` | Enable multimodal mode | False |

### Running Specific Tasks

```bash
# By ID range (0-9)
python runner.py --id_range "0:10" ...

# From specific ID
python runner.py --id_range "5:" ...

# By JSON file
echo '[35, 45, 46]' > ids.json
python runner.py --target_ids_json "ids.json" ...
```

## Output Files

Per task (`{model}_{folder_name}_{id}_*`):
- `_log.json` - Execution log
- `_final_report.txt` - Final report
- `_code.ipynb` - Jupyter notebook

Summary: `{model}_{date}_all.json`

## Supported Models

Any model with OpenAI-compatible API (OpenAI, OpenRouter, Anthropic, Google Gemini, etc.)


## Download Kaggle Dataset

Use dataset_download.py to download the needed datasets from kaggle.

## Dataset Format

```json
[
  {
    "id": 35,
    "folder_name": "bloomington-accidents",
    "dataset105": "/path/to/dataset",
    "question": "Your task question here..."
  }
]
```
