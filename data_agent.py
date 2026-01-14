import json
import traceback
from typing import Any, Callable, Dict, List, Optional
from prompts import IMPORT_PMT
from openai import OpenAI
from tools import jupyter_code_interpreter_schema  # 确保 tools.py 里有这个 schema
from utils import *
import random
import string
from utils import _sanitize_code_parentheses


class DataScienceAgent:

    def __init__(
            self,
            client: OpenAI,
            model: str,
            system_prompt: str = "",
            max_iterations: int = 20,
            is_multimodal: bool = False
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.is_multimodal = is_multimodal

        self.tools_registry: Dict[str, Callable] = {}
        self.notebook_cells = []
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def register_tool(self, name: str, func: Callable[[Dict[str, Any]], Any]):
        self.tools_registry[name] = func

    def _build_tools_schema(self):
        return [jupyter_code_interpreter_schema()]

    def _assistant_message_to_dict(self, msg) -> Dict[str, Any]:
        content = msg.content or "..."

        base: Dict[str, Any] = {
            "role": msg.role,
            "content": content,
        }
        if getattr(msg, "tool_calls", None):
            base["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return base

    def _add_markdown_cell(self, text):
        if not text: return
        self.notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def _add_code_cell(self, code, text_output, images_b64):
        outputs = []
        if text_output:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in text_output.split("\n")]
            })
        if images_b64:
            for img in images_b64:
                outputs.append({
                    "data": {
                        "image/png": img,
                        "text/plain": "<IPython.core.display.Image object>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })

        self.notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.split("\n")]
        })

    def reset_notebook_buffer(self):
        self.notebook_cells = []

    def save_notebook(self, filepath):
        notebook_json = {
            "cells": self.notebook_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {"name": "python", "version": "3.10"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook_json, f, indent=1, ensure_ascii=False)
        print(f"📘 Notebook saved to: {filepath}")

    def run(self, user_input: str, working_path: str, jupyter_sandbox, verbose: bool = True):

        reasoning_logs = []
        final_report = ""
        usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        self._add_markdown_cell(f"# Task Request\n**User**: {user_input}")
        self.messages.append({"role": "user", "content": user_input})

        tools_schema = self._build_tools_schema()

        def execute_jupyter(args: dict) -> dict:
            code = args.get("code", "")
            if 'google' in self.model:
                code = _sanitize_code_parentheses(code)
            # if 'matplotlib' or 'plt' in code and "%matplotlib inline" not in code:
            #     code = "%matplotlib inline\n" + code
            text_out, images = jupyter_sandbox.execute_code(code)
            compressed_text = compress_text_output(text_out)
            return {
                "text": compressed_text,
                "raw_text": text_out,
                "images": images
            }

        # imp_result = execute_jupyter({"code":IMPORT_PMT})
        magic_plt = execute_jupyter({"code":"%matplotlib inline"})

        self.tools_registry["jupyter_code_interpreter"] = execute_jupyter

        for step in range(self.max_iterations):

            step_header = f"[Agent] step {step}: call LLM"
            reasoning_logs.append(step_header)
            if verbose: print("\n" + step_header)

            try:

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=tools_schema,
                    tool_choice="auto",
                )

                if hasattr(resp, 'usage') and resp.usage:
                    usage_stats["prompt_tokens"] += resp.usage.prompt_tokens
                    usage_stats["completion_tokens"] += resp.usage.completion_tokens
                    usage_stats["total_tokens"] += resp.usage.total_tokens

            except Exception as e:
                error_log = f"API Error: {e}"
                reasoning_logs.append(error_log)
                print(error_log)
                break

            choice = resp.choices[0].message


            self.messages.append(self._assistant_message_to_dict(choice))
            tool_calls = getattr(choice, "tool_calls", None)

            if not tool_calls:
                final_report = choice.content or ""
                reasoning_logs.append("[Agent] final answer:")
                reasoning_logs.append(final_report)
                if verbose:
                    print("[Agent] final answer:")
                    print(final_report)

                gen_files = get_files_from_path(working_path)

                finished_log = {
                    "reasoning": "\n".join(reasoning_logs),
                    "final_report": final_report,
                    "finished": True,
                    "coding": len(self.notebook_cells) > 1,  # 简单判断有没有 coding
                    "generated_files": gen_files,
                    "token_usage": usage_stats
                }
                return finished_log, final_report

            for tool_call in tool_calls:
                tool_id = tool_call.id
                multimodal_message = None

                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments or "{}"

                log_tool = f"[Agent] tool_call -> {tool_name}({tool_args_str})"
                reasoning_logs.append(log_tool)
                if verbose: print(log_tool)

                tool_content_for_llm = "No outputs found in kernel."

                if tool_name not in self.tools_registry:
                    tool_content_for_llm = f"Unknown tool: {tool_name}"
                else:
                    try:
                        parsed_args = json.loads(tool_args_str)
                    except json.JSONDecodeError as e:
                        tool_content_for_llm = f"JSON Error: {e}"
                    else:
                        try:
                            code_to_run = parsed_args.get("code", "")
                            tool_result = self.tools_registry[tool_name](parsed_args)
                            text_output = tool_result["text"]
                            text_output_full = tool_result.get("raw_text", text_output)
                            images_b64 = tool_result["images"]
                            print(f"[Tool] tool result: {text_output_full}, \n{len(images_b64)}images.")

                            tool_content_for_llm = text_output if text_output and text_output != '' else "No execution information found in kernel."

                            self._add_code_cell(code_to_run, text_output_full, images_b64)

                            tool_output_text = text_output
                            if images_b64:
                                tool_output_text += f"\n[System] {len(images_b64)} image(s) generated."

                            if self.is_multimodal and images_b64:
                                content_parts = [{
                                    "type": "text",
                                    "text": "Here are the plots generated by the code:"
                                }]
                                for b64_str in images_b64:
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                                    })
                                multimodal_message = {
                                    "role": "user",
                                    "content": content_parts
                                }

                        except Exception as e:
                            reasoning_logs.append(f"Execution Exception: {repr(e)}")
                            tool_content_for_llm = f"Execution Exception: {repr(e)}\nYou MUST have a faster solution."
                            print(f"[Error] in execution {e}.")

                reasoning_logs.append(f"[Tool Output] {str(tool_content_for_llm)}")

                if "google" in self.model:
                    
                    self.messages.append({
                        "role": "user",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_content_for_llm
                    })

                else:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_content_for_llm
                    })

                if multimodal_message:
                    self.messages.append(multimodal_message)
                    reasoning_logs.append("[System] Sent Vision Observation to VLM.")

        final_report = "Returned with no final report."
        log_json = {
            "reasoning": "\n".join(reasoning_logs),
            "final_report": final_report,
            "finished": False,
            "token_usage": usage_stats
        }
        return log_json, final_report