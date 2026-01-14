import re
import os
import uuid
import base64
from typing import Any, Dict, List


def delete_color_control_char(string: str) -> str:
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', string)


def parse_jupyter_outputs(jupyter_response, working_path: str) -> Dict[str, Any]:

    execute_output: List[str] = []
    plot_files: List[str] = []

    outputs = getattr(jupyter_response, "outputs", [])

    for out in outputs:

        # ---- 1. STDOUT ----
        if out.output_type == "stream" and out.name == "stdout":
            text = out.text or ""
            execute_output.append(text)

        # ---- 2. EXECUTE RESULT ----
        elif out.output_type == "execute_result":
            if out.data:
                text = out.data.get("text/plain", "")
                if text:
                    execute_output.append(text)

        # ---- 3. ERROR ----
        elif out.output_type == "error":
            raw_error = "\n".join(out.traceback) if out.traceback else ""
            clean_error = delete_color_control_char(raw_error)
            execute_output.append(clean_error)

        # ---- 4. DISPLAY_DATA：Save figures ----
        elif out.output_type == "display_data":

            if not out.data:
                continue

            img_b64 = None
            ext = None

            if "image/png" in out.data:
                img_b64 = out.data["image/png"]
                ext = "png"
            elif "image/jpeg" in out.data:
                img_b64 = out.data["image/jpeg"]
                ext = "jpg"

            if img_b64:
                # 随机文件名
                fname = f"display_{uuid.uuid4().hex}.{ext}"
                fpath = os.path.join(working_path, fname)

                with open(fpath, "wb") as f:
                    f.write(base64.b64decode(img_b64))

                plot_files.append(fpath)

                execute_output.append(f"[Image saved: {fname}]")

    return {
        "execute_output": "".join(execute_output),
        "plot_files": plot_files
    }


def _sanitize_code_parentheses(code):
    result = []
    i = 0
    n = len(code)

    # 状态机变量
    in_string = False
    quote_char = None

    while i < n:
        char = code[i]

        is_escaped = (i > 0 and code[i - 1] == '\\')

        if not is_escaped:
            if not in_string and (code[i:i + 3] == '"""' or code[i:i + 3] == "'''"):
                in_string = True
                quote_char = code[i:i + 3]
                result.append(quote_char)
                i += 3
                continue
            elif in_string and quote_char in ['"""', "'''"] and code[i:i + 3] == quote_char:
                in_string = False
                quote_char = None
                result.append(code[i:i + 3])
                i += 3
                continue

            elif not in_string and (char == '"' or char == "'"):
                in_string = True
                quote_char = char
            elif in_string and char == quote_char and quote_char in ['"', "'"]:
                in_string = False
                quote_char = None

        if char == '\n':
            if in_string and quote_char in ['"', "'"]:
                result.append('\\n')
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return "".join(result)


def compress_text_output(text: str, max_chars=4000, max_lines=50) -> str:

    if not text:
        return text

    text = re.sub(r'[-=]{10,}', '---', text)

    lines = text.split('\n')

    if len(lines) > max_lines:
        # save top 15 rows
        head = lines[:15]
        # save the last 15 rows
        tail = lines[-15:]

        skipped = len(lines) - 30
        compressed_text = "\n".join(head) + \
                          f"\n\n... [Output truncated: skipped {skipped} lines of logs] ...\n\n" + \
                          "\n".join(tail)
        return compressed_text

    if len(text) > max_chars:
        # save top 40% and last 40% for long logs.
        cut_len = int(max_chars * 0.4)

        head = text[:cut_len]
        tail = text[-cut_len:]

        skipped_chars = len(text) - (cut_len * 2)
        return head + f"\n... [Output truncated: skipped {skipped_chars} chars] ...\n" + tail

    return text



def get_files_from_path(path):
    return [os.path.join(path, p) for p in os.listdir(path) if p]
