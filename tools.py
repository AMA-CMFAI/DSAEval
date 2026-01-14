
import requests

def jupyter_code_interpreter_schema() -> dict:
    """
    OpenAI style tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": "jupyter_code_interpreter",
            "description": (
                "Execute python code on a remote Jupyter kernel. "
                "The tool returns standard output (stdout) and any generated plots/images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. IMPORTANT: In certain contexts inside the code "
                        "(e.g., when printing text or setting titles within parentheses), "
                        "newline characters MUST be written as escaped '\\n' instead of literal newlines '\n'. "
                        "Example: print('Line1\\nLine2') is correct; "
                        "print('Line1\nLine2') is WRONG."
                    }
                },
                "required": ["code"],
            },
        },
    }