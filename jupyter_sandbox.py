import queue
import re
import time
from jupyter_client import KernelManager
from textwrap import dedent
from prompts import IMPORT_PMT


class JupyterSandbox:
    def __init__(self, timeout=6000):
        self.timeout = timeout
        self.km = None
        self.kc = None

    def __enter__(self):
        self.start_session()
        self.run_import_plot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def _get_kernel_process(self, kernel_manager):
        if hasattr(kernel_manager, 'kernel') and kernel_manager.kernel:
            return kernel_manager.kernel
        if hasattr(kernel_manager, 'provisioner') and kernel_manager.provisioner:
            if hasattr(kernel_manager.provisioner, 'process'):
                return kernel_manager.provisioner.process
        return None

    def start_session(self):
        print(f"🚀 Starting kernel 'dsa_eval'...")
        self.km = KernelManager(kernel_name='dsa_eval')

        try:
            self.km.start_kernel()
        except Exception as e:
            raise Exception(f"Failed to launch kernel process: {e}")

        time.sleep(1)

        process = self._get_kernel_process(self.km)

        if process:
            if process.poll() is not None:
                error_log = "Could not read stderr."
                if process.stderr:
                    try:
                        error_log = process.stderr.read().decode('utf-8', errors='replace')
                    except Exception:
                        pass

                print("❌ Kernel process died immediately!")
                print("=" * 20 + " KERNEL STDERR " + "=" * 20)
                print(error_log)
                print("=" * 50)
                self.close_session()
                raise RuntimeError(f"Kernel died on startup. Log: {error_log[:200]}...")

        self.kc = self.km.client()
        self.kc.start_channels()

        try:
            self.kc.wait_for_ready(timeout=60)
            print(f"✅ [Local Kernel Started] Ready!")
        except RuntimeError:
            if process and process.poll() is not None:
                error_log = "Unknown error"
                if process.stderr:
                    error_log = process.stderr.read().decode('utf-8', errors='replace')
                print("❌ Kernel died while waiting for readiness!")
                print("=" * 20 + " KERNEL STDERR " + "=" * 20)
                print(error_log)
                print("=" * 50)

            self.close_session()
            raise Exception("Failed to start local kernel: Timeout (Kernel might have crashed).")

    def run_import_plot(self):
        self.execute_code(IMPORT_PMT)
        self.execute_code("%matplotlib inline")

    def execute_code(self, code):
        if not self.kc:
            raise Exception("No active session.")

        msg_id = self.kc.execute(code)
        outputs = []
        images = []

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
            except queue.Empty:
                print(f"[Sandbox Warning] Local Kernel timed out after {self.timeout}s.")
                self.restart_session()
                raise TimeoutError(f"Local Kernel execution timed out and shut down.\n[Sandbox] Kernel Restarted.")



            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'stream':
                outputs.append(content['text'])
            elif msg_type == 'execute_result' or msg_type == 'display_data':
                data = content['data']
                if 'text/plain' in data:
                    outputs.append(data['text/plain'])
                if 'image/png' in data:
                    images.append(data['image/png'])
                    outputs.append("\n[Graphics: Image captured]")
            elif msg_type == 'error':
                traceback_list = content.get("traceback", [])
                traceback_str = "\n".join(traceback_list)
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_traceback = ansi_escape.sub('', traceback_str)
                error_msg = f"{content['ename']}: {content['evalue']}\nTraceback:\n{clean_traceback}"
                outputs.append(f"[Error] {error_msg}")
            elif msg_type == 'status':
                if content['execution_state'] == 'idle':
                    break

        return "".join(outputs), images

    def close_session(self):
        if self.kc:
            try:
                self.kc.stop_channels()
            except Exception:
                pass
            self.kc = None

        if self.km:
            try:
                # KernelManager 的 shutdown 是同步友好的
                self.km.shutdown_kernel(now=True)
                print(f"🗑️  [Local Kernel Killed]")
            except Exception as e:
                print(f"⚠️ Error shutting down kernel: {e}")
            self.km = None

    def restart_session(self):
        print("🔄 [Sandbox] Triggering Kernel Restart...")

        self.close_session()

        try:
            self.start_session()
            self.run_import_plot()
            print("✅ [Sandbox] Kernel Restarted Successfully.")
        except Exception as e:
            print(f"❌ [Sandbox] Kernel Restart Failed: {e}")
            raise e


if __name__ == '__main__':
    with JupyterSandbox() as sb:

        multi_plot_code = dedent("""
                %matplotlib inline
                import matplotlib.pyplot as plt
                import numpy as np
                """)

        plot_test2 = dedent("""
                plt.figure()
                plt.plot(np.sin(np.linspace(0, 10, 100)), color='red')
                plt.title("Plot 1: Sine")
                plt.show()  
                plt.close()
                print("Line...")
    
                plt.figure()
                plt.plot(np.cos(np.linspace(0, 10, 100)), color='green')
                plt.title("Plot 2: Cosine")
                plt.show() 
                plt.close()
            """)



        gpu_test = """
                import torch
                print(f"Is CUDA available? {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"Device Name: {torch.cuda.get_device_name(0)}")
                """
        step1 = "a='hello world'"

        step2 = "print(a)"
        text_output, img_list = sb.execute_code(multi_plot_code)
        print(text_output)

        text_output2, img_list2 = sb.execute_code(plot_test2)
        print(text_output2, len(img_list2))
        # print(text_output2)