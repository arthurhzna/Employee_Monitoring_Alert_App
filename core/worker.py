# worker.py

import threading
import traceback


def submit_task_to_worker(fn, *args, daemon=True, **kwargs):
    def _run():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"[THREAD ERROR] {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=daemon)
    thread.start()
    return thread