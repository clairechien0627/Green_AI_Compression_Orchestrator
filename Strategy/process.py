"""
Process isolation：在獨立的 spawn 子進程中執行壓縮/評估函式，
確保子進程結束後 OS 保證 100% 釋放 VRAM，避免長時間跑多個 trial 時記憶體殘留。
"""

import gc
import logging
import multiprocessing as mp
import signal
import sys
import time
import traceback

import torch

logger = logging.getLogger("Strategy.process")


def _worker(queue, func, *args, **kwargs):
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available() else 0.0
        )
        queue.put({"status": "success", "result": result, "peak_vram_mb": peak_mb})
    except Exception as e:
        queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_isolated(func, *args, **kwargs):
    """
    在獨立的 spawn 子進程中執行函式。
    子進程結束後 OS 保證 100% 釋放所有 VRAM，根本解決記憶體殘留問題。
    若主進程收到 kill signal 或發生例外，會先確保子進程被終止，避免留下孤兒 GPU 進程。

    Returns:
        (result, peak_vram_mb): 函式回傳值 + 子進程內的 GPU peak（MB）
    """
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(queue, func) + args, kwargs=kwargs)
    p.start()

    def cleanup_child(signum, frame):
        logger.warning(f"⚠️ Main process received kill signal ({signum})! Terminating child (PID: {p.pid})...")
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()  # Force kill if it refuses to terminate
        sys.exit(1)

    # Temporarily override standard kill signals to protect the child process
    original_sigterm = signal.signal(signal.SIGTERM, cleanup_child)
    original_sigint = signal.signal(signal.SIGINT, cleanup_child)

    try:
        p.join()
    except Exception:
        logger.warning(f"⚠️ Exception in main process! Terminating child (PID: {p.pid})...")
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
        raise
    finally:
        # Restore normal signal behavior once the isolated function finishes
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)

    if not queue.empty():
        res = queue.get()
        if res["status"] == "success":
            return res["result"], res.get("peak_vram_mb", 0.0)
        else:
            raise RuntimeError(f"Isolated process failed:\n{res['error']}\n{res['traceback']}")
    else:
        raise RuntimeError("子進程異常終止（可能是 OOM 被 OS 砍掉）。")


def run_isolated_oom_retry(func, *args, oom_wait_minutes: int = 10, **kwargs):
    """
    呼叫 run_isolated，若偵測到 OOM（進程被 OS 殺死或 CUDA OOM）則
    等待 oom_wait_minutes 分鐘後無限重試，直到成功為止。
    """
    attempt = 0
    while True:
        try:
            return run_isolated(func, *args, **kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            is_oom = (
                "process died unexpectedly" in err_msg or
                "out of memory" in err_msg or
                "outofmemoryerror" in err_msg or
                "cuda out of memory" in err_msg
            )
            if not is_oom:
                raise
            attempt += 1
            wait_sec = oom_wait_minutes * 60
            logger.warning(
                f"⚠️  OOM 偵測到（第 {attempt} 次），等待 {oom_wait_minutes} 分鐘後重試... "
                f"(原因: {str(e)[:120]})"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(wait_sec)
            logger.info(f"OOM 等待結束，開始第 {attempt + 1} 次嘗試...")
