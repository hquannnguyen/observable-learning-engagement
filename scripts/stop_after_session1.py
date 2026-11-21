#!/usr/bin/env python3
"""Watcher script to stop batch pose extraction right after session_001 completes.

Monitors the log and process status of task-1935 (PID 44632).
Once session_001 finishes and is fully saved to CSV & Parquet, terminates
the process to prevent session_002 from starting.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = Path(r"C:\Users\pc\.gemini\antigravity\brain\eb9cc1ab-a6f0-412b-a0bf-f59ff9ac9291\.system_generated\tasks\task-1935.log")
TARGET_PID = 44632
STATUS_FILE = ROOT / "data" / "poses" / "session_001_yolo11_extracted.status"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [watcher_stop_session1]: %(message)s",
)
logger = logging.getLogger("watcher")


def is_pid_running(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in proc.stdout
    except Exception:
        return False


def main() -> None:
    logger.info("Watcher started for PID %d.", TARGET_PID)
    logger.info("Monitoring %s for session_001 completion...", LOG_FILE)

    while True:
        time.sleep(10)

        # 1. Check if extraction process is still running
        if not is_pid_running(TARGET_PID):
            logger.info("Target PID %d is no longer active. Watcher finished.", TARGET_PID)
            break

        # 2. Check if session_001 has finished in logs
        if LOG_FILE.is_file():
            try:
                content = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
                if "FINISHED session_001" in content or "[1/2] FINISHED session_001" in content:
                    logger.info("Detected 'FINISHED session_001' in logs!")
                    # Wait 5 seconds to guarantee Parquet file is completely flushed and closed
                    time.sleep(5)

                    logger.info("Terminating process tree for PID %d to stop before session_002 starts...", TARGET_PID)
                    kill_res = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(TARGET_PID)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    logger.info("taskkill result: %s", kill_res.stdout.strip() or kill_res.stderr.strip())

                    # Write completion status file
                    STATUS_FILE.write_text(
                        f"session_001 YOLO11 HD extraction completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                        f"Process stopped per user request before session_002.\n",
                        encoding="utf-8",
                    )
                    logger.info("Status recorded at %s. Done!", STATUS_FILE)
                    break
            except Exception as e:
                logger.warning("Error checking log file: %s", e)


if __name__ == "__main__":
    main()
