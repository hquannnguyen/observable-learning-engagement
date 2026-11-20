"""Resolve ffmpeg from the active environment or an explicit local dependency."""
import shutil
import sys
from pathlib import Path

def ffmpeg_executable():
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        local = Path(__file__).resolve().parents[2] / ".runtime-deps"
        if not local.is_dir():
            raise RuntimeError("Install requirements.txt to enable browser-compatible H.264 preview")
        sys.path.insert(0, str(local))
        from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()
