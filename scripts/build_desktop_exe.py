#!/usr/bin/env python3
"""Build a lightweight native Windows `.exe` launcher (`AnnotationStudio_Desktop.exe`).

Uses Windows' built-in C# compiler (`csc.exe`, winexe target -> zero console popup)
to launch `app/desktop_studio.py` with `pythonw.exe` directly in the workspace.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")

CS_TEMPLATE = r"""
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace AnnotationStudioLauncher
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string rootDir = exeDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string scriptPath = Path.Combine(rootDir, "app", "desktop_studio.py");
            if (!File.Exists(scriptPath))
            {
                MessageBox.Show(
                    "Khong tim thay app\\desktop_studio.py trong: " + rootDir,
                    "Annotation Studio Desktop",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            string[] candidates = new string[]
            {
                @"{PYTHONW_PATH}",
                @"{PYTHON_PATH}",
                "pythonw.exe",
                "python.exe"
            };

            string chosen = null;
            foreach (string cand in candidates)
            {
                if (string.IsNullOrWhiteSpace(cand)) continue;
                if (Path.IsPathRooted(cand))
                {
                    if (File.Exists(cand)) { chosen = cand; break; }
                }
                else
                {
                    chosen = cand;
                    break;
                }
            }

            if (chosen == null)
            {
                MessageBox.Show(
                    "Khong tim thay Python runtime de chay app\\desktop_studio.py.",
                    "Annotation Studio Desktop",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = chosen;
            psi.Arguments = "\"" + scriptPath + "\"";
            psi.WorkingDirectory = rootDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;

            try
            {
                Process.Start(psi);
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Khong mo duoc Desktop Studio: " + ex.Message,
                    "Annotation Studio Desktop",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }
    }
}
"""


def build_exe(output_path: Path, csc_path: Path = DEFAULT_CSC) -> Path:
    if not csc_path.is_file():
        raise FileNotFoundError(f"Missing Windows C# compiler: {csc_path}")
    py_exe = Path(sys.executable)
    pyw_exe = py_exe.with_name("pythonw.exe")
    src = (
        CS_TEMPLATE
        .replace("{PYTHONW_PATH}", str(pyw_exe if pyw_exe.is_file() else ""))
        .replace("{PYTHON_PATH}", str(py_exe))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".cs", dir=output_path.parent, delete=False) as fh:
        cs_file = Path(fh.name)
        fh.write(src)
    try:
        cmd = [
            str(csc_path),
            "/nologo",
            "/target:winexe",
            "/optimize+",
            f"/out:{output_path}",
            str(cs_file),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        cs_file.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AnnotationStudio_Desktop.exe launcher.")
    parser.add_argument("--out", default=str(ROOT / "AnnotationStudio_Desktop.exe"))
    args = parser.parse_args()
    out = build_exe(Path(args.out))
    print(f"Built native Windows launcher: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
