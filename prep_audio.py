"""
Local audio prep. Resamples to 16 kHz mono 16-bit PCM via ffmpeg.

Usage:
    python prep_audio.py <input.wav> [--sr 16000] [--channels 1]

Writes <input>.prepped.wav next to the input.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()

    src = Path(args.audio).resolve()
    if not src.is_file():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not on PATH", file=sys.stderr)
        sys.exit(1)

    dst = src.with_suffix(".prepped.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", str(args.sr),
        "-ac", str(args.channels),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    print(f"[prep] {src.name} -> {dst.name}")
    subprocess.run(cmd, check=True)

    src_mb = src.stat().st_size / (1024 * 1024)
    dst_mb = dst.stat().st_size / (1024 * 1024)
    ratio = src_mb / dst_mb if dst_mb else 0
    print(f"[prep] in:  {src_mb:.1f} MB")
    print(f"[prep] out: {dst_mb:.1f} MB ({ratio:.1f}x smaller)")
    print(f"[prep] path: {dst}")


if __name__ == "__main__":
    main()
