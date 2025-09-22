import os
import argparse
import subprocess
import shutil


def has_audio_stream(input_path: str) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # If ffprobe is missing, assume there is audio when filename suggests so
        return "with_audio" in os.path.basename(input_path)
    try:
        # Query first audio stream
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            input_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return False


def add_waves(day: int, waves_path: str | None, volume: float, input_path: str | None, output_path: str | None) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg nicht gefunden. Bitte ffmpeg in PATH installieren.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    days_dir = os.path.join(script_dir, "days")

    # Resolve input
    if input_path:
        in_mp4 = input_path if os.path.isabs(input_path) else os.path.join(days_dir, input_path)
    else:
        # Prefer with_audio, fallback to plain
        cand1 = os.path.join(days_dir, f"day_{day}_with_audio.mp4")
        cand2 = os.path.join(days_dir, f"day_{day}.mp4")
        in_mp4 = cand1 if os.path.exists(cand1) else cand2

    if not os.path.exists(in_mp4):
        raise FileNotFoundError(f"Eingabevideo nicht gefunden: {in_mp4}")

    # Resolve waves mp3
    if waves_path is None:
        waves_path = os.path.join(script_dir, "sea-waves.mp3")
    if not os.path.isabs(waves_path):
        waves_path = os.path.join(script_dir, waves_path)
    if not os.path.exists(waves_path):
        raise FileNotFoundError(f"Waves MP3 nicht gefunden: {waves_path}")

    # Resolve output
    if output_path:
        out_mp4 = output_path if os.path.isabs(output_path) else os.path.join(days_dir, output_path)
    else:
        base = os.path.splitext(os.path.basename(in_mp4))[0]
        out_mp4 = os.path.join(days_dir, base + "_with_waves.mp4")

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    # Build ffmpeg command
    # Loop waves to cover full video, keep video stream copy, mix quietly
    input_has_audio = has_audio_stream(in_mp4)

    if input_has_audio:
        # Mix existing audio with quiet waves
        filter_complex = f"[1:a]volume={volume}[aw];[0:a][aw]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            in_mp4,
            "-stream_loop",
            "-1",
            "-i",
            waves_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            out_mp4,
        ]
    else:
        # Use only waves as audio, quieted
        filter_complex = f"[1:a]volume={volume}[aout]"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            in_mp4,
            "-stream_loop",
            "-1",
            "-i",
            waves_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            out_mp4,
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4


def main():
    parser = argparse.ArgumentParser(description="Lege sea-waves.mp3 leise als Hintergrund unter das Video eines Tages.")
    parser.add_argument("day", type=int, help="Tageszahl, z.B. 4 für day_4.mp4")
    parser.add_argument("--waves", type=str, default=None, help="Pfad zur sea-waves.mp3 (Standard: ./sea-waves.mp3)")
    parser.add_argument("--volume", type=float, default=0.08, help="Lautstärke der Wellen (0..1), Standard 0.08 — sehr leise")
    parser.add_argument("--in", dest="input_path", type=str, default=None, help="Eingabevideo überschreiben (relativ zu ./days)")
    parser.add_argument("--out", dest="output_path", type=str, default=None, help="Ausgabedatei überschreiben (relativ zu ./days)")
    args = parser.parse_args()

    out = add_waves(
        day=args.day,
        waves_path=args.waves,
        volume=args.volume,
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
