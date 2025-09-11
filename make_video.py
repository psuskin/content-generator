import os
import math
import argparse
import pygame
import numpy as np
import cv2

from challenge import run_simulation


def render_multiline_center(surface, text_lines, font, color, center_x, top_y, line_spacing=10):
    y = top_y
    for line in text_lines:
        surf = font.render(line, True, color)
        rect = surf.get_rect()
        rect.midtop = (center_x, y)
        surface.blit(surf, rect)
        y += rect.height + line_spacing
    return y


def record_video(
    sides: int,
    large_width: int = 500,
    small_width: int = 15,
    speed: float = 300.0,
    sim_fps: int = 120,
    out_path: str | None = None,
    video_w: int = 1080,
    video_h: int = 1920,
    sim_size: int = 1080,
):
    # Output path default
    day = sides - 2
    # Ensure output folder 'days' exists
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "days")
    os.makedirs(out_dir, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(out_dir, f"day_{day}.mp4")
    else:
        # If relative, put inside days/
        if not os.path.isabs(out_path):
            out_path = os.path.join(out_dir, out_path)

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(sim_fps), (video_w, video_h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer. Ensure the codec is available.")

    # Create a Pygame surface to compose the final frame (1080x1920)
    frame_surface = pygame.Surface((video_w, video_h))

    # Fonts: Comic Sans for top/bottom captions
    try:
        title_font = pygame.font.SysFont("Comic Sans MS", 72)
        info_font = pygame.font.SysFont("Comic Sans MS", 56)
    except Exception:
        # Fallback if Comic Sans not found
        title_font = pygame.font.SysFont(None, 72)
        info_font = pygame.font.SysFont(None, 56)

    # Static title text content
    title_lines = [
        f"Day {day}",
        "Adding one side each",
        "day until it's a circle",
    ]

    # Where to place the simulation on the 1080x1920 canvas
    sim_dest_x = (video_w - sim_size) // 2
    # Move the simulation (and title – which is positioned relative to sim_dest_y) down by the same offset
    y_offset = 160  # px
    sim_dest_y = (video_h - sim_size) // 2 + y_offset

    # Full-frame Matrix background (pure black base)
    bg_surface = pygame.Surface((video_w, video_h), pygame.SRCALPHA)
    bg_surface.fill((0, 0, 0, 0))
    try:
        bg_font = pygame.font.SysFont("Consolas", 22)
    except Exception:
        bg_font = pygame.font.SysFont(None, 22)
    glyph_w_bg, glyph_h_bg = bg_font.size("0")
    n_cols_bg = max(1, video_w // max(1, glyph_w_bg))
    matrix_chars = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    head_color = (100, 210, 100)
    glyph_cache_bg = {ch: bg_font.render(ch, True, head_color) for ch in set(matrix_chars)}
    import random
    col_y_bg = [random.uniform(-glyph_h_bg * 20.0, 0.0) for _ in range(n_cols_bg)]
    col_speed_bg = [random.uniform(90.0, 180.0) for _ in range(n_cols_bg)]
    col_char_bg = [random.choice(matrix_chars) for _ in range(n_cols_bg)]
    bg_surface.set_alpha(80)

    # Collect collision timestamps to later add audio clicks
    collision_times = []

    # Auto-scale: make the large n-gon take most of the simulation width.
    # Then scale the small n-gon and speed by the same factor, and finally double the speed
    # so generation finishes more quickly by default.
    auto_large_width = int(sim_size * 0.96)
    scale_factor = auto_large_width / 500.0  # reference was 500px
    large_width = auto_large_width
    small_width = int(round(small_width * scale_factor))
    speed_used = speed * scale_factor * 1.0

    # State in closure to stop when finished
    state = {"finished": False, "tail": 0}

    frame_counter = 0

    def on_frame(canvas: pygame.Surface, current_small_px: float, current_large_px: float):
        # Compose final frame
        frame_surface.fill((0, 0, 0))  # pure black background
        # Update background animation
        dt_bg = 1.0 / float(sim_fps)
        bg_surface.fill((0, 0, 0, 64), special_flags=pygame.BLEND_RGBA_SUB)
        for i in range(n_cols_bg):
            y_old = col_y_bg[i]
            col_y_bg[i] += col_speed_bg[i] * dt_bg
            y_new = col_y_bg[i]
            if int(y_new / max(1, glyph_h_bg)) != int(y_old / max(1, glyph_h_bg)):
                col_char_bg[i] = random.choice(matrix_chars)
            x = i * glyph_w_bg
            glyph = glyph_cache_bg[col_char_bg[i]]
            bg_surface.blit(glyph, (x, int(y_new)))
            if y_new > video_h + glyph_h_bg * 6:
                col_y_bg[i] = random.uniform(-glyph_h_bg * 20.0, 0.0)
                col_speed_bg[i] = random.uniform(90.0, 180.0)
                col_char_bg[i] = random.choice(matrix_chars)
        frame_surface.blit(bg_surface, (0, 0))
        # Scale simulation canvas to target sim_size (square) and center
        sim_scaled = pygame.transform.smoothscale(canvas, (sim_size, sim_size))
        frame_surface.blit(sim_scaled, (sim_dest_x, sim_dest_y))

        # Top title (Comic Sans), centered above simulation (raised a bit)
        render_multiline_center(
            frame_surface,
            title_lines,
            title_font,
            (255, 255, 255),
            center_x=video_w // 2,
            top_y=max(40, sim_dest_y - 320),
            line_spacing=8,
        )

        # Bottom info: current smaller px (rounded) / larger px (int)
        small_rounded = int(round(current_small_px))
        large_int = int(round(current_large_px))
        info_text = f"{small_rounded}px / {large_int}px"
        info_surf = info_font.render(info_text, True, (255, 255, 255))
        info_rect = info_surf.get_rect()
        info_rect.midtop = (video_w // 2, sim_dest_y + sim_size + 20)
        frame_surface.blit(info_surf, info_rect)

        # Convert surface to bytes (RGB) efficiently and write via OpenCV
        raw = pygame.image.tostring(frame_surface, "RGB")
        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((video_h, video_w, 3))  # pygame string is row-major RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        # Lightweight progress every ~2 seconds
        nonlocal frame_counter
        frame_counter += 1
        if frame_counter % max(1, (sim_fps * 2)) == 0:
            print(f"Rendered {frame_counter} frames (~{frame_counter/sim_fps:.1f}s)")

        # Stop condition: when small reaches (or exceeds) large
        if not state["finished"] and small_rounded >= large_int:
            state["finished"] = True
            state["tail"] = 6  # write a few extra frames for a clean ending
        elif state["finished"]:
            state["tail"] -= 1
            if state["tail"] <= 0:
                # Ask the simulation loop to quit gracefully
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    # Run the simulation headless; it will call on_frame every frame
    # Provide collision callback to gather timestamps
    def _on_collision(t):
        collision_times.append(float(t))

    run_simulation(
        sides=sides,
        large_width=large_width,
        small_width=small_width,
        start_speed=speed_used,
        window_size=(sim_size, sim_size),
        fps=sim_fps,
        frame_callback=on_frame,
        headless=True,
        render_scale_override=2.0,  # lower supersampling for faster headless rendering
        enable_matrix=False,
        collision_callback=_on_collision,
    )

    writer.release()

    import shutil, tempfile, subprocess, wave
    ffmpeg = shutil.which("ffmpeg")
    thock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thock.mp3")
    if ffmpeg and os.path.exists(thock_path) and collision_times:
        # Work in a temp directory and convert click to a known PCM format
        with tempfile.TemporaryDirectory() as workdir:
            click_wav = os.path.join(workdir, "click.wav")
            # Force PCM s16le, 44.1kHz, stereo for consistent mixing
            subprocess.run([ffmpeg, "-y", "-i", thock_path, "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", click_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Read click wav into numpy array
            with wave.open(click_wav, "rb") as wf:
                n_channels = wf.getnchannels()
                sr = wf.getframerate()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                click_bytes = wf.readframes(nframes)
            if sampwidth != 2:
                # Should not happen due to -acodec pcm_s16le
                raise RuntimeError("Unexpected click WAV sample width; expected 16-bit")
            click_arr = np.frombuffer(click_bytes, dtype=np.int16).reshape(-1, n_channels).astype(np.int32)

            # Determine output audio length from rendered frames (plus a small tail)
            duration_s = max(0.0, frame_counter / float(sim_fps))
            total_samples = int(math.ceil((duration_s + 0.25) * sr))
            total_samples = max(total_samples, click_arr.shape[0])

            # Mix clicks into an int32 accumulator to avoid overflow during summation
            mix_acc = np.zeros((total_samples, n_channels), dtype=np.int32)
            # Apply gain similar to prior ffmpeg volume=0.7
            gain = 0.7
            click_scaled = (click_arr * gain).astype(np.int32)
            clen = click_scaled.shape[0]
            for t in collision_times:
                start = int(t * sr)
                if start >= total_samples:
                    continue
                end = min(total_samples, start + clen)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                mix_acc[start:end, :] += click_scaled[:seg_len, :]

            # Clip to int16 and write WAV
            mix_int16 = np.clip(mix_acc, -32768, 32767).astype(np.int16)
            mixed_wav = os.path.join(workdir, "mixed.wav")
            with wave.open(mixed_wav, "wb") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(mix_int16.tobytes())

            # Mux into a new MP4 without re-encoding video; keep shortest
            out_mux = os.path.splitext(out_path)[0] + "_with_audio.mp4"
            cmd = [ffmpeg, "-y", "-i", out_path, "-i", mixed_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest", out_mux]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out_path = out_mux
            print(f"Audio muxed: {len(collision_times)} clicks -> {out_path}")
    else:
        if not ffmpeg:
            print("Skipping audio mux: ffmpeg not found in PATH")
        elif not os.path.exists(thock_path):
            print("Skipping audio mux: thock.mp3 not found next to script")
        elif not collision_times:
            print("Skipping audio mux: no collision timestamps recorded")
            
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render a 1080x1920 vertical video of the n-gon simulation with captions.")
    parser.add_argument("--sides", type=int, required=True, help="Number of sides for both polygons (>=3). Day is computed as sides-2.")
    parser.add_argument("--large", type=int, default=500, help="Large polygon width in pixels (across)")
    parser.add_argument("--small", type=int, default=15, help="Small polygon starting width in pixels (across)")
    parser.add_argument("--speed", type=float, default=300.0, help="Initial speed to the right in px/s")
    parser.add_argument("--fps", type=int, default=120, help="Video and simulation FPS (high-fps recommended)")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path (default: day_{sides-2}.mp4)")
    args = parser.parse_args()

    sides = max(3, args.sides)
    pygame.init()
    try:
        out_path = record_video(
            sides=sides,
            large_width=args.large,
            small_width=args.small,
            speed=args.speed,
            sim_fps=args.fps,
            out_path=args.out,
        )
        print(f"Saved: {out_path}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
