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
    sim_size: int = 1000,
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
    sim_dest_y = (video_h - sim_size) // 2

    # State in closure to stop when finished
    state = {"finished": False, "tail": 0}

    frame_counter = 0

    def on_frame(canvas: pygame.Surface, current_small_px: float, current_large_px: float):
        # Compose final frame
        frame_surface.fill((0, 0, 0))
        # Scale simulation canvas to target sim_size (square) and center
        sim_scaled = pygame.transform.smoothscale(canvas, (sim_size, sim_size))
        frame_surface.blit(sim_scaled, (sim_dest_x, sim_dest_y))

        # Top title (Comic Sans), centered above simulation
        render_multiline_center(
            frame_surface,
            title_lines,
            title_font,
            (255, 255, 255),
            center_x=video_w // 2,
            top_y=max(40, sim_dest_y - 280),
            line_spacing=8,
        )

        # Bottom info: current smaller px (rounded) / larger px (int)
        small_rounded = int(round(current_small_px))
        large_int = int(round(current_large_px))
        info_text = f"{small_rounded} / {large_int}"
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
    run_simulation(
        sides=sides,
        large_width=large_width,
        small_width=small_width,
        start_speed=speed,
        window_size=(sim_size, sim_size),
        fps=sim_fps,
        frame_callback=on_frame,
    headless=True,
    render_scale_override=2.0,  # lower supersampling for faster headless rendering
    enable_matrix=False,
    )

    writer.release()
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
