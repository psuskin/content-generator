import os
import sys
import glob
import argparse
import cv2
import numpy as np
import shutil

try:
	from PIL import Image, ImageDraw, ImageFont
	PIL_AVAILABLE = True
except Exception:
	PIL_AVAILABLE = False


def overlay_centered_text_cv(img, text, y, font=cv2.FONT_HERSHEY_TRIPLEX, font_scale=2.2, color=(255, 255, 255), thickness=6):
	"""Centered text using OpenCV Hershey font with strong outline (fallback)."""
	h, w = img.shape[:2]
	(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
	x = (w - text_w) // 2
	# Outline
	cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 8, cv2.LINE_AA)
	cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
	# Fill
	cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def find_comic_sans_font():
	"""Best-effort search for Comic Sans on Windows; returns a path or None."""
	candidates = [
		r"C:\\Windows\\Fonts\\comicbd.ttf",  # Comic Sans MS Bold
		r"C:\\Windows\\Fonts\\comic.ttf",    # Comic Sans MS
		r"/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS_Bold.ttf",
		r"/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf",
	]
	for p in candidates:
		if os.path.exists(p):
			return p
	return None


def overlay_centered_text_pil(img_bgr, lines, top_y, font_size, line_spacing=1.15):
	"""Draw multi-line centered text using PIL with thick stroke. Modifies image in place (BGR)."""
	# Convert BGR to RGB for PIL
	rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(rgb)
	draw = ImageDraw.Draw(pil_img)
	font_path = find_comic_sans_font()
	try:
		font = ImageFont.truetype(font_path, font_size) if (PIL_AVAILABLE and font_path) else None
	except Exception:
		font = None

	# Fallback to default PIL font if Comic Sans not found
	if font is None and PIL_AVAILABLE:
		try:
			font = ImageFont.load_default()
		except Exception:
			font = None

	w, h = pil_img.size
	y = top_y
	# Choose stroke width relative to font size for a thick, social look
	stroke_width = max(4, font_size // 10)
	for i, text in enumerate(lines):
		if not PIL_AVAILABLE or font is None:
			# Fallback to OpenCV if PIL or font not available
			overlay_centered_text_cv(img_bgr, text, y, font_scale=max(1.8, font_size / 48.0))
		else:
			# Centered horizontally using bbox for accurate size with stroke
			bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
			tw = bbox[2] - bbox[0]
			th = bbox[3] - bbox[1]
			x = (w - tw) // 2
			# Stroke (outline) + fill
			draw.text((x, y - th), text, font=font, fill=(255, 255, 255),
					  stroke_width=stroke_width, stroke_fill=(0, 0, 0))
			# Advance y
			y += int(th * line_spacing)

	# Write back to BGR
	if PIL_AVAILABLE and font is not None:
		updated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
		img_bgr[:, :, :] = updated


def zoom_to_fill(img_bgr, scale=1.08, target_size=None):
	"""Zoom image around center by `scale` and center-crop/pad to original (or target) size.
	Returns BGR image with same size as target.
	"""
	h, w = img_bgr.shape[:2]
	if target_size is None:
		tw, th = w, h
	else:
		tw, th = target_size

	# Scale
	new_w = max(1, int(w * scale))
	new_h = max(1, int(h * scale))
	resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

	# Center-crop to target size
	x0 = max(0, (new_w - tw) // 2)
	y0 = max(0, (new_h - th) // 2)
	cropped = resized[y0:y0 + th, x0:x0 + tw]

	# If scaled too small (shouldn't happen with scale>1), pad to target
	if cropped.shape[0] != th or cropped.shape[1] != tw:
		pad_top = max(0, (th - cropped.shape[0]) // 2)
		pad_bottom = th - cropped.shape[0] - pad_top
		pad_left = max(0, (tw - cropped.shape[1]) // 2)
		pad_right = tw - cropped.shape[1] - pad_left
		cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

	return cropped


def process_video(input_path, output_path=None, top_text="Point Battle", bottom_text="With Funk Infernal in the background"):
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input not found: {input_path}")

	cap = cv2.VideoCapture(input_path)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open input: {input_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Rotate 90 degrees clockwise -> output size is (height, width)
	out_w, out_h = height, width

	if output_path is None:
		base, ext = os.path.splitext(input_path)
		output_path = f"{base}_vertical.mp4"

	# Ensure output directory exists
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
	if not writer.isOpened():
		cap.release()
		raise RuntimeError(f"Could not open output: {output_path}")

	# Choose font sizes based on output width for consistent look
	# 1080x1920 -> ~88px top, ~80px bottom; scale proportionally
	top_font_px = max(40, int(88 * (out_w / 1080.0)))
	bottom_font_px = max(36, int(80 * (out_w / 1080.0)))

	# Positions: keep text away from circle area; move top text further down
	top_y = int(out_h * 0.2)
	# Move bottom text a bit higher than before (was ~93%)
	bottom_y = int(out_h * 0.8)

	# Prepare bottom text with a manual line break between "in" and "the"
	def split_bottom_text(txt: str):
		# Prefer splitting at " in the " exactly once
		key = " in the "
		if key in txt:
			left, right = txt.split(key, 1)
			return [f"{left} in", f"the {right}"]
		# Fallback heuristic: find " in the" without trailing space
		if " in the" in txt:
			idx = txt.find(" in the")
			left = txt[: idx + 3]  # include " in"
			right = txt[idx + 4 :].lstrip()  # start at "the ..."
			return [left, right]
		# Otherwise, split in half as a safe fallback
		words = txt.split()
		mid = len(words) // 2 if len(words) > 1 else 1
		return [" ".join(words[:mid]), " ".join(words[mid:])]

	bottom_lines = split_bottom_text(bottom_text)

	processed = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Rotate 90 degrees clockwise and apply slight center zoom to fill width
		rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		zoom_factor = 1.08
		frame_bgr = zoom_to_fill(rotated, scale=zoom_factor, target_size=(out_w, out_h))

		# Overlay texts (prefer PIL Comic Sans if available)
		if PIL_AVAILABLE:
			overlay_centered_text_pil(frame_bgr, [top_text], top_y, top_font_px)
			overlay_centered_text_pil(frame_bgr, bottom_lines, bottom_y, bottom_font_px)
		else:
			# Fallback to OpenCV fonts with heavier outline
			overlay_centered_text_cv(frame_bgr, top_text, top_y, font_scale=2.2 * (out_w / 1080.0))
			# Draw bottom as two lines using approximate spacing
			line_gap = int(bottom_font_px * 1.15)
			y = bottom_y
			for i, line in enumerate(bottom_lines):
				overlay_centered_text_cv(frame_bgr, line, y, font_scale=2.0 * (out_w / 1080.0))
				y += line_gap

		writer.write(frame_bgr)
		processed += 1

	writer.release()
	cap.release()

	# Basic sanity check
	if processed == 0:
		raise RuntimeError("No frames processed; check input file.")

	return output_path


def find_latest_recording(recordings_dir="recordings"):
	os.makedirs(recordings_dir, exist_ok=True)
	files = sorted(
		glob.glob(os.path.join(recordings_dir, "*.mp4")),
		key=lambda p: os.path.getmtime(p),
		reverse=True,
	)
	return files[0] if files else None


def list_recordings(recordings_dir="recordings"):
	patterns = [os.path.join(recordings_dir, "*.mp4")]
	files = []
	for pat in patterns:
		files.extend(glob.glob(pat))
	# Exclude files in subfolders like 'used' and 'transposed'
	rec_abs = os.path.abspath(recordings_dir)
	files = [f for f in files if os.path.abspath(os.path.dirname(f)) == rec_abs]
	# Sort by mtime ascending for deterministic order
	files.sort(key=lambda p: os.path.getmtime(p))
	return files


def ensure_dir(path):
	os.makedirs(path, exist_ok=True)
	return path


def unique_dest_path(dir_path, base_name):
	"""Return a unique destination path under dir_path for base_name."""
	dest = os.path.join(dir_path, base_name)
	if not os.path.exists(dest):
		return dest
	name, ext = os.path.splitext(base_name)
	i = 1
	while True:
		alt = os.path.join(dir_path, f"{name}_{i}{ext}")
		if not os.path.exists(alt):
			return alt
		i += 1


def process_all_recordings(recordings_dir="recordings", top_text="Point Battle", bottom_text="With Funk Infernal in the background"):
	recordings_dir = os.path.abspath(recordings_dir)
	used_dir = ensure_dir(os.path.join(recordings_dir, "used"))
	transposed_dir = ensure_dir(os.path.join(recordings_dir, "transposed"))

	files = list_recordings(recordings_dir)
	if not files:
		print("No recordings found to process.")
		return 0

	processed = 0
	for inp in files:
		try:
			base_name = os.path.basename(inp)
			out_path = unique_dest_path(transposed_dir, base_name)
			# Write transposed output directly to transposed folder
			process_video(inp, out_path, top_text, bottom_text)
			# Move original to used folder (ensure unique name)
			used_path = unique_dest_path(used_dir, base_name)
			shutil.move(inp, used_path)
			print(f"Processed: {base_name} -> transposed/{os.path.basename(out_path)}; original -> used/{os.path.basename(used_path)}")
			processed += 1
		except Exception as e:
			print(f"Error processing {inp}: {e}")
			continue

	return processed


def main():
	parser = argparse.ArgumentParser(description="Rotate simulation recordings to 9:16 and add social captions.")
	parser.add_argument("input", nargs="?", help="Input MP4 file. If omitted, the newest file in 'recordings' is used (unless --batch).")
	parser.add_argument("--output", "-o", help="Output MP4 path. Defaults to '<input>_vertical.mp4'.")
	parser.add_argument("--top", default="Point Battle", help="Top caption text.")
	parser.add_argument("--bottom", default="With Funk Infernal in the background", help="Bottom caption text.")
	parser.add_argument("--batch", action="store_true", help="Process all MP4 files in the 'recordings' folder.")
	parser.add_argument("--recordings-dir", default="recordings", help="Recordings directory for --batch mode (default: recordings)")
	args = parser.parse_args()

	# Default: process all recordings if no explicit input is provided
	if args.batch or not args.input:
		count = process_all_recordings(args.recordings_dir, args.top, args.bottom)
		print(f"Batch complete. Processed {count} file(s).")
		return 0 if count >= 0 else 2
	else:
		input_path = args.input
		try:
			out = process_video(input_path, args.output, args.top, args.bottom)
			print(f"Created: {out}")
			return 0
		except Exception as e:
			print(f"Error: {e}")
			return 2


if __name__ == "__main__":
	sys.exit(main())

