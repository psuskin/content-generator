import math
import argparse
import pygame
import os
import colorsys
import random


def regular_polygon_vertices(n, center, radius, rotation=0.0):
	cx, cy = center
	return [
		(
			cx + radius * math.cos(rotation + 2 * math.pi * i / n),
			cy + radius * math.sin(rotation + 2 * math.pi * i / n),
		)
		for i in range(n)
	]


def edge_normals_ccw(vertices, center):
	"""Return list of (p0, p1, n_out) for each edge with outward unit normal (pointing away from center)."""
	m = len(vertices)
	cx, cy = center
	edges = []
	for i in range(m):
		p0 = vertices[i]
		p1 = vertices[(i + 1) % m]
		ex = p1[0] - p0[0]
		ey = p1[1] - p0[1]
		# Two candidate normals
		nx1, ny1 = ey, -ex
		nx2, ny2 = -ey, ex
		# Choose outward by testing against vector to center
		mx, my = (p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5
		to_center_x, to_center_y = cx - mx, cy - my
		# Outward normal should have negative dot with vector to center
		if nx1 * to_center_x + ny1 * to_center_y < 0:
			nx, ny = nx1, ny1
		else:
			nx, ny = nx2, ny2
		length = math.hypot(nx, ny)
		if length == 0:
			continue
		edges.append((p0, p1, (nx / length, ny / length)))
	return edges


def reflect_velocity(vx, vy, nx, ny):
	# Reflect v across plane with unit normal n
	dot = vx * nx + vy * ny
	rvx = vx - 2 * dot * nx
	rvy = vy - 2 * dot * ny
	return rvx, rvy


def run_simulation(
	sides=3,
	large_width=500,
	small_width=10,
	start_speed=300.0,
	window_size=(600, 600),
	fps=120,
	frame_callback=None,
	headless=False,
	render_scale_override=None,
	enable_matrix=True,
	collision_callback=None,
):
	# Optional headless mode for offline rendering without opening a window
	if headless:
		# Use dummy video driver so pygame doesn't try to open a window
		os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
	pygame.init()
	# Display scale: show more pixels while keeping proportions
	display_scale = 1.0
	screen_size = (int(window_size[0] * display_scale), int(window_size[1] * display_scale))
	screen = None
	if not headless:
		screen = pygame.display.set_mode(screen_size)
		pygame.display.set_caption("n-gon in n-gon — elastic growth sim")
	clock = pygame.time.Clock()
	font = pygame.font.SysFont(None, 24)

	# --- Rendering scale (supersampling for sharper image) ---
	render_scale = float(render_scale_override) if render_scale_override else 3.0  # allow override in headless
	internal_size = (int(window_size[0] * render_scale), int(window_size[1] * render_scale))
	canvas = pygame.Surface(internal_size)  # opaque draw target
	# Persistent trail surface (draw once, then blit each frame onto canvas)
	trail_surface = pygame.Surface(internal_size, pygame.SRCALPHA)
	trail_surface.fill((0, 0, 0, 0))

	# --- Matrix rain background setup (optional) ---
	matrix_surface = None
	if enable_matrix:
		matrix_surface = pygame.Surface(internal_size, pygame.SRCALPHA)
		matrix_surface.fill((0, 0, 0, 0))
		# Smaller, subtler glyphs
		matrix_font_size = max(6, int(8 * render_scale))
		try:
			matrix_font = pygame.font.SysFont("Consolas", matrix_font_size)
		except Exception:
			matrix_font = pygame.font.SysFont(None, matrix_font_size)
		glyph_w, glyph_h = matrix_font.size("0")
		n_cols = max(1, internal_size[0] // max(1, glyph_w))
		matrix_chars = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		# Pre-render glyphs for performance (bright head color, tails fade via surface fade)
		# Dimmer green for subtle look
		head_color = (100, 210, 100)
		glyph_cache = {ch: matrix_font.render(ch, True, head_color) for ch in set(matrix_chars)}
		# Column state
		col_y = [random.uniform(-glyph_h * 20.0, 0.0) for _ in range(n_cols)]
		# Slower drift for subtle motion
		col_speed = [random.uniform(50.0 * render_scale, 120.0 * render_scale) for _ in range(n_cols)]
		col_char = [random.choice(matrix_chars) for _ in range(n_cols)]
		# Global low opacity for subtlety
		matrix_surface.set_alpha(70)

	# --- Audio setup (skip in headless) ---
	clack_sound = None
	if not headless:
		try:
			pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
			script_dir = os.path.dirname(os.path.abspath(__file__))
			sound_path = os.path.join(script_dir, "thock.wav")
			clack_sound = pygame.mixer.Sound(sound_path)
			# Optional: adjust volume slightly if needed
			clack_sound.set_volume(0.6)
		except Exception:
			clack_sound = None

	cx, cy = internal_size[0] // 2, internal_size[1] // 2

	# Interpret width as 2 * circumradius (scale to internal pixels)
	R_large = (large_width / 2.0) * render_scale
	r_small = (small_width / 2.0) * render_scale

	# Track elapsed simulated time for event callbacks
	elapsed_s = 0.0

	# Orientation: point a vertex upward (rotation = -pi/2)
	rotation = -math.pi / 2.0

	# Precompute unit polygon offsets and large polygon geometry
	unit_offsets = [
		(
			math.cos(rotation + 2.0 * math.pi * i / sides),
			math.sin(rotation + 2.0 * math.pi * i / sides),
		)
		for i in range(sides)
	]
	# Static large polygon (float) and edges with outward normals
	large_poly_float = [(cx + R_large * ox, cy + R_large * oy) for (ox, oy) in unit_offsets]
	edges_pre = edge_normals_ccw(large_poly_float, (cx, cy))
	# Support dot for unit small polygon along each outward normal
	support_dot_max_unit = []
	for (_e0, _e1, n_out) in edges_pre:
		nx, ny = n_out
		max_dot = max(ox * nx + oy * ny for (ox, oy) in unit_offsets)
		support_dot_max_unit.append(max_dot)
	# Precompute draw vertices for large polygon with straightened bottom row
	y_max_f = max(y for (_, y) in large_poly_float)
	y_max_row = int(round(y_max_f))
	large_verts_draw = []
	flat_tol = 0.75 * render_scale
	for x, y in large_poly_float:
		rx = int(round(x))
		ry = y_max_row if abs(y - y_max_f) < flat_tol else int(round(y))
		large_verts_draw.append((rx, ry))

	# Small polygon state (no rotation dynamics for simplicity)
	pos_x, pos_y = float(cx), float(cy)
	# Initial direction: a few degrees above straight right (upward in screen space)
	_angle_deg = 7.0
	_angle = math.radians(_angle_deg)
	vel_x = (start_speed * render_scale) * math.cos(_angle)
	# Negative vy to go upwards on the screen (y grows downward in Pygame)
	vel_y = -(start_speed * render_scale) * math.sin(_angle)

	running = True
	finished = False  # stop physics once small polygon reaches large size
	collision_cooldown = 0  # frames to avoid multiple counts on same hit cluster

	# --- Rainbow trail setup ---
	trail = []  # kept for potential future use
	trail_spacing = 5.0 * render_scale  # slightly more frequent drops
	trail_max = 2000  # cap to avoid perf/memory blowup (not critical with persistent surface)
	last_trail_x, last_trail_y = pos_x, pos_y
	last_trail_r = r_small
	trail_hue = 0.0
	hue_step = 0.007  # slightly reduced to keep gradient smooth with more drops
	# Outline width (screen px) -> internal px
	outline_screen_w = 3
	outline_w = max(1, int(round(outline_screen_w * render_scale)))
	# No extra shrink; we'll place clones so the outer outline touches the boundary exactly
	outline_pad_internal = 0.0

	# --- Finish snapping: when nearly full size, snap to exact size and center for a clean climax ---
	finish_snap_screen = 10.0  # px tolerance in screen space (slightly looser to avoid stalling)
	finish_snap_internal = finish_snap_screen * render_scale
	center_snap_screen = 1.0  # px tolerance in screen space (position tolerance)
	center_snap_internal = center_snap_screen * render_scale

	# Helper: maximum feasible radius at (x,y) without crossing any edge
	def compute_r_max(x, y):
		rmax = float('inf')
		for idx, (e0, _e1, n_out) in enumerate(edges_pre):
			nx, ny = n_out
			d_center = (x - e0[0]) * nx + (y - e0[1]) * ny
			s = support_dot_max_unit[idx]
			if s <= 0:
				continue
			rmax_i = -d_center / s
			if rmax_i < rmax:
				rmax = rmax_i
		return max(0.0, rmax)

	# If already at or above target size, clamp and freeze
	if r_small >= R_large:
		r_small = R_large
		vel_x = 0.0
		vel_y = 0.0
		finished = True

	while running:
		# Fixed dt in headless mode for deterministic, smooth offline rendering
		if headless:
			dt = 1.0 / float(fps)
		else:
			dt = clock.tick(fps) / 1000.0

		# Track elapsed simulated time
		elapsed_s += dt

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		# Update Matrix rain background (if enabled)
		if enable_matrix and matrix_surface is not None:
			# Gentle fade of previous frame to create trails
			matrix_surface.fill((0, 0, 0, 64), special_flags=pygame.BLEND_RGBA_SUB)
			for i in range(n_cols):
				y_old = col_y[i]
				col_y[i] += col_speed[i] * dt
				y_new = col_y[i]
				# Change char when stepping into a new row
				if int(y_new / max(1, glyph_h)) != int(y_old / max(1, glyph_h)):
					col_char[i] = random.choice(matrix_chars)
				x = i * glyph_w
				glyph = glyph_cache[col_char[i]]
				matrix_surface.blit(glyph, (x, int(y_new)))
				# Reset when off screen
				if y_new > internal_size[1] + glyph_h * 6:
					col_y[i] = random.uniform(-glyph_h * 20.0, 0.0)
					col_speed[i] = random.uniform(50.0 * render_scale, 120.0 * render_scale)
					col_char[i] = random.choice(matrix_chars)

		# Integrate position with continuous-time collision resolution for straight trajectories
		if not finished:
			# Tiny pre-step correction if marginally outside due to numeric drift
			max_pen0 = 0.0
			hit_n0 = None
			for idx, (e0, _e1, n_out) in enumerate(edges_pre):
				nx, ny = n_out
				d_center = (pos_x - e0[0]) * nx + (pos_y - e0[1]) * ny
				worst = d_center + r_small * support_dot_max_unit[idx]
				if worst > max_pen0:
					max_pen0 = worst
					hit_n0 = n_out
			if max_pen0 > 0.0 and hit_n0 is not None:
				nx, ny = hit_n0
				pos_x -= nx * max_pen0
				pos_y -= ny * max_pen0

			# Swept step: find earliest collision time-of-impact within dt
			t_remaining = dt
			collisions_in_step = 0
			# Limit iterations to avoid infinite loops on corner cases
			for _sweep_iter in range(8):
				# Find earliest positive t where we would violate a constraint
				t_hit = None
				hit_normal = None
				for idx, (e0, _e1, n_out) in enumerate(edges_pre):
					nx, ny = n_out
					d_center = (pos_x - e0[0]) * nx + (pos_y - e0[1]) * ny
					phi = d_center + r_small * support_dot_max_unit[idx]
					# If already outside, ignore here (pre-step correction should have fixed it)
					# Compute relative rate along normal
					dv = vel_x * nx + vel_y * ny
					if dv <= 0.0:
						continue
					# We are inside when phi <= 0. Impact when phi reaches 0 from below
					# t = -phi / dv
					t = -phi / dv
					if 0.0 <= t <= t_remaining:
						if t_hit is None or t < t_hit:
							t_hit = t
							hit_normal = n_out
				# No collision within remaining time: advance and finish
				if t_hit is None:
					pos_x += vel_x * t_remaining
					pos_y += vel_y * t_remaining
					break
				# Advance to impact, reflect, continue with remaining time
				pos_x += vel_x * t_hit
				pos_y += vel_y * t_hit
				nx, ny = hit_normal
				vel_x, vel_y = reflect_velocity(vel_x, vel_y, nx, ny)
				# Extra trail drop exactly at collision point, projected to touch the boundary
				pxc, pyc = pos_x, pos_y
				r_draw_c = max(1.0, r_small - outline_pad_internal)
				r_eff_c = r_draw_c + (outline_w * 0.5)
				for _ in range(8):
					max_pen_c = 0.0
					hit_n_c = None
					for idxc, (e0c, _e1c, n_outc) in enumerate(edges_pre):
						nxc, nyc = n_outc
						d_center_c = (pxc - e0c[0]) * nxc + (pyc - e0c[1]) * nyc
						worst_c = d_center_c + r_eff_c * support_dot_max_unit[idxc]
						if worst_c > max_pen_c:
							max_pen_c = worst_c
							hit_n_c = n_outc
					if not (max_pen_c > 0.0 and hit_n_c is not None):
						break
					nxc, nyc = hit_n_c
					pxc -= nxc * max_pen_c
					pyc -= nyc * max_pen_c
				# Final clamp w/o epsilon
				max_pen_c = 0.0
				hit_n_c = None
				for idxc, (e0c, _e1c, n_outc) in enumerate(edges_pre):
					nxc, nyc = n_outc
					d_center_c = (pxc - e0c[0]) * nxc + (pyc - e0c[1]) * nyc
					worst_c = d_center_c + r_eff_c * support_dot_max_unit[idxc]
					if worst_c > max_pen_c:
						max_pen_c = worst_c
						hit_n_c = n_outc
				if max_pen_c > 0.0 and hit_n_c is not None:
					nxc, nyc = hit_n_c
					pxc -= nxc * max_pen_c
				# Draw the collision clone
				h = (trail_hue % 1.0)
				rf, gf, bf = colorsys.hsv_to_rgb(h, 0.85, 0.95)
				color_c = (int(rf * 255), int(gf * 255), int(bf * 255))
				poly_c = [(pxc + r_draw_c * ox, pyc + r_draw_c * oy) for (ox, oy) in unit_offsets]
				verts_c = [(int(round(x)), int(round(y))) for x, y in poly_c]
				pygame.draw.polygon(trail_surface, color_c, verts_c)
				pygame.draw.polygon(trail_surface, (255, 255, 255), verts_c, width=outline_w)
				trail_hue += hue_step
				# Nudge slightly inside to avoid immediate re-hit due to precision
				pos_x -= nx * 1e-6
				pos_y -= ny * 1e-6
				t_remaining -= t_hit
				collisions_in_step += 1
				if t_remaining <= 0.0:
					break
			# Growth and sound once per frame on first collision cluster
			if collisions_in_step > 0 and collision_cooldown == 0:
				if clack_sound is not None:
					clack_sound.play()
				# External collision event callback (e.g., audio schedule)
				if collision_callback is not None:
					try:
						collision_callback(elapsed_s)
					except Exception:
						pass
				# Compute feasible target radius at current center
				r_max_cur = compute_r_max(pos_x, pos_y)
				target_r = min(R_large, r_max_cur)
				# Compute remaining width (diameter) in screen pixels to keep units consistent
				remaining_width_screen = max(0.0, (2.0 * R_large - 2.0 * r_small) / render_scale)
				# Pace growth roughly proportional to remaining size (100 steps across the width)
				inc_width_screen = max(1.0, math.ceil(remaining_width_screen / 100.0))
				# Convert width increment (screen) to radius increment (internal)
				inc_radius = (inc_width_screen * render_scale) * 0.5
				inc_radius = min(inc_radius, max(0.0, target_r - r_small))
				r_small += inc_radius
				# Finish: if size is within tolerance, snap to center and finalize
				if (R_large - r_small) <= finish_snap_internal:
					# Snap position to center to ensure exact fit
					pos_x = float(cx)
					pos_y = float(cy)
					r_small = R_large
					vel_x = 0.0
					vel_y = 0.0
					finished = True
				collision_cooldown = int(max(1, fps // 30))

		# Build polygons (initial for collision checks)
		def build_small_poly(xc, yc, rad):
			return [(xc + rad * ox, yc + rad * oy) for (ox, oy) in unit_offsets]

		small_poly = build_small_poly(pos_x, pos_y, r_small)

		if collision_cooldown > 0:
			collision_cooldown -= 1

		# Finalization: if size is within tolerance, snap to center and finish
		if not finished:
			if (R_large - r_small) <= finish_snap_internal:
				pos_x = float(cx)
				pos_y = float(cy)
				r_small = R_large
				vel_x = 0.0
				vel_y = 0.0
				finished = True

		# Rebuild small polygon after any updates (for accurate drawing)
		small_poly = build_small_poly(pos_x, pos_y, r_small)

		# Drop a trail clone at regular distance intervals (background only)
		if not finished:
			dx = pos_x - last_trail_x
			dy = pos_y - last_trail_y
			dist = math.hypot(dx, dy)
			if dist >= trail_spacing:
				# Emit multiple evenly spaced clones if we've moved more than one spacing
				steps = int(dist // trail_spacing)
				ux = dx / dist
				uy = dy / dist
				for s in range(1, steps + 1):
					px = last_trail_x + ux * trail_spacing * s
					py = last_trail_y + uy * trail_spacing * s
					# Interpolate radius across the segment to track growth smoothly
					tseg = s / float(steps)
					r_at = last_trail_r + (r_small - last_trail_r) * tseg
					# Shrink so that outline stays inside the boundary
					r_draw = max(1.0, r_at - outline_pad_internal)
					# Project sample so that the outer outline touches the boundary.
					# Use an effective radius that includes half the outline width.
					r_eff = r_draw + (outline_w * 0.5)
					for _ in range(8):
						max_pen_s = 0.0
						hit_n_s = None
						for idx, (e0s, _e1s, n_outs) in enumerate(edges_pre):
							nx, ny = n_outs
							d_center = (px - e0s[0]) * nx + (py - e0s[1]) * ny
							worst = d_center + r_eff * support_dot_max_unit[idx]
							if worst > max_pen_s:
								max_pen_s = worst
								hit_n_s = n_outs
						if not (max_pen_s > 0.0 and hit_n_s is not None):
							break
						nx, ny = hit_n_s
						px -= nx * max_pen_s
						py -= ny * max_pen_s
					# One last clamp if still slightly outside (no extra epsilon to avoid visible gap)
					max_pen_s = 0.0
					hit_n_s = None
					for idx, (e0s, _e1s, n_outs) in enumerate(edges_pre):
						nx, ny = n_outs
						d_center = (px - e0s[0]) * nx + (py - e0s[1]) * ny
						worst = d_center + r_eff * support_dot_max_unit[idx]
						if worst > max_pen_s:
							max_pen_s = worst
							hit_n_s = n_outs
					if max_pen_s > 0.0 and hit_n_s is not None:
						nx, ny = hit_n_s
						px -= nx * max_pen_s
					# Color from HSV rainbow (soft saturation/value)
					h = (trail_hue % 1.0)
					rf, gf, bf = colorsys.hsv_to_rgb(h, 0.85, 0.95)
					color = (int(rf * 255), int(gf * 255), int(bf * 255))
					poly_at = [(px + r_draw * ox, py + r_draw * oy) for (ox, oy) in unit_offsets]
					# Use round() for stable rasterization (avoids 1px slant on flat edges)
					verts_int = [(int(round(x)), int(round(y))) for x, y in poly_at]
					# Draw directly to persistent trail surface (performance)
					pygame.draw.polygon(trail_surface, color, verts_int)
					pygame.draw.polygon(trail_surface, (255, 255, 255), verts_int, width=outline_w)
					trail_hue += hue_step
				# Advance the last_trail marker to the last emitted position
				last_trail_x = last_trail_x + ux * trail_spacing * steps
				last_trail_y = last_trail_y + uy * trail_spacing * steps
				last_trail_r = r_small

		# Draw
		# Draw into internal canvas (pure black background)
		canvas.fill((0, 0, 0))
		# Background: Matrix rain
		if enable_matrix and matrix_surface is not None:
			canvas.blit(matrix_surface, (0, 0))
		# Trail in background: blit accumulated surface (newer paint over older)
		canvas.blit(trail_surface, (0, 0))

		# Large polygon outline (precomputed draw vertices)
		pygame.draw.polygon(canvas, (200, 200, 220), large_verts_draw, width=max(1, int(round(3 * render_scale))))

		# Small polygon filled (white), drawn with rounded vertices
		small_verts_draw = [(int(round(x)), int(round(y))) for x, y in small_poly]
		pygame.draw.polygon(canvas, (255, 255, 255), small_verts_draw)

		# Invoke frame callback before presenting, passing current widths in screen px
		if frame_callback is not None:
			current_small_px = 2 * r_small / render_scale
			current_large_px = 2 * R_large / render_scale
			# Provide the canvas surface; callback may copy/scale as needed
			frame_callback(canvas, current_small_px, current_large_px)

		# Present only if not headless
		if not headless and screen is not None:
			scaled = pygame.transform.smoothscale(canvas, screen_size)
			screen.blit(scaled, (0, 0))
			# HUD text (draw after scaling to keep text size consistent)
			hud = [
				f"sides: {sides}",
				f"small width: {2*r_small/render_scale:.1f}px",
				f"speed: {math.hypot(vel_x, vel_y)/render_scale:.1f}px/s",
			]
			y = 10
			for line in hud:
				surf = font.render(line, True, (240, 240, 245))
				screen.blit(surf, (10, y))
				y += 18
			pygame.display.flip()

	pygame.quit()


def main():
	parser = argparse.ArgumentParser(description="Small regular n-gon bouncing inside a larger same n-gon (elastic, growth on collisions)")
	parser.add_argument("--sides", type=int, default=3, help="Number of sides for both polygons (>=3)")
	parser.add_argument("--large", type=float, default=500.0, help="Large polygon width in pixels (across)")
	parser.add_argument("--small", type=float, default=15.0, help="Small polygon starting width in pixels (across)")
	parser.add_argument("--speed", type=float, default=300.0, help="Initial speed to the right in px/s")
	parser.add_argument("--fps", type=int, default=120, help="Target FPS")
	args = parser.parse_args()

	sides = max(3, args.sides)
	run_simulation(
		sides=sides,
		large_width=args.large,
		small_width=args.small,
		start_speed=args.speed,
		fps=args.fps,
	)


if __name__ == "__main__":
	main()

