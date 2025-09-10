import math
import argparse
import pygame
import os
import colorsys


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
):
	pygame.init()
	screen = pygame.display.set_mode(window_size)
	pygame.display.set_caption("n-gon in n-gon — elastic growth sim")
	clock = pygame.time.Clock()
	font = pygame.font.SysFont(None, 24)

	# --- Rendering scale (supersampling for sharper image) ---
	render_scale = 2.0  # internal rendering at 2x, then smooth downscale
	internal_size = (int(window_size[0] * render_scale), int(window_size[1] * render_scale))
	canvas = pygame.Surface(internal_size)  # opaque draw target
	# Persistent trail surface (draw once, then blit each frame onto canvas)
	trail_surface = pygame.Surface(internal_size, pygame.SRCALPHA)
	trail_surface.fill((0, 0, 0, 0))

	# --- Audio setup: load external thock.wav from this script's directory ---
	clack_sound = None
	try:
		pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
		script_dir = os.path.dirname(os.path.abspath(__file__))
		sound_path = os.path.join(script_dir, "thock.mp3")
		clack_sound = pygame.mixer.Sound(sound_path)
		# Optional: adjust volume slightly if needed
		clack_sound.set_volume(0.6)
	except Exception:
		clack_sound = None

	cx, cy = internal_size[0] // 2, internal_size[1] // 2

	# Interpret width as 2 * circumradius (scale to internal pixels)
	R_large = (large_width / 2.0) * render_scale
	r_small = (small_width / 2.0) * render_scale

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
	trail_spacing = 6.0 * render_scale  # internal pixels between drops
	trail_max = 2000  # cap to avoid perf/memory blowup (not critical with persistent surface)
	last_trail_x, last_trail_y = pos_x, pos_y
	last_trail_r = r_small
	trail_hue = 0.0
	hue_step = 0.008  # hue increment per drop
	# Outline width (screen px) -> internal px
	outline_screen_w = 3
	outline_w = max(1, int(round(outline_screen_w * render_scale)))
	# Shrink trail clones so outline stays inside the large polygon
	outline_pad_internal = (outline_screen_w * 0.5 + 1.0) * render_scale

	# If already at or above target size, clamp and freeze
	if r_small >= R_large:
		r_small = R_large
		vel_x = 0.0
		vel_y = 0.0
		finished = True

	while running:
		dt = clock.tick(fps) / 1000.0

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		# Integrate position (only while not finished)
		if not finished:
			pos_x += vel_x * dt
			pos_y += vel_y * dt

		# Build polygons (initial for collision checks)
		def build_small_poly(xc, yc, rad):
			return [(xc + rad * ox, yc + rad * oy) for (ox, oy) in unit_offsets]

		small_poly = build_small_poly(pos_x, pos_y, r_small)

		# Collision detection and growth only while not finished
		if not finished:
			# Support-based collision: center-plane distance + support radius
			max_pen = 0.0
			hit_normal = None
			for idx, (e0, _e1, n_out) in enumerate(edges_pre):
				nx, ny = n_out
				d_center = (pos_x - e0[0]) * nx + (pos_y - e0[1]) * ny
				worst = d_center + r_small * support_dot_max_unit[idx]
				if worst > max_pen:
					max_pen = worst
					hit_normal = n_out

			collided = max_pen > 0.0 and hit_normal is not None
			if collided:
				nx, ny = hit_normal
				# Push small polygon inside by penetration depth along inward direction
				pos_x -= nx * max_pen
				pos_y -= ny * max_pen
				# Reflect velocity (perfectly elastic)
				vel_x, vel_y = reflect_velocity(vel_x, vel_y, nx, ny)
				# Growth on collision (once per cluster of contacts) with clamp to large size
				if collision_cooldown == 0:
					# Play clack sound once per collision cluster (if audio available)
					if clack_sound is not None:
						clack_sound.play()
					current_small_width = 2.0 * r_small  # internal px
					target_width_internal = large_width * render_scale
					remaining = max(0.0, target_width_internal - current_small_width)
					if remaining <= 0.0:
						# Reached or exceeded target size: clamp and finish
						r_small = R_large
						vel_x = 0.0
						vel_y = 0.0
						finished = True
					else:
						# Compute increment in screen px then scale to internal px to preserve behavior
						remaining_screen = remaining / render_scale
						inc_screen = max(1.0, math.ceil(remaining_screen / 100.0))
						inc_internal = inc_screen * render_scale
						inc_internal = min(inc_internal, remaining)  # do not overshoot
						r_small += inc_internal / 2.0
						if r_small >= R_large:
							r_small = R_large
							vel_x = 0.0
							vel_y = 0.0
							finished = True
					collision_cooldown = int(max(1, fps // 30))  # ~1-2 frames
			if collision_cooldown > 0:
				collision_cooldown -= 1

		# Final clamp check independent of collisions
		if not finished and 2.0 * r_small >= (large_width * render_scale):
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
					# Project sample inside the large polygon if slightly outside
					# Do up to two correction passes using the worst-penetration edge
					for _ in range(6):
						max_pen_s = 0.0
						hit_n_s = None
						for idx, (e0s, _e1s, n_outs) in enumerate(edges_pre):
							nx, ny = n_outs
							d_center = (px - e0s[0]) * nx + (py - e0s[1]) * ny
							worst = d_center + r_draw * support_dot_max_unit[idx]
							if worst > max_pen_s:
								max_pen_s = worst
								hit_n_s = n_outs
						if not (max_pen_s > 0.0 and hit_n_s is not None):
							break
						nx, ny = hit_n_s
						px -= nx * max_pen_s
						py -= ny * max_pen_s
					# One last clamp if still slightly outside
					max_pen_s = 0.0
					hit_n_s = None
					for idx, (e0s, _e1s, n_outs) in enumerate(edges_pre):
						nx, ny = n_outs
						d_center = (px - e0s[0]) * nx + (py - e0s[1]) * ny
						worst = d_center + r_draw * support_dot_max_unit[idx]
						if worst > max_pen_s:
							max_pen_s = worst
							hit_n_s = n_outs
					if max_pen_s > 0.0 and hit_n_s is not None:
						nx, ny = hit_n_s
						px -= nx * (max_pen_s + 1e-6)
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
		# Draw into internal canvas
		canvas.fill((15, 15, 18))
		# Trail in background: blit accumulated surface (newer paint over older)
		canvas.blit(trail_surface, (0, 0))

		# Large polygon outline (precomputed draw vertices)
		pygame.draw.polygon(canvas, (200, 200, 220), large_verts_draw, width=max(1, int(round(3 * render_scale))))

		# Small polygon filled (white), drawn with rounded vertices
		small_verts_draw = [(int(round(x)), int(round(y))) for x, y in small_poly]
		pygame.draw.polygon(canvas, (255, 255, 255), small_verts_draw)

		# Scale down to screen with smoothing for crisp edges
		scaled = pygame.transform.smoothscale(canvas, window_size)
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
	parser.add_argument("--small", type=float, default=10.0, help="Small polygon starting width in pixels (across)")
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

