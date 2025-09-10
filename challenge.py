import math
import argparse
import pygame


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
	window_size=(720, 720),
	fps=120,
):
	pygame.init()
	screen = pygame.display.set_mode(window_size)
	pygame.display.set_caption("n-gon in n-gon — elastic growth sim")
	clock = pygame.time.Clock()
	font = pygame.font.SysFont(None, 24)

	cx, cy = window_size[0] // 2, window_size[1] // 2

	# Interpret width as 2 * circumradius
	R_large = large_width / 2.0
	r_small = small_width / 2.0

	# Orientation: point a vertex upward (rotation = -pi/2)
	rotation = -math.pi / 2.0

	# Small polygon state (no rotation dynamics for simplicity)
	pos_x, pos_y = float(cx), float(cy)
	# Initial direction: a few degrees above straight right (upward in screen space)
	_angle_deg = 7.0
	_angle = math.radians(_angle_deg)
	vel_x = start_speed * math.cos(_angle)
	# Negative vy to go upwards on the screen (y grows downward in Pygame)
	vel_y = -start_speed * math.sin(_angle)

	running = True
	finished = False  # stop physics once small polygon reaches large size
	collision_cooldown = 0  # frames to avoid multiple counts on same hit cluster

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
		large_poly = regular_polygon_vertices(sides, (cx, cy), R_large, rotation)
		small_poly = regular_polygon_vertices(sides, (pos_x, pos_y), r_small, rotation)

		# Collision detection and growth only while not finished
		if not finished:
			# Collision detection: compute max penetration against any large-poly edge
			edges = edge_normals_ccw(large_poly, (cx, cy))
			max_pen = 0.0
			hit_normal = None
			hit_edge = None
			for (e0, e1, n_out) in edges:
				nx, ny = n_out
				# For each vertex of the small polygon, compute signed distance to edge plane
				# Inside region should have distance <= 0; outside => positive
				worst = -1e9
				for vxp, vyp in small_poly:
					dist = (vxp - e0[0]) * nx + (vyp - e0[1]) * ny
					if dist > worst:
						worst = dist
				if worst > max_pen:
					max_pen = worst
					hit_normal = n_out
					hit_edge = (e0, e1)

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
					current_small_width = 2.0 * r_small
					remaining = max(0.0, large_width - current_small_width)
					if remaining <= 0.0:
						# Reached or exceeded target size: clamp and finish
						r_small = R_large
						vel_x = 0.0
						vel_y = 0.0
						finished = True
					else:
						inc = max(1.0, math.ceil(remaining / 100.0))
						inc = min(inc, remaining)  # do not overshoot
						r_small += inc / 2.0
						if r_small >= R_large:
							r_small = R_large
							vel_x = 0.0
							vel_y = 0.0
							finished = True
					collision_cooldown = int(max(1, fps // 30))  # ~1-2 frames
			if collision_cooldown > 0:
				collision_cooldown -= 1

		# Final clamp check independent of collisions
		if not finished and 2.0 * r_small >= large_width:
			r_small = R_large
			vel_x = 0.0
			vel_y = 0.0
			finished = True

		# Rebuild small polygon after any updates (for accurate drawing)
		small_poly = regular_polygon_vertices(sides, (pos_x, pos_y), r_small, rotation)

		# Draw
		screen.fill((15, 15, 18))
		# Large polygon outline
		pygame.draw.polygon(screen, (200, 200, 220), [(int(x), int(y)) for x, y in large_poly], width=3)
		# Small polygon filled (white)
		pygame.draw.polygon(screen, (255, 255, 255), [(int(x), int(y)) for x, y in small_poly])

		# HUD text
		hud = [
			f"sides: {sides}",
			f"small width: {2*r_small:.1f}px",
			f"speed: {math.hypot(vel_x, vel_y):.1f}px/s",
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

