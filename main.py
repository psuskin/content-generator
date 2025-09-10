import pygame
import numpy as np
import random
import math
import config

# Vibrant, eye-catching color palette with high contrast
VIBRANT_COLORS = [
    (255, 50, 50),    # Bright Red
    (50, 255, 50),    # Bright Green  
    (50, 50, 255),    # Bright Blue
    (255, 255, 50),   # Bright Yellow
    (255, 50, 255),   # Bright Magenta
    (50, 255, 255),   # Bright Cyan
    (255, 150, 50),   # Bright Orange
    (150, 50, 255),   # Bright Purple
    (255, 50, 150),   # Bright Pink
    (150, 255, 50),   # Bright Lime
    (50, 150, 255),   # Bright Sky Blue
    (255, 200, 50),   # Bright Gold
    (255, 100, 100),  # Light Red
    (100, 255, 100),  # Light Green
    (100, 100, 255),  # Light Blue
    (200, 255, 100),  # Light Yellow-Green
    (255, 100, 200),  # Light Pink
    (100, 255, 200),  # Light Turquoise
    (200, 100, 255),  # Light Violet
    (255, 200, 100),  # Light Peach
]

def color_distance(color1, color2):
    """Calculate the perceptual distance between two RGB colors."""
    # Use weighted Euclidean distance for better perceptual accuracy
    r_diff = color1[0] - color2[0]
    g_diff = color1[1] - color2[1]
    b_diff = color1[2] - color2[2]
    
    # Weight green more heavily as human eyes are more sensitive to it
    return math.sqrt(2 * r_diff**2 + 4 * g_diff**2 + 3 * b_diff**2)

def generate_distinct_colors(num_colors):
    """
    Generate a list of distinct, vibrant colors.
    Uses predefined palette first, then generates additional colors ensuring minimum distance.
    """
    if num_colors <= len(VIBRANT_COLORS):
        # Use predefined palette, shuffled for variety
        selected_colors = VIBRANT_COLORS[:num_colors].copy()
        random.shuffle(selected_colors)
        return selected_colors
    
    # Start with all predefined colors
    colors = VIBRANT_COLORS.copy()
    
    # Generate additional colors if needed
    min_distance = 100  # Minimum perceptual distance between colors
    max_attempts = 1000  # Prevent infinite loops
    
    while len(colors) < num_colors:
        attempts = 0
        found_color = False
        
        while attempts < max_attempts and not found_color:
            # Generate a bright, saturated color
            # Ensure at least one channel is high (for vibrancy)
            if random.random() < 0.33:
                # Red-dominant
                new_color = (random.randint(200, 255), random.randint(0, 150), random.randint(0, 150))
            elif random.random() < 0.5:
                # Green-dominant  
                new_color = (random.randint(0, 150), random.randint(200, 255), random.randint(0, 150))
            else:
                # Blue-dominant
                new_color = (random.randint(0, 150), random.randint(0, 150), random.randint(200, 255))
            
            # Check distance from all existing colors
            min_dist = min(color_distance(new_color, existing) for existing in colors)
            
            if min_dist >= min_distance:
                colors.append(new_color)
                found_color = True
            
            attempts += 1
        
        if not found_color:
            # If we can't find a distant color, reduce the minimum distance requirement
            min_distance = max(50, min_distance - 10)
    
    # Shuffle for variety
    random.shuffle(colors)
    return colors[:num_colors]

class Line:
    """
    Represents a line connecting a point to the wall with ROBUST collision detection.
    """
    def __init__(self, point_id, wall_angle, color):
        self.point_id = point_id  # ID of the point this line belongs to
        self.wall_angle = wall_angle  # Angle on wall where line connects (fixed)
        self.color = color  # Same color as the point
        self.active = True  # Whether the line is still active
        self.dashed = False  # Whether the line is dashed (hit once)
        
        # Simple collision tracking - just track which points have been near this line
        self.collision_cooldown = {}  # point_id -> frames_remaining
    
    def get_wall_position(self, center_x, center_y, radius):
        """Get the position on the wall where this line connects."""
        wall_x = center_x + radius * math.cos(self.wall_angle)
        wall_y = center_y + radius * math.sin(self.wall_angle)
        return wall_x, wall_y
    
    def get_point_position(self, points):
        """Get the current position of the point this line belongs to."""
        for point in points:
            if point.id == self.point_id:
                return point.x, point.y
        return None, None
    
    def check_point_line_intersection(self, test_point, center_x, center_y, radius, points):
        """
        Center-path intersection check.
        Returns True if the center path of the moving point (prev->curr) crosses this line segment.
        """
        # Skip own lines (point can't cross its own lines)
        if test_point.id == self.point_id:
            return False
            
        # Only different colors should interact - skip same-color collisions
        if test_point.color == self.color:
            return False
        
        # Check cooldown - prevent rapid repeated collisions
        if test_point.id in self.collision_cooldown:
            self.collision_cooldown[test_point.id] -= 1
            if self.collision_cooldown[test_point.id] > 0:
                return False
            else:
                del self.collision_cooldown[test_point.id]
        
        # Get static wall endpoint
        wall_x, wall_y = self.get_wall_position(center_x, center_y, radius)

        # Find owning point object (for prev/current interpolation)
        owner_point = None
        for p in points:
            if p.id == self.point_id:
                owner_point = p
                break
        if owner_point is None:
            return False

        # Quick reject if positions are not finite
        if not all(
            math.isfinite(v)
            for v in [
                test_point.prev_x, test_point.prev_y, test_point.x, test_point.y,
                owner_point.prev_x, owner_point.prev_y, owner_point.x, owner_point.y,
                wall_x, wall_y,
            ]
        ):
            return False

        # Exact continuous-time detection via solving cross((P(t)-W), (O(t)-W)) = 0 for t in [0,1]
        # Then verify P(t) lies between O(t) and W (segment bounds).
        P0x, P0y = test_point.prev_x, test_point.prev_y
        P1x, P1y = test_point.x, test_point.y
        dPx, dPy = (P1x - P0x), (P1y - P0y)

        O0x, O0y = owner_point.prev_x, owner_point.prev_y
        O1x, O1y = owner_point.x, owner_point.y
        dOx, dOy = (O1x - O0x), (O1y - O0y)

        Wx, Wy = wall_x, wall_y

        Ax, Ay = (P0x - Wx), (P0y - Wy)
        Bx, By = (O0x - Wx), (O0y - Wy)

        def cross2(x1, y1, x2, y2):
            return x1 * y2 - y1 * x2

        def dot2(x1, y1, x2, y2):
            return x1 * x2 + y1 * y2

        # c(t) = C2*t^2 + C1*t + C0
        C2 = cross2(dPx, dPy, dOx, dOy)
        C1 = cross2(dPx, dPy, Bx, By) + cross2(Ax, Ay, dOx, dOy)
        C0 = cross2(Ax, Ay, Bx, By)

        eps = 1e-9

        def check_t(t):
            if t < -1e-6 or t > 1 + 1e-6:
                return False
            # Clamp t to [0,1] for numerical stability
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            # Positions at time t
            Px = P0x + dPx * t
            Py = P0y + dPy * t
            Ox = O0x + dOx * t
            Oy = O0y + dOy * t
            vx, vy = (Wx - Ox), (Wy - Oy)  # segment vector O(t)->W
            denom = dot2(vx, vy, vx, vy)
            if denom <= eps:
                # Degenerate segment (owner at wall). Consider hit only if P ≈ W
                if (abs(Px - Wx) <= 1e-6 and abs(Py - Wy) <= 1e-6):
                    return True
                return False
            sx, sy = (Px - Ox), (Py - Oy)
            s = dot2(sx, sy, vx, vy) / denom
            if s < -1e-6 or s > 1 + 1e-6:
                return False
            # Optional: ensure near-collinearity numerically
            dist_perp = abs(cross2(sx, sy, vx, vy)) / math.sqrt(denom)
            return dist_perp <= 1e-5

        roots = []
        if abs(C2) > eps:
            disc = C1 * C1 - 4.0 * C2 * C0
            if disc >= -1e-12:
                if disc < 0.0:
                    disc = 0.0
                sqrt_disc = math.sqrt(disc)
                t1 = (-C1 - sqrt_disc) / (2.0 * C2)
                t2 = (-C1 + sqrt_disc) / (2.0 * C2)
                roots.extend([t1, t2])
        else:
            # Linear or constant
            if abs(C1) > eps:
                t = -C0 / C1
                roots.append(t)
            else:
                # Constant: either always collinear (C0≈0) or never
                if abs(C0) <= 1e-12:
                    # Fully collinear over the frame: decide if exists t with P between O and W.
                    # Use a fixed unit direction u along the common line and analyze h(t) = f(t)*(f(t)-g(t)) <= 0.
                    # f(t) = dot(P(t)-W, u) = aP + bP t; g(t) = dot(O(t)-W, u) = aO + bO t
                    # Choose direction u
                    dirx, diry = Ax, Ay
                    if abs(dirx) + abs(diry) <= 1e-12:
                        dirx, diry = Bx, By
                    if abs(dirx) + abs(diry) <= 1e-12:
                        # Fall back to movement dir to define the line
                        dirx, diry = (dPx if abs(dPx) >= abs(dOx) else dOx), (dPy if abs(dPy) >= abs(dOy) else dOy)
                    norm = math.hypot(dirx, diry)
                    if norm <= 1e-15:
                        # Pathologically degenerate; no reliable intersection
                        return False
                    ux, uy = dirx / norm, diry / norm

                    aP = dot2(Ax, Ay, ux, uy)
                    bP = dot2(dPx, dPy, ux, uy)
                    aO = dot2(Bx, By, ux, uy)
                    bO = dot2(dOx, dOy, ux, uy)

                    # h(t) = f^2 - f*g = (bP^2 - bP*bO) t^2 + (2 aP bP - aP bO - aO bP) t + (aP^2 - aP aO)
                    qa = (bP * bP - bP * bO)
                    qb = (2.0 * aP * bP - aP * bO - aO * bP)
                    qc = (aP * aP - aP * aO)

                    def h(t):
                        return ((qa * t + qb) * t + qc)

                    # Check if h(t) <= 0 for some t in [0,1]
                    # Consider roots and interval signs
                    candidates = [0.0, 1.0]
                    if abs(qa) > 1e-15:
                        disc2 = qb * qb - 4.0 * qa * qc
                        if disc2 >= -1e-12:
                            if disc2 < 0.0:
                                disc2 = 0.0
                            r = math.sqrt(disc2)
                            candidates.extend([(-qb - r) / (2.0 * qa), (-qb + r) / (2.0 * qa)])
                    elif abs(qb) > 1e-15:
                        candidates.append(-qc / qb)

                    # Evaluate midpoints between sorted candidates inside [0,1]
                    candidates = [t for t in candidates if -1e-6 <= t <= 1 + 1e-6]
                    candidates = sorted(set(max(0.0, min(1.0, t)) for t in candidates))
                    eval_points = set(candidates)
                    for i in range(len(candidates) - 1):
                        mid = 0.5 * (candidates[i] + candidates[i + 1])
                        eval_points.add(mid)
                    for t in eval_points:
                        if h(t) <= 1e-10:
                            self.collision_cooldown[test_point.id] = 30
                            return True
                    return False
                else:
                    # Not collinear anywhere in the frame
                    return False

        for t in roots:
            if check_t(t):
                self.collision_cooldown[test_point.id] = 30
                return True

        return False

class Point:
    """
    Represents a point in the simulation with position, velocity, and physics properties.
    """
    _id_counter = 0  # Class variable for unique IDs
    
    def __init__(self, x, y, vx, vy, radius=None, color=(255, 255, 255)):
        # Assign unique ID
        Point._id_counter += 1
        self.id = Point._id_counter
        
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius if radius is not None else config.POINT_RADIUS
        self.color = color
        self.mass = config.POINT_MASS
        self.max_speed = config.MAX_POINT_SPEED
        
        # Line management
        self.line_ids = []  # IDs of lines belonging to this point
        
        # Movement tracking for robust collision detection
        self.prev_x = x
        self.prev_y = y
    
    def validate_values(self):
        """Ensure all values are valid (not NaN or infinity)."""
        # Check for NaN or infinity and reset to safe values
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            self.x = config.SAFE_FALLBACK_X
            self.y = config.SAFE_FALLBACK_Y
        
        if not (math.isfinite(self.vx) and math.isfinite(self.vy)):
            self.vx = config.SAFE_FALLBACK_SPEED
            self.vy = config.SAFE_FALLBACK_SPEED
        
        # Clamp speed to maximum
        speed = self.get_speed()
        if speed > self.max_speed:
            self.set_speed(self.max_speed)
    
    def update(self, dt):
        """Update position based on velocity."""
        # Store previous position for collision detection
        self.prev_x = self.x
        self.prev_y = self.y
        
        # Limit dt to prevent large jumps
        dt = min(dt, config.MAX_TIME_STEP)
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Validate values after update
        self.validate_values()
    
    def get_movement_path(self):
        """Get the movement path from previous to current position."""
        return self.prev_x, self.prev_y, self.x, self.y
    
    def get_speed(self):
        """Get current speed magnitude."""
        speed_squared = self.vx ** 2 + self.vy ** 2
        if not math.isfinite(speed_squared) or speed_squared < 0:
            return 0
        return math.sqrt(speed_squared)
    
    def set_speed(self, speed):
        """Set speed while maintaining direction."""
        if not math.isfinite(speed) or speed < 0:
            speed = 0
        
        current_speed = self.get_speed()
        if current_speed > config.MIN_SPEED_EPSILON and math.isfinite(current_speed):
            factor = speed / current_speed
            self.vx *= factor
            self.vy *= factor
        else:
            # If current speed is 0 or invalid, set random direction
            angle = random.uniform(0, 2 * math.pi)
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)
        
        # Validate after setting
        self.validate_values()

class CircleSimulation:
    """
    Main simulation class handling physics, collisions, and rendering.
    """
    def __init__(self, width=None, height=None, num_points=None, energy_factor=None):
        pygame.init()
        
        self.width = width if width is not None else config.WINDOW_WIDTH
        self.height = height if height is not None else config.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(config.WINDOW_TITLE)
        
        # Simulation parameters
        self.num_points = num_points if num_points is not None else config.DEFAULT_NUM_POINTS
        self.energy_factor = energy_factor if energy_factor is not None else config.DEFAULT_ENERGY_FACTOR
        self.base_speed = config.BASE_SPEED
        
        # UI state
        self.show_ui = True  # Toggle for showing/hiding UI
        
        # Circle boundary
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.circle_radius = min(self.width, self.height) // 2 - config.CIRCLE_MARGIN
        
        # Initialize points and lines
        self.points = []
        self.lines = {}  # Dictionary mapping line_id to Line object
        self.line_id_counter = 0
        self.generate_points()
        
        # Clock for consistent frame rate
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Font for UI
        self.font = pygame.font.Font(None, config.UI_FONT_SIZE)
    
    def generate_points(self):
        """Generate points with random positions and velocities, ensuring minimum distance."""
        self.points.clear()
        self.lines.clear()
        self.line_id_counter = 0
        Point._id_counter = 0  # Reset point ID counter
        
        # Generate distinct colors for all points at once
        distinct_colors = generate_distinct_colors(self.num_points)
        
        min_distance = config.MIN_DISTANCE_BETWEEN_POINTS
        max_attempts = config.MAX_PLACEMENT_ATTEMPTS
        
        for i in range(self.num_points):
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Generate random position within circle
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, self.circle_radius - config.POINT_RADIUS - 10)
                
                x = self.center_x + radius * math.cos(angle)
                y = self.center_y + radius * math.sin(angle)
                
                # Check distance from other points
                valid_position = True
                for existing_point in self.points:
                    distance = math.sqrt((x - existing_point.x) ** 2 + (y - existing_point.y) ** 2)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    # Generate random velocity direction
                    vel_angle = random.uniform(0, 2 * math.pi)
                    vx = self.base_speed * math.cos(vel_angle)
                    vy = self.base_speed * math.sin(vel_angle)
                    
                    # Use pre-generated distinct color for this point
                    color = distinct_colors[i]
                    
                    point = Point(x, y, vx, vy, color=color)
                    self.points.append(point)
                    
                    # Generate initial lines for this point
                    self.generate_lines_for_point(point)
                    
                    placed = True
                
                attempts += 1
            
            if not placed:
                print(f"Warning: Could not place point {i+1} after {max_attempts} attempts")
    
    def generate_lines_for_point(self, point):
        """Generate initial lines connecting a point to the wall."""
        # Find the closest point on the wall (minimum distance)
        dx = point.x - self.center_x
        dy = point.y - self.center_y
        center_angle = math.atan2(dy, dx)
        
        # Generate three lines: center and two at arc offsets
        arc_offset = math.radians(config.LINE_ARC_DEGREES)
        wall_angles = [
            center_angle - arc_offset,  # Left line
            center_angle,               # Center line
            center_angle + arc_offset   # Right line
        ]
        
        for wall_angle in wall_angles:
            line_id = self.line_id_counter
            self.line_id_counter += 1
            
            line = Line(point.id, wall_angle, point.color)
            self.lines[line_id] = line
            point.line_ids.append(line_id)
    
    def check_line_collisions(self):
        """Check if any points cross lines of different colors using SIMPLE, ROBUST collision detection."""
        lines_to_remove = []
        # Aggregate unique crossings per line within this frame
        collisions_per_line = {}  # line_id -> set(point_id)

        # Check each point against all lines and record hits
        for point in self.points:
            for line_id, line in self.lines.items():
                if not line.active:
                    continue
                if line.check_point_line_intersection(point, self.center_x, self.center_y, self.circle_radius, self.points):
                    s = collisions_per_line.setdefault(line_id, set())
                    s.add(point.id)

        # Apply results in a second pass to allow multiple transitions per line per frame
        for line_id, point_ids in collisions_per_line.items():
            if line_id not in self.lines:
                continue
            line = self.lines[line_id]
            if not line.active:
                continue
            hit_count = len(point_ids)
            if not line.dashed:
                # If two or more distinct points hit in the same frame, remove directly; otherwise dash
                if hit_count >= 2:
                    line.active = False
                    lines_to_remove.append(line_id)
                elif hit_count == 1:
                    line.dashed = True
            else:
                # Already dashed: any hit removes
                if hit_count >= 1:
                    line.active = False
                    lines_to_remove.append(line_id)

        # Remove inactive lines
        self._remove_lines(lines_to_remove)
    
    def _remove_lines(self, lines_to_remove):
        """Efficiently remove multiple lines."""
        for line_id in lines_to_remove:
            if line_id in self.lines:
                line = self.lines[line_id]
                # Remove line ID from the point's line list
                for point in self.points:
                    if point.id == line.point_id and line_id in point.line_ids:
                        point.line_ids.remove(line_id)
                        break
                del self.lines[line_id]
    
    def remove_points_without_lines(self):
        """Remove points that have no more lines connecting them to the wall."""
        points_to_remove = []
        
        for point in self.points:
            # Count active lines for this point
            active_lines = sum(1 for line_id in point.line_ids if line_id in self.lines and self.lines[line_id].active)
            
            if active_lines == 0:
                points_to_remove.append(point)
        
        # Remove points and their remaining line references
        for point in points_to_remove:
            # Remove any remaining line references
            for line_id in point.line_ids:
                if line_id in self.lines:
                    del self.lines[line_id]
            self.points.remove(point)
    
    def check_circle_collision(self, point):
        """Check and handle collision with circle boundary."""
        distance_from_center = math.sqrt(
            (point.x - self.center_x) ** 2 + (point.y - self.center_y) ** 2
        )
        
        # Validate distance calculation
        if not math.isfinite(distance_from_center):
            distance_from_center = 0
        
        if distance_from_center + point.radius >= self.circle_radius:
            # Collision with circle boundary - reflect velocity and generate new lines
            
            # Calculate reflection
            if distance_from_center > config.MIN_DISTANCE_EPSILON:
                # Normal vector pointing inward
                nx = -(point.x - self.center_x) / distance_from_center
                ny = -(point.y - self.center_y) / distance_from_center
                
                # Reflect velocity
                dot_product = point.vx * nx + point.vy * ny
                point.vx = point.vx - 2 * dot_product * nx
                point.vy = point.vy - 2 * dot_product * ny
                
                # Apply energy factor
                point.vx *= self.energy_factor
                point.vy *= self.energy_factor
            
            # Move point back inside circle
            if distance_from_center > 0:
                penetration = (distance_from_center + point.radius) - self.circle_radius
                point.x -= penetration * (point.x - self.center_x) / distance_from_center
                point.y -= penetration * (point.y - self.center_y) / distance_from_center
            
            # Generate new lines for this collision
            self.generate_collision_lines(point)
            
            # Validate point after collision
            point.validate_values()
    
    def generate_collision_lines(self, point):
        """Generate three new lines when a point collides with the wall."""
        # Calculate collision angle on the wall
        dx = point.x - self.center_x
        dy = point.y - self.center_y
        collision_angle = math.atan2(dy, dx)
        
        # Generate three lines: center at collision point and two at arc offsets
        arc_offset = math.radians(config.LINE_ARC_DEGREES)
        wall_angles = [
            collision_angle - arc_offset,  # Left line
            collision_angle,               # Center line (at collision point)
            collision_angle + arc_offset   # Right line
        ]
        
        for wall_angle in wall_angles:
            line_id = self.line_id_counter
            self.line_id_counter += 1
            
            line = Line(point.id, wall_angle, point.color)
            self.lines[line_id] = line
            point.line_ids.append(line_id)
    
    def check_point_collision(self, point1, point2):
        """Check and handle elastic collision between two points with energy factor."""
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        # Validate distance calculation
        if not math.isfinite(distance):
            return
        
        if distance < point1.radius + point2.radius:
            # Collision detected
            if distance < config.MIN_DISTANCE_EPSILON:
                distance = config.MIN_DISTANCE_EPSILON
                # Use random separation direction
                angle = random.uniform(0, 2 * math.pi)
                dx = distance * math.cos(angle)
                dy = distance * math.sin(angle)
            
            # Normalize collision vector
            nx = dx / distance
            ny = dy / distance
            
            # Validate normal vector
            if not (math.isfinite(nx) and math.isfinite(ny)):
                return
            
            # Relative velocity
            dvx = point2.vx - point1.vx
            dvy = point2.vy - point1.vy
            
            # Relative velocity along collision normal
            dvn = dvx * nx + dvy * ny
            
            # Don't resolve if velocities are separating
            if dvn > 0:
                return
            
            # Collision impulse (elastic collision)
            impulse_magnitude = 2 * dvn / (point1.mass + point2.mass)
            
            # Limit impulse to prevent extreme velocities
            impulse_magnitude = max(-config.MAX_IMPULSE, min(config.MAX_IMPULSE, impulse_magnitude))
            
            # Apply impulse with energy factor
            impulse_x = impulse_magnitude * nx * self.energy_factor
            impulse_y = impulse_magnitude * ny * self.energy_factor
            
            point1.vx += impulse_x * point2.mass
            point1.vy += impulse_y * point2.mass
            point2.vx -= impulse_x * point1.mass
            point2.vy -= impulse_y * point1.mass
            
            # Separate overlapping points
            overlap = (point1.radius + point2.radius) - distance
            separation_distance = min(config.MAX_SEPARATION, overlap * 0.5)
            
            point1.x -= separation_distance * nx
            point1.y -= separation_distance * ny
            point2.x += separation_distance * nx
            point2.y += separation_distance * ny
            
            # Validate both points after collision
            point1.validate_values()
            point2.validate_values()
    
    def update_physics(self, dt):
        """Update all physics calculations with robust collision detection."""
        # Limit time step to prevent instability
        dt = min(dt, config.MAX_TIME_STEP)
        
        # Update point positions (this stores previous positions automatically)
        for point in self.points:
            point.update(dt)
        
        # Validate all points before collision detection
        for point in self.points:
            point.validate_values()
            
            # Keep points within reasonable bounds
            if (point.x < -config.BOUNDS_BUFFER or point.x > self.width + config.BOUNDS_BUFFER or 
                point.y < -config.BOUNDS_BUFFER or point.y > self.height + config.BOUNDS_BUFFER):
                # Reset point to center if it goes too far out of bounds
                point.prev_x = self.center_x  # Update previous position too
                point.prev_y = self.center_y
                point.x = self.center_x + random.uniform(-config.RESET_RANDOMNESS, config.RESET_RANDOMNESS)
                point.y = self.center_y + random.uniform(-config.RESET_RANDOMNESS, config.RESET_RANDOMNESS)
        
        # Check line collisions FIRST (before new lines are generated)
        # This ensures we catch all crossings before positions change further
        self.check_line_collisions()
        
        # Check circle collisions (which may generate new lines)
        for point in self.points:
            self.check_circle_collision(point)

        # Second pass: catch crossings with lines created during this frame
        # Uses the same prev->curr motion, so continuous-time check remains valid
        self.check_line_collisions()

        # Check point-to-point collisions
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                self.check_point_collision(self.points[i], self.points[j])

        # Remove points that have no remaining lines
        self.remove_points_without_lines()
    
    def render(self):
        """Render the entire simulation."""
        # Clear screen
        self.screen.fill(config.BACKGROUND_COLOR)
        
        # Draw circle boundary
        pygame.draw.circle(
            self.screen, 
            config.CIRCLE_BOUNDARY_COLOR, 
            (self.center_x, self.center_y), 
            self.circle_radius, 
            config.CIRCLE_BOUNDARY_WIDTH
        )
        
        # Draw lines from points to wall
        self.render_lines()
        
        # Draw points
        for point in self.points:
            # Ensure coordinates are valid and within reasonable bounds
            try:
                x = int(point.x) if math.isfinite(point.x) else self.center_x
                y = int(point.y) if math.isfinite(point.y) else self.center_y
                radius = int(point.radius) if math.isfinite(point.radius) else config.POINT_RADIUS
                
                # Clamp coordinates to screen bounds (with some buffer)
                x = max(-config.BOUNDS_BUFFER, min(self.width + config.BOUNDS_BUFFER, x))
                y = max(-config.BOUNDS_BUFFER, min(self.height + config.BOUNDS_BUFFER, y))
                radius = max(config.MIN_RENDER_RADIUS, min(config.MAX_RENDER_RADIUS, radius))
                
                pygame.draw.circle(
                    self.screen,
                    point.color,
                    (x, y),
                    radius
                )
            except (TypeError, ValueError, OverflowError):
                # If there's any error drawing, skip this point
                continue
        
        # Draw UI if enabled
        if self.show_ui:
            self.render_ui()
        
        pygame.display.flip()
    
    def render_lines(self):
        """Render all active lines connecting points to the wall."""
        for line_id, line in self.lines.items():
            if not line.active:
                continue
            
            try:
                # Get wall position
                wall_x, wall_y = line.get_wall_position(self.center_x, self.center_y, self.circle_radius)
                
                # Get point position
                point_x, point_y = line.get_point_position(self.points)
                
                if point_x is None or point_y is None:
                    continue
                
                # Ensure coordinates are finite
                if not all(math.isfinite(coord) for coord in [wall_x, wall_y, point_x, point_y]):
                    continue
                
                # Convert to integers for drawing
                start_pos = (int(point_x), int(point_y))
                end_pos = (int(wall_x), int(wall_y))
                
                # Choose line style based on state
                if line.dashed:
                    # Draw dashed line
                    self._draw_dashed_line(line.color, start_pos, end_pos, config.LINE_WIDTH)
                else:
                    # Draw solid line
                    pygame.draw.line(self.screen, line.color, start_pos, end_pos, config.LINE_WIDTH)
            
            except (TypeError, ValueError, OverflowError):
                # If there's any error drawing, skip this line
                continue
    
    def _draw_dashed_line(self, color, start_pos, end_pos, width):
        """Draw a dashed line between two points."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 1:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Draw dashed segments
        dash_length = 10
        gap_length = 5
        segment_length = dash_length + gap_length
        
        current_pos = 0
        while current_pos < length:
            # Start of dash
            start_x = x1 + current_pos * dx
            start_y = y1 + current_pos * dy
            
            # End of dash
            end_pos_current = min(current_pos + dash_length, length)
            end_x = x1 + end_pos_current * dx
            end_y = y1 + end_pos_current * dy
            
            # Draw dash segment
            pygame.draw.line(
                self.screen, 
                color, 
                (int(start_x), int(start_y)), 
                (int(end_x), int(end_y)), 
                width
            )
            
            current_pos += segment_length
    
    def render_ui(self):
        """Render the user interface overlay."""
        y_offset = config.UI_MARGIN
        
        # Title
        title_text = self.font.render("Point Collision Simulation", True, config.UI_TITLE_COLOR)
        self.screen.blit(title_text, (config.UI_MARGIN, y_offset))
        y_offset += config.UI_LINE_HEIGHT
        
        # Simulation stats
        stats = [
            f"Points: {len(self.points)}",
            f"Lines: {len([l for l in self.lines.values() if l.active])}",
            f"Energy Factor: {self.energy_factor:.3f}"
        ]
        
        for stat in stats:
            stat_text = self.font.render(stat, True, config.UI_TEXT_COLOR)
            self.screen.blit(stat_text, (config.UI_MARGIN, y_offset))
            y_offset += config.UI_STATS_SPACING
        
        # Show average speed
        if self.points:
            valid_speeds = []
            for point in self.points:
                speed = point.get_speed()
                if math.isfinite(speed):
                    valid_speeds.append(speed)
            
            if valid_speeds:
                avg_speed = sum(valid_speeds) / len(valid_speeds)
                avg_speed_text = self.font.render(f"Avg Speed: {avg_speed:.1f}", True, config.UI_TEXT_COLOR)
                self.screen.blit(avg_speed_text, (config.UI_MARGIN, y_offset))
                y_offset += config.UI_STATS_SPACING
                
                # Show max speed as well
                max_speed = max(valid_speeds)
                max_speed_text = self.font.render(f"Max Speed: {max_speed:.1f}", True, config.UI_TEXT_COLOR)
                self.screen.blit(max_speed_text, (config.UI_MARGIN, y_offset))
        
        # Draw controls
        controls_start_y = self.height - config.CONTROLS_FROM_BOTTOM
        for i, control in enumerate(config.CONTROLS):
            color = config.UI_TITLE_COLOR if i == 0 else config.UI_SECONDARY_COLOR
            control_text = self.font.render(control, True, color)
            self.screen.blit(control_text, (config.UI_MARGIN, controls_start_y + i * config.CONTROLS_LINE_HEIGHT))
    
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == config.KEY_EXIT:
                    self.running = False
                
                elif event.key == config.KEY_ENERGY_UP:
                    self.energy_factor = min(config.MAX_ENERGY_FACTOR, 
                                           self.energy_factor + config.ENERGY_FACTOR_STEP)
                
                elif event.key == config.KEY_ENERGY_DOWN:
                    self.energy_factor = max(config.MIN_ENERGY_FACTOR, 
                                           self.energy_factor - config.ENERGY_FACTOR_STEP)
                
                elif event.key in (config.KEY_ADD_POINT_1, config.KEY_ADD_POINT_2):
                    if len(self.points) < config.MAX_POINTS:
                        self.num_points += 1
                        self.generate_points()
                
                elif event.key == config.KEY_REMOVE_POINT:
                    if len(self.points) > config.MIN_POINTS:
                        self.num_points -= 1
                        self.generate_points()
                
                elif event.key == config.KEY_RESET:
                    self.generate_points()
                
                elif event.key == config.KEY_TOGGLE_UI:
                    self.show_ui = not self.show_ui
    
    def run(self):
        """Main simulation loop."""
        while self.running:
            dt = self.clock.tick(config.FPS) / 1000.0  # Delta time in seconds
            
            self.handle_events()
            self.update_physics(dt)
            self.render()
        
        pygame.quit()

def main():
    """Main function to start the simulation."""
    simulation = CircleSimulation(
        width=config.WINDOW_WIDTH,
        height=config.WINDOW_HEIGHT,
        num_points=config.DEFAULT_NUM_POINTS,
        energy_factor=config.DEFAULT_ENERGY_FACTOR
    )
    simulation.run()

if __name__ == "__main__":
    main()
