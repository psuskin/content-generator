import pygame
import numpy as np
import random
import math
import config

class Line:
    """
    Represents a line connecting a point to the wall.
    """
    def __init__(self, point_id, wall_angle, color):
        self.point_id = point_id  # ID of the point this line belongs to
        self.wall_angle = wall_angle  # Angle on wall where line connects (fixed)
        self.color = color  # Same color as the point
        self.active = True  # Whether the line is still active
        self.dashed = False  # Whether the line is dashed (hit once)
        self.last_collision_point_id = None  # Track which point last collided to prevent double hits
    
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
    
    def check_point_intersection(self, point, center_x, center_y, radius, points):
        """Check if a point crosses this line using path-based collision detection."""
        if not self.active:
            return False
        
        # Get current positions
        wall_x, wall_y = self.get_wall_position(center_x, center_y, radius)
        point_x, point_y = self.get_point_position(points)
        
        if point_x is None or point_y is None:
            return False
        
        # Get movement path of the crossing point
        prev_cross_x, prev_cross_y, curr_cross_x, curr_cross_y = point.get_movement_path()
        
        # Check both current position and movement path
        return (self._check_position_intersection(curr_cross_x, curr_cross_y, point_x, point_y, wall_x, wall_y) or
                self._check_path_intersection(prev_cross_x, prev_cross_y, curr_cross_x, curr_cross_y, 
                                            point_x, point_y, wall_x, wall_y))
    
    def _check_position_intersection(self, cross_x, cross_y, point_x, point_y, wall_x, wall_y):
        """Check if a position intersects with the line."""
        # Calculate distance from crossing point to line (from point to wall)
        line_length = math.sqrt((wall_x - point_x)**2 + (wall_y - point_y)**2)
        if line_length < config.MIN_LINE_SEGMENT_LENGTH:
            return False
        
        # Vector from point to wall
        line_dx = wall_x - point_x
        line_dy = wall_y - point_y
        
        # Vector from point to crossing position
        cross_dx = cross_x - point_x
        cross_dy = cross_y - point_y
        
        # Project crossing position onto line
        dot_product = (cross_dx * line_dx + cross_dy * line_dy) / (line_length * line_length)
        
        # Check if projection is within line bounds
        if dot_product < 0 or dot_product > 1:
            return False
        
        # Closest point on line to the crossing position
        closest_x = point_x + dot_product * line_dx
        closest_y = point_y + dot_product * line_dy
        
        # Distance from crossing position to line
        distance = math.sqrt((cross_x - closest_x)**2 + (cross_y - closest_y)**2)
        
        return distance <= config.LINE_COLLISION_TOLERANCE
    
    def _check_path_intersection(self, prev_x, prev_y, curr_x, curr_y, point_x, point_y, wall_x, wall_y):
        """Check if the movement path intersects with the line using line-line intersection."""
        # Use subdivisions to check multiple points along the path for high-speed movements
        subdivisions = config.LINE_COLLISION_SUBDIVISIONS
        
        for i in range(subdivisions + 1):
            t = i / subdivisions
            # Interpolate position along movement path
            interp_x = prev_x + t * (curr_x - prev_x)
            interp_y = prev_y + t * (curr_y - prev_y)
            
            if self._check_position_intersection(interp_x, interp_y, point_x, point_y, wall_x, wall_y):
                return True
        
        # Additionally, check for direct line-line intersection
        return self._check_line_line_intersection(prev_x, prev_y, curr_x, curr_y, point_x, point_y, wall_x, wall_y)
    
    def _check_line_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect using parametric equations."""
        # Line 1: movement path (x1,y1) to (x2,y2)
        # Line 2: connection line (x3,y3) to (x4,y4)
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return False
        
        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection occurs within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            
            # Verify intersection is within tolerance (accounting for point radius)
            movement_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if movement_length > 0:
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
                    
                    # Create point with random color
                    color = (
                        random.randint(config.POINT_COLOR_MIN, config.POINT_COLOR_MAX),
                        random.randint(config.POINT_COLOR_MIN, config.POINT_COLOR_MAX),
                        random.randint(config.POINT_COLOR_MIN, config.POINT_COLOR_MAX)
                    )
                    
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
        """Check if any points cross lines of different colors using robust collision detection."""
        lines_to_remove = []
        
        # Create a list of all active lines for more efficient checking
        active_lines = [(line_id, line) for line_id, line in self.lines.items() if line.active]
        
        # Reset collision tracking for lines that are no longer being touched
        for line_id, line in active_lines:
            if line.last_collision_point_id is not None:
                # Check if the last colliding point is still close to this line
                colliding_point = None
                for point in self.points:
                    if point.id == line.last_collision_point_id:
                        colliding_point = point
                        break
                
                if colliding_point is None:
                    # Point was removed, reset collision tracking
                    line.last_collision_point_id = None
                elif not line.check_point_intersection(colliding_point, self.center_x, self.center_y, self.circle_radius, self.points):
                    # Point moved away from line, reset collision tracking
                    line.last_collision_point_id = None
        
        # Check each point against all lines
        for point in self.points:
            # Skip if point has minimal movement (optimization)
            movement_distance = math.sqrt((point.x - point.prev_x)**2 + (point.y - point.prev_y)**2)
            
            for line_id, line in active_lines:
                # Don't check collision with own lines
                if point.id == line.point_id:
                    continue
                
                # Skip if line is already marked for removal
                if line_id in lines_to_remove:
                    continue
                
                # Enhanced collision detection with new two-stage system
                if self._enhanced_line_collision_check(point, line, line_id):
                    # Line collision detected - mark line for removal (second hit)
                    line.active = False
                    lines_to_remove.append(line_id)
        
        # Remove inactive lines
        self._remove_lines(lines_to_remove)
    
    def _enhanced_line_collision_check(self, point, line, line_id):
        """Perform enhanced collision checking with multiple strategies."""
        # Strategy 1: Standard intersection check
        if line.check_point_intersection(point, self.center_x, self.center_y, self.circle_radius, self.points):
            return self._handle_line_collision(point, line, line_id)
        
        # Strategy 2: Check if point is about to cross line (predictive)
        if self._predictive_collision_check(point, line):
            return self._handle_line_collision(point, line, line_id)
        
        # Strategy 3: Check for high-speed tunneling
        if self._tunneling_check(point, line):
            return self._handle_line_collision(point, line, line_id)
        
        return False
    
    def _handle_line_collision(self, point, line, line_id):
        """Handle a line collision with the new two-stage system."""
        # Check if the colliding point has a different color than the line
        if point.color != line.color:
            # Prevent the same point from triggering multiple collisions immediately
            if line.last_collision_point_id == point.id:
                return False  # Same point still colliding, ignore
            
            line.last_collision_point_id = point.id  # Record this collision
            
            if not line.dashed:
                # First collision: make line dashed
                line.dashed = True
                return False  # Don't remove the line yet
            else:
                # Second collision: remove the line
                return True  # Remove the line
        
        return False  # No collision or same color
    
    def _predictive_collision_check(self, point, line):
        """Check if point will cross line in the next frame."""
        # Get line endpoints
        wall_x, wall_y = line.get_wall_position(self.center_x, self.center_y, self.circle_radius)
        point_x, point_y = line.get_point_position(self.points)
        
        if point_x is None or point_y is None:
            return False
        
        # Predict next position
        next_dt = 1.0 / config.FPS  # Assume consistent frame rate
        next_x = point.x + point.vx * next_dt
        next_y = point.y + point.vy * next_dt
        
        # Check if predicted path would cross the line
        return line._check_line_line_intersection(point.x, point.y, next_x, next_y, point_x, point_y, wall_x, wall_y)
    
    def _tunneling_check(self, point, line):
        """Check for high-speed tunneling through lines."""
        # Get movement distance
        movement_distance = math.sqrt((point.x - point.prev_x)**2 + (point.y - point.prev_y)**2)
        
        # If movement is very large relative to collision tolerance, use finer subdivision
        if movement_distance > config.LINE_COLLISION_TOLERANCE * 2:
            # Get line endpoints
            wall_x, wall_y = line.get_wall_position(self.center_x, self.center_y, self.circle_radius)
            point_x, point_y = line.get_point_position(self.points)
            
            if point_x is None or point_y is None:
                return False
            
            # Use higher subdivision count for very fast movements
            subdivisions = max(config.LINE_COLLISION_SUBDIVISIONS, int(movement_distance / config.LINE_COLLISION_TOLERANCE))
            subdivisions = min(subdivisions, 50)  # Cap to prevent performance issues
            
            for i in range(subdivisions + 1):
                t = i / subdivisions
                # Interpolate position along movement path
                interp_x = point.prev_x + t * (point.x - point.prev_x)
                interp_y = point.prev_y + t * (point.y - point.prev_y)
                
                if line._check_position_intersection(interp_x, interp_y, point_x, point_y, wall_x, wall_y):
                    return True
        
        return False
    
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
            # Remove all lines belonging to this point
            lines_to_remove = []
            for line_id, line in self.lines.items():
                if line.point_id == point.id:
                    lines_to_remove.append(line_id)
            
            for line_id in lines_to_remove:
                del self.lines[line_id]
            
            # Remove point from points list
            self.points.remove(point)
            self.num_points = len(self.points)
    
    def check_circle_collision(self, point):
        """Check and handle collision with circle boundary."""
        distance_from_center = math.sqrt(
            (point.x - self.center_x) ** 2 + (point.y - self.center_y) ** 2
        )
        
        # Validate distance calculation
        if not math.isfinite(distance_from_center):
            # Reset point to center if position is invalid
            point.x = self.center_x
            point.y = self.center_y
            distance_from_center = 0
        
        if distance_from_center + point.radius >= self.circle_radius:
            # Calculate collision normal (pointing inward)
            if distance_from_center > 0:
                nx = (self.center_x - point.x) / distance_from_center
                ny = (self.center_y - point.y) / distance_from_center
            else:
                # If point is at center, use random direction
                angle = random.uniform(0, 2 * math.pi)
                nx = math.cos(angle)
                ny = math.sin(angle)
            
            # Reflect velocity (elastic collision with boundary)
            dot_product = point.vx * nx + point.vy * ny
            point.vx -= 2 * dot_product * nx
            point.vy -= 2 * dot_product * ny
            
            # Apply energy factor boost (same as point-point collisions)
            current_speed = point.get_speed()
            if current_speed > config.MIN_SPEED_EPSILON:
                point.set_speed(current_speed * self.energy_factor)
            
            # Move point back inside circle
            if distance_from_center > 0:
                overlap = (distance_from_center + point.radius) - self.circle_radius
                point.x -= overlap * (point.x - self.center_x) / distance_from_center
                point.y -= overlap * (point.y - self.center_y) / distance_from_center
            else:
                # If at center, move to safe position
                safe_radius = self.circle_radius - point.radius - 10
                angle = random.uniform(0, 2 * math.pi)
                point.x = self.center_x + safe_radius * math.cos(angle)
                point.y = self.center_y + safe_radius * math.sin(angle)
            
            # Generate new lines at collision point
            self.generate_collision_lines(point)
            
            # Final validation
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
            
            # ELASTIC COLLISION PHYSICS WITH ENERGY FACTOR
            # For elastic collision between equal masses, velocities exchange along normal
            
            # Store original speeds for energy factor calculation
            speed1_before = point1.get_speed()
            speed2_before = point2.get_speed()
            
            # Velocity components along collision normal
            v1n = point1.vx * nx + point1.vy * ny
            v2n = point2.vx * nx + point2.vy * ny
            
            # Velocity components perpendicular to collision normal
            tx = -ny  # Tangent vector perpendicular to normal
            ty = nx
            v1t = point1.vx * tx + point1.vy * ty
            v2t = point2.vx * tx + point2.vy * ty
            
            # For elastic collision with equal masses, normal components are exchanged
            # Tangential components remain unchanged
            v1n_new = v2n
            v2n_new = v1n
            
            # Reconstruct velocity vectors
            point1.vx = v1n_new * nx + v1t * tx
            point1.vy = v1n_new * ny + v1t * ty
            point2.vx = v2n_new * nx + v2t * tx
            point2.vy = v2n_new * ny + v2t * ty
            
            # Apply energy factor boost to both points
            # This adds energy to the system, making it more chaotic over time
            speed1_after_elastic = point1.get_speed()
            speed2_after_elastic = point2.get_speed()
            
            # Apply energy factor with validation
            if speed1_after_elastic > config.MIN_SPEED_EPSILON:
                point1.set_speed(speed1_after_elastic * self.energy_factor)
            if speed2_after_elastic > config.MIN_SPEED_EPSILON:
                point2.set_speed(speed2_after_elastic * self.energy_factor)
            
            # Validate final velocities
            point1.validate_values()
            point2.validate_values()
            
            # Separate overlapping points
            overlap = (point1.radius + point2.radius) - distance
            separation = overlap / 2
            
            # Limit separation to prevent extreme position changes
            max_separation = config.MAX_SEPARATION
            separation = min(separation, max_separation)
            
            point1.x -= separation * nx
            point1.y -= separation * ny
            point2.x += separation * nx
            point2.y += separation * ny
            
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
        
        # Check point-to-point collisions
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                self.check_point_collision(self.points[i], self.points[j])
        
        # Check line collisions again after all position changes
        # This catches any new crossings created by collision responses
        self.check_line_collisions()
        
        # Remove points that have no more lines
        self.remove_points_without_lines()
    
    def render(self):
        """Render the simulation."""
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
                
                # Get current point position
                point_x, point_y = line.get_point_position(self.points)
                
                # Ensure coordinates are valid
                if not (math.isfinite(wall_x) and math.isfinite(wall_y) and 
                       point_x is not None and point_y is not None and
                       math.isfinite(point_x) and math.isfinite(point_y)):
                    continue
                
                # Draw line from point to wall (solid or dashed)
                if line.dashed:
                    # Draw dashed line
                    self._draw_dashed_line(line.color, (int(point_x), int(point_y)), (int(wall_x), int(wall_y)), config.LINE_WIDTH)
                else:
                    # Draw solid line
                    pygame.draw.line(
                        self.screen,
                        line.color,
                        (int(point_x), int(point_y)),
                        (int(wall_x), int(wall_y)),
                        config.LINE_WIDTH
                    )
            except (TypeError, ValueError, OverflowError):
                # If there's any error drawing, skip this line
                continue
    
    def _draw_dashed_line(self, color, start_pos, end_pos, width):
        """Draw a dashed line between two points."""
        dash_length = 10
        gap_length = 5
        
        # Calculate line vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        line_length = math.sqrt(dx * dx + dy * dy)
        
        if line_length == 0:
            return
        
        # Normalize direction vector
        dx_norm = dx / line_length
        dy_norm = dy / line_length
        
        # Draw dashes
        current_distance = 0
        while current_distance < line_length:
            # Start of dash
            dash_start_x = start_pos[0] + current_distance * dx_norm
            dash_start_y = start_pos[1] + current_distance * dy_norm
            
            # End of dash
            dash_end_distance = min(current_distance + dash_length, line_length)
            dash_end_x = start_pos[0] + dash_end_distance * dx_norm
            dash_end_y = start_pos[1] + dash_end_distance * dy_norm
            
            # Draw the dash
            pygame.draw.line(self.screen, color, 
                           (int(dash_start_x), int(dash_start_y)), 
                           (int(dash_end_x), int(dash_end_y)), width)
            
            # Move to next dash (including gap)
            current_distance += dash_length + gap_length
    
    def render_ui(self):
        """Render UI elements (statistics and controls)."""
        # Draw UI information
        y_offset = config.UI_MARGIN
        
        energy_text = self.font.render(f"Energy Factor: {self.energy_factor:.2f}", True, config.UI_TEXT_COLOR)
        self.screen.blit(energy_text, (config.UI_MARGIN, y_offset))
        y_offset += config.UI_STATS_SPACING
        
        points_text = self.font.render(f"Points: {len(self.points)}", True, config.UI_TEXT_COLOR)
        self.screen.blit(points_text, (config.UI_MARGIN, y_offset))
        y_offset += config.UI_STATS_SPACING
        
        # Show active lines count (solid and dashed)
        active_lines = sum(1 for line in self.lines.values() if line.active)
        dashed_lines = sum(1 for line in self.lines.values() if line.active and line.dashed)
        solid_lines = active_lines - dashed_lines
        lines_text = self.font.render(f"Lines: {solid_lines} solid, {dashed_lines} dashed", True, config.UI_TEXT_COLOR)
        self.screen.blit(lines_text, (config.UI_MARGIN, y_offset))
        y_offset += config.UI_STATS_SPACING
        
        # Calculate average speed
        if self.points:
            valid_speeds = []
            for point in self.points:
                speed = point.get_speed()
                if math.isfinite(speed):
                    valid_speeds.append(speed)
            
            if valid_speeds:
                avg_speed = sum(valid_speeds) / len(valid_speeds)
                speed_text = self.font.render(f"Avg Speed: {avg_speed:.1f}", True, config.UI_TEXT_COLOR)
                self.screen.blit(speed_text, (config.UI_MARGIN, y_offset))
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