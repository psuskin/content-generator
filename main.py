import pygame
import numpy as np
import random
import math
import config

class Point:
    """
    Represents a point in the simulation with position, velocity, and physics properties.
    """
    def __init__(self, x, y, vx, vy, radius=None, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius if radius is not None else config.POINT_RADIUS
        self.color = color
        self.mass = config.POINT_MASS
        self.max_speed = config.MAX_POINT_SPEED
    
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
        # Limit dt to prevent large jumps
        dt = min(dt, config.MAX_TIME_STEP)
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Validate values after update
        self.validate_values()
    
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
        
        # Initialize points
        self.points = []
        self.generate_points()
        
        # Clock for consistent frame rate
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Font for UI
        self.font = pygame.font.Font(None, config.UI_FONT_SIZE)
    
    def generate_points(self):
        """Generate points with random positions and velocities, ensuring minimum distance."""
        self.points.clear()
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
                    placed = True
                
                attempts += 1
            
            if not placed:
                print(f"Warning: Could not place point {i+1} after {max_attempts} attempts")
    
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
            
            # Reflect velocity
            dot_product = point.vx * nx + point.vy * ny
            point.vx -= 2 * dot_product * nx
            point.vy -= 2 * dot_product * ny
            
            # Apply energy factor with safety limits
            current_speed = point.get_speed()
            new_speed = current_speed * self.energy_factor
            
            # Limit maximum speed increase per collision
            max_speed_increase = current_speed * config.MAX_SPEED_INCREASE_PER_COLLISION
            new_speed = min(new_speed, max_speed_increase)
            
            point.set_speed(new_speed)
            
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
            
            # Final validation
            point.validate_values()
    
    def check_point_collision(self, point1, point2):
        """Check and handle collision between two points."""
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
            
            # Collision impulse (assuming equal masses)
            impulse = 2 * dvn / (point1.mass + point2.mass)
            
            # Limit impulse to prevent extreme velocity changes
            max_impulse = config.MAX_IMPULSE
            impulse = max(-max_impulse, min(max_impulse, impulse))
            
            # Update velocities
            point1.vx += impulse * point2.mass * nx
            point1.vy += impulse * point2.mass * ny
            point2.vx -= impulse * point1.mass * nx
            point2.vy -= impulse * point1.mass * ny
            
            # Apply energy factor with safety limits
            current_speed1 = point1.get_speed()
            current_speed2 = point2.get_speed()
            
            new_speed1 = current_speed1 * self.energy_factor
            new_speed2 = current_speed2 * self.energy_factor
            
            # Limit maximum speed increase per collision
            max_speed_increase1 = current_speed1 * config.MAX_SPEED_INCREASE_PER_COLLISION
            max_speed_increase2 = current_speed2 * config.MAX_SPEED_INCREASE_PER_COLLISION
            
            new_speed1 = min(new_speed1, max_speed_increase1)
            new_speed2 = min(new_speed2, max_speed_increase2)
            
            point1.set_speed(new_speed1)
            point2.set_speed(new_speed2)
            
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
        """Update all physics calculations."""
        # Limit time step to prevent instability
        dt = min(dt, config.MAX_TIME_STEP)
        
        # Update point positions
        for point in self.points:
            point.update(dt)
        
        # Validate all points before collision detection
        for point in self.points:
            point.validate_values()
            
            # Keep points within reasonable bounds
            if (point.x < -config.BOUNDS_BUFFER or point.x > self.width + config.BOUNDS_BUFFER or 
                point.y < -config.BOUNDS_BUFFER or point.y > self.height + config.BOUNDS_BUFFER):
                # Reset point to center if it goes too far out of bounds
                point.x = self.center_x + random.uniform(-config.RESET_RANDOMNESS, config.RESET_RANDOMNESS)
                point.y = self.center_y + random.uniform(-config.RESET_RANDOMNESS, config.RESET_RANDOMNESS)
        
        # Check circle collisions
        for point in self.points:
            self.check_circle_collision(point)
        
        # Check point-to-point collisions
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                self.check_point_collision(self.points[i], self.points[j])
    
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