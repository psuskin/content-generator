import pygame
import numpy as np
import random
import math

class Point:
    """
    Represents a point in the simulation with position, velocity, and physics properties.
    """
    def __init__(self, x, y, vx, vy, radius=10, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.mass = 1.0  # All points have equal mass for simplicity
        self.max_speed = 2000  # Maximum allowed speed to prevent instability
    
    def validate_values(self):
        """Ensure all values are valid (not NaN or infinity)."""
        # Check for NaN or infinity and reset to safe values
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            self.x = 400  # Reset to center-ish position
            self.y = 300
        
        if not (math.isfinite(self.vx) and math.isfinite(self.vy)):
            self.vx = 0
            self.vy = 0
        
        # Clamp speed to maximum
        speed = self.get_speed()
        if speed > self.max_speed:
            self.set_speed(self.max_speed)
    
    def update(self, dt):
        """Update position based on velocity."""
        # Limit dt to prevent large jumps
        dt = min(dt, 0.016)  # Max 16ms time step
        
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
        if current_speed > 0 and math.isfinite(current_speed):
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
    def __init__(self, width=800, height=600, num_points=5, energy_factor=1.0):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Point Collision Simulation")
        
        # Simulation parameters
        self.num_points = num_points
        self.energy_factor = energy_factor
        self.base_speed = 100  # pixels per second
        
        # Circle boundary
        self.center_x = width // 2
        self.center_y = height // 2
        self.circle_radius = min(width, height) // 2 - 50
        
        # Initialize points
        self.points = []
        self.generate_points()
        
        # Clock for consistent frame rate
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Font for UI
        self.font = pygame.font.Font(None, 36)
    
    def generate_points(self):
        """Generate points with random positions and velocities, ensuring minimum distance."""
        self.points.clear()
        min_distance = 50  # Increased minimum distance for larger points
        max_attempts = 1000
        
        for i in range(self.num_points):
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Generate random position within circle
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, self.circle_radius - 30)  # More margin for larger points
                
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
                        random.randint(100, 255),
                        random.randint(100, 255),
                        random.randint(100, 255)
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
            max_speed_increase = current_speed * 1.5  # Max 50% increase per collision
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
            if distance < 0.1:  # Very close or overlapping
                distance = 0.1
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
            max_impulse = 500  # Limit impulse magnitude
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
            max_speed_increase1 = current_speed1 * 1.5
            max_speed_increase2 = current_speed2 * 1.5
            
            new_speed1 = min(new_speed1, max_speed_increase1)
            new_speed2 = min(new_speed2, max_speed_increase2)
            
            point1.set_speed(new_speed1)
            point2.set_speed(new_speed2)
            
            # Separate overlapping points
            overlap = (point1.radius + point2.radius) - distance
            separation = overlap / 2
            
            # Limit separation to prevent extreme position changes
            max_separation = 20
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
        dt = min(dt, 0.016)  # Max 16ms time step
        
        # Update point positions
        for point in self.points:
            point.update(dt)
        
        # Validate all points before collision detection
        for point in self.points:
            point.validate_values()
            
            # Keep points within reasonable bounds
            if (point.x < -100 or point.x > self.width + 100 or 
                point.y < -100 or point.y > self.height + 100):
                # Reset point to center if it goes too far out of bounds
                point.x = self.center_x + random.uniform(-50, 50)
                point.y = self.center_y + random.uniform(-50, 50)
        
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
        self.screen.fill((0, 0, 0))
        
        # Draw circle boundary
        pygame.draw.circle(
            self.screen, 
            (100, 100, 100), 
            (self.center_x, self.center_y), 
            self.circle_radius, 
            2
        )
        
        # Draw points
        for point in self.points:
            # Ensure coordinates are valid and within reasonable bounds
            try:
                x = int(point.x) if math.isfinite(point.x) else self.center_x
                y = int(point.y) if math.isfinite(point.y) else self.center_y
                radius = int(point.radius) if math.isfinite(point.radius) else 10
                
                # Clamp coordinates to screen bounds (with some buffer)
                x = max(-100, min(self.width + 100, x))
                y = max(-100, min(self.height + 100, y))
                radius = max(1, min(50, radius))
                
                pygame.draw.circle(
                    self.screen,
                    point.color,
                    (x, y),
                    radius
                )
            except (TypeError, ValueError, OverflowError):
                # If there's any error drawing, skip this point
                continue
        
        # Draw UI information
        energy_text = self.font.render(f"Energy Factor: {self.energy_factor:.2f}", True, (255, 255, 255))
        self.screen.blit(energy_text, (10, 10))
        
        points_text = self.font.render(f"Points: {len(self.points)}", True, (255, 255, 255))
        self.screen.blit(points_text, (10, 50))
        
        # Calculate average speed
        if self.points:
            valid_speeds = []
            for point in self.points:
                speed = point.get_speed()
                if math.isfinite(speed):
                    valid_speeds.append(speed)
            
            if valid_speeds:
                avg_speed = sum(valid_speeds) / len(valid_speeds)
                speed_text = self.font.render(f"Avg Speed: {avg_speed:.1f}", True, (255, 255, 255))
                self.screen.blit(speed_text, (10, 90))
                
                # Show max speed as well
                max_speed = max(valid_speeds)
                max_speed_text = self.font.render(f"Max Speed: {max_speed:.1f}", True, (255, 255, 255))
                self.screen.blit(max_speed_text, (10, 130))
        
        # Draw controls
        controls = [
            "Controls:",
            "↑/↓: Adjust energy factor",
            "+/-: Add/remove points",
            "R: Reset simulation",
            "ESC: Exit"
        ]
        
        for i, control in enumerate(controls):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            control_text = self.font.render(control, True, color)
            self.screen.blit(control_text, (10, self.height - 150 + i * 25))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_UP:
                    self.energy_factor = min(2.0, self.energy_factor + 0.05)
                
                elif event.key == pygame.K_DOWN:
                    self.energy_factor = max(0.5, self.energy_factor - 0.05)
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    if len(self.points) < 20:
                        self.num_points += 1
                        self.generate_points()
                
                elif event.key == pygame.K_MINUS:
                    if len(self.points) > 1:
                        self.num_points -= 1
                        self.generate_points()
                
                elif event.key == pygame.K_r:
                    self.generate_points()
    
    def run(self):
        """Main simulation loop."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            
            self.handle_events()
            self.update_physics(dt)
            self.render()
        
        pygame.quit()

def main():
    """Main function to start the simulation."""
    simulation = CircleSimulation(
        width=800,
        height=600,
        num_points=5,
        energy_factor=1.0
    )
    simulation.run()

if __name__ == "__main__":
    main()