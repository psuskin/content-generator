# Solution Space Optimizer for Point Collision Simulation
# Finds starting conditions that generate videos of at least 60s duration

import random
import math
import time
import json
import os
from datetime import datetime
from main import CircleSimulation, Point
import config

# Import color generation functions from main
from main import generate_distinct_colors

class StartingCondition:
    """
    Represents a set of starting conditions for the simulation.
    """
    def __init__(self, num_points=5, width=1920, height=1080, energy_factor=1.1):
        self.num_points = num_points
        self.width = width
        self.height = height
        self.energy_factor = energy_factor
        
        # Calculate circle parameters
        self.center_x = width // 2
        self.center_y = height // 2
        self.circle_radius = min(width, height) // 2 - config.CIRCLE_MARGIN
        
        # Store point configurations: [(x, y, vx, vy, color), ...]
        self.points = []
        self.fitness = 0.0  # Duration this configuration achieves
        self.tested = False
        
    def generate_random(self):
        """Generate random starting conditions."""
        self.points = []
        
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
                # Use smaller radius to ensure points start well inside
                max_radius = self.circle_radius - config.POINT_RADIUS - 20
                radius = random.uniform(config.POINT_RADIUS + 10, max_radius)
                
                x = self.center_x + radius * math.cos(angle)
                y = self.center_y + radius * math.sin(angle)
                
                # Check minimum distance from other points
                valid = True
                for px, py, _, _, _ in self.points:
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    # Generate random velocity
                    speed = random.uniform(config.BASE_SPEED * 0.5, config.BASE_SPEED * 1.5)
                    direction = random.uniform(0, 2 * math.pi)
                    vx = speed * math.cos(direction)
                    vy = speed * math.sin(direction)
                    
                    # Use pre-generated distinct color for this point
                    color = distinct_colors[i]
                    
                    self.points.append((x, y, vx, vy, color))
                    placed = True
                
                attempts += 1
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """Create a mutated version of this starting condition."""
        mutated = StartingCondition(self.num_points, self.width, self.height, self.energy_factor)
        mutated.points = []
        
        for x, y, vx, vy, color in self.points:
            new_x, new_y, new_vx, new_vy = x, y, vx, vy
            
            if random.random() < mutation_rate:
                # Mutate position (keeping within circle)
                pos_change = mutation_strength * config.MIN_DISTANCE_BETWEEN_POINTS
                new_x += random.uniform(-pos_change, pos_change)
                new_y += random.uniform(-pos_change, pos_change)
                
                # Ensure still within circle
                dist_from_center = math.sqrt((new_x - self.center_x)**2 + (new_y - self.center_y)**2)
                max_dist = self.circle_radius - config.POINT_RADIUS - 10
                if dist_from_center > max_dist:
                    # Project back into circle
                    angle = math.atan2(new_y - self.center_y, new_x - self.center_x)
                    new_x = self.center_x + max_dist * math.cos(angle)
                    new_y = self.center_y + max_dist * math.sin(angle)
            
            if random.random() < mutation_rate:
                # Mutate velocity
                speed_change = mutation_strength * config.BASE_SPEED
                new_vx += random.uniform(-speed_change, speed_change)
                new_vy += random.uniform(-speed_change, speed_change)
                
                # Limit speed
                speed = math.sqrt(new_vx**2 + new_vy**2)
                max_speed = config.BASE_SPEED * 2.0
                if speed > max_speed:
                    new_vx = (new_vx / speed) * max_speed
                    new_vy = (new_vy / speed) * max_speed
            
            mutated.points.append((new_x, new_y, new_vx, new_vy, color))
        
        return mutated
    
    def crossover(self, other):
        """Create offspring by combining this condition with another."""
        offspring = StartingCondition(self.num_points, self.width, self.height, self.energy_factor)
        offspring.points = []
        
        for i in range(self.num_points):
            # Choose parent randomly for each point
            if random.random() < 0.5:
                parent_point = self.points[i]
            else:
                parent_point = other.points[i] if i < len(other.points) else self.points[i]
            
            offspring.points.append(parent_point)
        
        return offspring
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'num_points': self.num_points,
            'width': self.width,
            'height': self.height,
            'energy_factor': self.energy_factor,
            'points': self.points,
            'fitness': self.fitness
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        condition = cls(data['num_points'], data['width'], data['height'], data['energy_factor'])
        condition.points = data['points']
        condition.fitness = data['fitness']
        condition.tested = True
        return condition

class SolutionSpaceOptimizer:
    """
    Optimizes starting conditions to find the solution space for long-duration videos.
    """
    def __init__(self, target_duration=60, population_size=50, width=1920, height=1080, 
                 energy_factor=1.1, num_points=5, simulation_speed_multiplier=4.0):
        self.target_duration = target_duration
        self.population_size = population_size
        self.width = width
        self.height = height
        self.energy_factor = energy_factor
        self.num_points = num_points
        self.simulation_speed_multiplier = simulation_speed_multiplier
        
        # Evolution parameters
        self.mutation_rate = 0.15
        self.mutation_strength = 0.25
        self.elite_ratio = 0.2  # Top 20% survive each generation
        self.crossover_ratio = 0.6  # 60% of new population from crossover
        
        # Solution tracking
        self.solution_space = []  # Successful conditions
        self.generation = 0
        self.population = []
        
        # Create results directory
        self.results_dir = "solution_space"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")
    
    def create_test_simulation(self, condition):
        """Create a simulation from a starting condition for testing."""
        # Temporarily override config settings for high resolution
        original_point_radius = config.POINT_RADIUS
        original_line_width = config.LINE_WIDTH
        original_circle_boundary_width = config.CIRCLE_BOUNDARY_WIDTH
        original_circle_margin = config.CIRCLE_MARGIN
        original_min_distance = config.MIN_DISTANCE_BETWEEN_POINTS
        original_base_speed = config.BASE_SPEED
        
        # Calculate scaling factor
        base_width = 800
        base_height = 600
        width_scale = self.width / base_width
        height_scale = self.height / base_height
        scale_factor = min(width_scale, height_scale)
        
        # Scale visual elements
        config.POINT_RADIUS = int(config.POINT_RADIUS * 1.75 * scale_factor)
        config.LINE_WIDTH = max(2, int(config.LINE_WIDTH * scale_factor))
        config.CIRCLE_BOUNDARY_WIDTH = max(2, int(config.CIRCLE_BOUNDARY_WIDTH * scale_factor))
        config.CIRCLE_MARGIN = int(config.CIRCLE_MARGIN * scale_factor)
        config.MIN_DISTANCE_BETWEEN_POINTS = int(config.MIN_DISTANCE_BETWEEN_POINTS * scale_factor)
        config.BASE_SPEED = config.BASE_SPEED / self.simulation_speed_multiplier
        
        # Create simulation without display
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        import pygame
        pygame.init()
        
        simulation = CircleSimulation(
            width=condition.width,
            height=condition.height,
            num_points=0,  # Start with no points, we'll add them manually
            energy_factor=condition.energy_factor
        )
        
        # Override screen to dummy surface
        simulation.screen = pygame.Surface((condition.width, condition.height))
        simulation.show_ui = False
        
        # Clear default points and add our custom ones
        simulation.points = []
        simulation.lines = {}
        simulation.line_id_counter = 0
        Point._id_counter = 0
        
        # Add points from condition
        for i, (x, y, vx, vy, color) in enumerate(condition.points):
            point = Point(x, y, vx, vy, color=color)
            simulation.points.append(point)
            simulation.generate_lines_for_point(point)
        
        # Restore original settings
        config.POINT_RADIUS = original_point_radius
        config.LINE_WIDTH = original_line_width
        config.CIRCLE_BOUNDARY_WIDTH = original_circle_boundary_width
        config.CIRCLE_MARGIN = original_circle_margin
        config.MIN_DISTANCE_BETWEEN_POINTS = original_min_distance
        config.BASE_SPEED = original_base_speed
        
        return simulation
    
    def test_condition_duration(self, condition):
        """Test how long a condition survives and return the duration."""
        if condition.tested:
            return condition.fitness
        
        simulation = self.create_test_simulation(condition)
        
        start_time = time.time()
        frame_count = 0
        fps = 60
        max_frames = int(self.target_duration * 1.5 * fps)  # Test up to 1.5x target duration
        
        base_dt = 1.0 / fps
        
        while frame_count < max_frames:
            # Check if simulation ended early
            if len(simulation.points) == 0:
                break
            
            if len(simulation.points) == 1 and frame_count < (self.target_duration * fps):
                break  # Boring scenario
            
            # Run accelerated physics
            for _ in range(int(self.simulation_speed_multiplier)):
                simulation.update_physics(base_dt)
            
            frame_count += 1
        
        duration = frame_count / fps
        condition.fitness = duration
        condition.tested = True
        
        elapsed_time = time.time() - start_time
        
        return duration
    
    def initialize_population(self):
        """Create initial random population."""
        print(f"Initializing population of {self.population_size} random conditions...")
        self.population = []
        
        for i in range(self.population_size):
            condition = StartingCondition(self.num_points, self.width, self.height, self.energy_factor)
            condition.generate_random()
            self.population.append(condition)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.population_size} conditions")
    
    def evaluate_population(self):
        """Test all conditions in the population."""
        print(f"Evaluating generation {self.generation}...")
        
        successful_count = 0
        total_duration = 0
        best_duration = 0
        
        for i, condition in enumerate(self.population):
            if not condition.tested:
                duration = self.test_condition_duration(condition)
                
                if duration >= self.target_duration:
                    successful_count += 1
                    if condition not in self.solution_space:
                        self.solution_space.append(condition)
                
                total_duration += duration
                best_duration = max(best_duration, duration)
                
                if (i + 1) % 10 == 0:
                    avg_duration = total_duration / (i + 1)
                    print(f"  Tested {i + 1}/{len(self.population)}: avg={avg_duration:.1f}s, best={best_duration:.1f}s, solutions={successful_count}")
        
        avg_duration = total_duration / len(self.population)
        print(f"Generation {self.generation} complete: avg={avg_duration:.1f}s, best={best_duration:.1f}s")
        print(f"Successful conditions: {successful_count}/{len(self.population)} ({100*successful_count/len(self.population):.1f}%)")
        print(f"Total solution space size: {len(self.solution_space)}")
    
    def evolve_population(self):
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness (duration)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Select elites (top performers)
        elite_count = int(self.population_size * self.elite_ratio)
        elites = self.population[:elite_count]
        
        # Create new population
        new_population = elites.copy()
        
        # Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_ratio and len(elites) >= 2:
                # Crossover
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                offspring = parent1.crossover(parent2)
            else:
                # Mutation of elite
                parent = random.choice(elites)
                offspring = parent.mutate(self.mutation_rate, self.mutation_strength)
            
            new_population.append(offspring)
        
        self.population = new_population
        self.generation += 1
    
    def save_solution_space(self):
        """Save the current solution space to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solution_space_{timestamp}_gen{self.generation}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        data = {
            'generation': self.generation,
            'target_duration': self.target_duration,
            'num_solutions': len(self.solution_space),
            'config': {
                'num_points': self.num_points,
                'width': self.width,
                'height': self.height,
                'energy_factor': self.energy_factor,
                'simulation_speed_multiplier': self.simulation_speed_multiplier
            },
            'solutions': [condition.to_dict() for condition in self.solution_space]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.solution_space)} solutions to {filename}")
        return filepath
    
    def load_solution_space(self, filepath):
        """Load solution space from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.solution_space = [StartingCondition.from_dict(sol) for sol in data['solutions']]
        self.generation = data['generation']
        
        print(f"Loaded {len(self.solution_space)} solutions from {filepath}")
    
    def optimize(self, max_generations=20, min_solutions=50):
        """Run optimization to find solution space."""
        print(f"Starting optimization for {max_generations} generations...")
        print(f"Target: Find conditions that achieve {self.target_duration}s+ duration")
        print(f"Goal: Collect at least {min_solutions} solutions")
        print("=" * 60)
        
        # Initialize if needed
        if not self.population:
            self.initialize_population()
        
        for gen in range(max_generations):
            print(f"\n=== GENERATION {self.generation} ===")
            
            # Evaluate current population
            self.evaluate_population()
            
            # Save progress
            if self.generation % 5 == 0 or len(self.solution_space) >= min_solutions:
                self.save_solution_space()
            
            # Check if we have enough solutions
            if len(self.solution_space) >= min_solutions:
                print(f"\n🎉 Success! Found {len(self.solution_space)} solutions (target: {min_solutions})")
                break
            
            # Evolve to next generation
            if gen < max_generations - 1:  # Don't evolve on last iteration
                self.evolve_population()
        
        # Final save
        final_file = self.save_solution_space()
        
        print(f"\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE")
        print(f"Generations: {self.generation}")
        print(f"Solutions found: {len(self.solution_space)}")
        print(f"Success rate: {100 * len(self.solution_space) / (self.generation * self.population_size):.2f}%")
        print(f"Final results saved to: {final_file}")
        
        return self.solution_space

def main():
    """Main function to run solution space optimization."""
    # Configuration
    TARGET_DURATION = 60  # seconds
    NUM_POINTS = 5
    ENERGY_FACTOR = 1.1  # Match the manual simulation that achieved 60+ seconds
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    SIMULATION_SPEED_MULTIPLIER = 4.0
    
    # Optimization parameters
    POPULATION_SIZE = 30  # Smaller for faster iteration
    MAX_GENERATIONS = 25
    MIN_SOLUTIONS = 100  # Target number of solutions to find
    
    print("Point Collision Simulation - Solution Space Optimizer")
    print("=" * 60)
    print(f"Searching for starting conditions that achieve {TARGET_DURATION}s+ duration")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Max generations: {MAX_GENERATIONS}")
    print(f"Target solutions: {MIN_SOLUTIONS}")
    print("-" * 60)
    
    # Create optimizer
    optimizer = SolutionSpaceOptimizer(
        target_duration=TARGET_DURATION,
        population_size=POPULATION_SIZE,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        energy_factor=ENERGY_FACTOR,
        num_points=NUM_POINTS,
        simulation_speed_multiplier=SIMULATION_SPEED_MULTIPLIER
    )
    
    # Run optimization
    solutions = optimizer.optimize(MAX_GENERATIONS, MIN_SOLUTIONS)
    
    if solutions:
        print(f"\n📊 Solution Space Analysis:")
        durations = [sol.fitness for sol in solutions]
        print(f"  Average duration: {sum(durations)/len(durations):.1f}s")
        print(f"  Best duration: {max(durations):.1f}s")
        print(f"  Worst duration: {min(durations):.1f}s")
        print(f"\n✓ Ready to generate videos from solution space!")
    else:
        print(f"\n❌ No solutions found. Try adjusting parameters.")

if __name__ == "__main__":
    main()
