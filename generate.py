# Video Generation Script for Point Collision Simulation
# Generates videos of simulations without UI, with specific parameters and conditions

import pygame
import cv2
import numpy as np
import os
import time
import datetime
import json
import random
from main import CircleSimulation, Point
import config

class VideoGenerator:
    """
    Generates videos of the simulation without UI display.
    Can use either random starting conditions or optimized solution space.
    """
    def __init__(self, energy_factor=1.01, width=800, height=600, fps=60, simulation_speed_multiplier=4.0, solution_space_file=None):
        self.energy_factor = energy_factor
        self.width = width
        self.height = height
        self.fps = fps
        self.simulation_speed_multiplier = simulation_speed_multiplier  # How much faster to run simulation
        self.target_duration = 60  # seconds to run before speed boost
        self.final_duration = 70   # total duration including speed boost
        
        # Solution space for optimized generation
        self.solution_space = []
        self.solution_space_file = solution_space_file
        if solution_space_file and os.path.exists(solution_space_file):
            self.load_solution_space(solution_space_file)
        
        # Create recordings directory
        self.recordings_dir = "recordings"
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
            print(f"Created recordings directory: {self.recordings_dir}")
        
        # Initialize pygame without display
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        
        # Create a surface for rendering (no window)
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
    
    def create_simulation(self, num_points=5):
        """Create a new simulation instance for video generation."""
        # Calculate scaling factor based on resolution increase
        base_width = 800
        base_height = 600
        width_scale = self.width / base_width
        height_scale = self.height / base_height
        scale_factor = min(width_scale, height_scale)  # Use smaller scale to maintain aspect ratio
        
        # Temporarily override visual settings for higher resolution
        original_point_radius = config.POINT_RADIUS
        original_line_width = config.LINE_WIDTH
        original_circle_boundary_width = config.CIRCLE_BOUNDARY_WIDTH
        original_circle_margin = config.CIRCLE_MARGIN
        original_min_distance = config.MIN_DISTANCE_BETWEEN_POINTS
        
        # Scale up visual elements proportionally
        config.POINT_RADIUS = int(config.POINT_RADIUS * 1.75 * scale_factor)  # 75% bigger + resolution scaling
        config.LINE_WIDTH = max(2, int(config.LINE_WIDTH * scale_factor))
        config.CIRCLE_BOUNDARY_WIDTH = max(2, int(config.CIRCLE_BOUNDARY_WIDTH * scale_factor))
        config.CIRCLE_MARGIN = int(config.CIRCLE_MARGIN * scale_factor)
        config.MIN_DISTANCE_BETWEEN_POINTS = int(config.MIN_DISTANCE_BETWEEN_POINTS * scale_factor)
        
        # Create a modified simulation that doesn't show UI
        simulation = CircleSimulation(
            width=self.width,
            height=self.height,
            num_points=num_points,
            energy_factor=self.energy_factor
        )
        
        # Restore original settings
        config.POINT_RADIUS = original_point_radius
        config.LINE_WIDTH = original_line_width
        config.CIRCLE_BOUNDARY_WIDTH = original_circle_boundary_width
        config.CIRCLE_MARGIN = original_circle_margin
        config.MIN_DISTANCE_BETWEEN_POINTS = original_min_distance
        
        # Override the screen to use our surface
        simulation.screen = self.screen
        
        # Disable UI display
        simulation.show_ui = False
        
        return simulation
    
    def load_solution_space(self, filepath):
        """Load optimized solution space from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.solution_space = data['solutions']
            print(f"Loaded {len(self.solution_space)} optimized solutions from {filepath}")
            
            # Verify compatibility
            if data['config']['width'] != self.width or data['config']['height'] != self.height:
                print(f"Warning: Solution space resolution ({data['config']['width']}x{data['config']['height']}) differs from target ({self.width}x{self.height})")
            
            if abs(data['config']['energy_factor'] - self.energy_factor) > 0.01:
                print(f"Warning: Solution space energy factor ({data['config']['energy_factor']}) differs from target ({self.energy_factor})")
            
        except Exception as e:
            print(f"Error loading solution space: {e}")
            self.solution_space = []
    
    def create_simulation_from_solution(self, solution_data):
        """Create a simulation from solution space data."""
        # Calculate scaling factor based on resolution increase
        base_width = 800
        base_height = 600
        width_scale = self.width / base_width
        height_scale = self.height / base_height
        scale_factor = min(width_scale, height_scale)
        
        # Temporarily override visual settings for higher resolution
        original_point_radius = config.POINT_RADIUS
        original_line_width = config.LINE_WIDTH
        original_circle_boundary_width = config.CIRCLE_BOUNDARY_WIDTH
        original_circle_margin = config.CIRCLE_MARGIN
        original_min_distance = config.MIN_DISTANCE_BETWEEN_POINTS
        original_base_speed = config.BASE_SPEED
        
        # Scale up visual elements proportionally
        config.POINT_RADIUS = int(config.POINT_RADIUS * 1.75 * scale_factor)
        config.LINE_WIDTH = max(2, int(config.LINE_WIDTH * scale_factor))
        config.CIRCLE_BOUNDARY_WIDTH = max(2, int(config.CIRCLE_BOUNDARY_WIDTH * scale_factor))
        config.CIRCLE_MARGIN = int(config.CIRCLE_MARGIN * scale_factor)
        config.MIN_DISTANCE_BETWEEN_POINTS = int(config.MIN_DISTANCE_BETWEEN_POINTS * scale_factor)
        
        # Scale BASE_SPEED proportionally for video recording so points move at same relative speed
        config.BASE_SPEED = int(config.BASE_SPEED * scale_factor)
        
        # Create simulation with no default points
        simulation = CircleSimulation(
            width=self.width,
            height=self.height,
            num_points=0,  # We'll add points manually
            energy_factor=self.energy_factor
        )
        
        # Clear default points and set up for custom points
        simulation.points = []
        simulation.lines = {}
        simulation.line_id_counter = 0
        Point._id_counter = 0
        
        # Add points from solution data
        for x, y, vx, vy, color in solution_data['points']:
            point = Point(x, y, vx, vy, color=tuple(color))
            simulation.points.append(point)
            simulation.generate_lines_for_point(point)
        
        # Restore original settings
        config.POINT_RADIUS = original_point_radius
        config.LINE_WIDTH = original_line_width
        config.CIRCLE_BOUNDARY_WIDTH = original_circle_boundary_width
        config.CIRCLE_MARGIN = original_circle_margin
        config.MIN_DISTANCE_BETWEEN_POINTS = original_min_distance
        config.BASE_SPEED = original_base_speed
        
        # Override the screen to use our surface
        simulation.screen = self.screen
        
        # Disable UI display
        simulation.show_ui = False
        
        return simulation
    
    def surface_to_array(self, surface):
        """Convert pygame surface to numpy array for OpenCV."""
        # Get the raw surface data
        w, h = surface.get_size()
        raw = pygame.image.tostring(surface, 'RGB')
        
        # Convert to numpy array and reshape
        array = np.frombuffer(raw, dtype=np.uint8)
        array = array.reshape((h, w, 3))
        
        # OpenCV uses BGR, pygame uses RGB
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        return array
    
    def test_simulation_viability(self, simulation):
        """
        Test if a simulation will last the full duration without recording video.
        Returns True if simulation is viable, False otherwise.
        This runs at accelerated speed for quick testing.
        """
        start_time = time.time()
        frame_count = 0
        
        # Calculate time step for accelerated simulation
        base_dt = 1.0 / self.fps
        
        # Calculate total frames needed for target duration (in normal time)
        target_frames = int(self.target_duration * self.fps)
        
        print(f"Testing simulation viability with {len(simulation.points)} points, energy factor {self.energy_factor}")
        print(f"Testing at {self.simulation_speed_multiplier}x speed for faster evaluation...")
        
        while frame_count < target_frames:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_time
            
            # Calculate current video time based on frame count (normal speed equivalent)
            video_time = frame_count / self.fps
            
            # Check if all points are gone (simulation ended early)
            if len(simulation.points) == 0:
                print(f"✗ Test failed at {video_time:.1f}s - all points eliminated (real time: {elapsed_real_time:.1f}s)")
                return False
            
            # Check if only one point remains before reaching target duration
            if len(simulation.points) == 1:
                print(f"✗ Test failed at {video_time:.1f}s - only one point remaining (real time: {elapsed_real_time:.1f}s)")
                return False
            
            # Run accelerated physics with larger time step for faster testing
            # This is more accurate than multiple small updates
            accelerated_dt = base_dt * self.simulation_speed_multiplier
            simulation.update_physics(accelerated_dt)
            
            frame_count += 1
            
            # Print progress every 15 seconds of video time (less frequent for testing)
            if frame_count % (self.fps * 15) == 0:
                actual_speedup = video_time / elapsed_real_time if elapsed_real_time > 0 else 0
                print(f"  Test progress: {video_time:.1f}s video time ({elapsed_real_time:.1f}s real, {actual_speedup:.1f}x speed), Points: {len(simulation.points)}")
        
        elapsed_real_time = time.time() - start_time
        print(f"✓ Test passed: {self.target_duration:.0f}s simulation with {len(simulation.points)} points (real time: {elapsed_real_time:.1f}s)")
        return True
    
    def run_simulation_for_video(self, simulation, video_writer):
        """
        Run a simulation and record it to video in REAL TIME.
        This records the full 70-second video without any acceleration or speed boosts.
        Returns True when completed successfully.
        """
        start_time = time.time()
        frame_count = 0
        
        # Calculate time step for normal speed recording
        base_dt = 1.0 / self.fps
        
        # Calculate total frames needed
        target_frames = int(self.target_duration * self.fps)  # 60 seconds
        final_frames = int(self.final_duration * self.fps)   # 70 seconds
        
        print(f"Recording {self.final_duration}s video in real time...")
        print(f"Expected recording time: {self.final_duration:.0f}s real time")
        
        while frame_count < final_frames:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_time
            
            # Calculate current video time
            video_time = frame_count / self.fps
            
            # Remove speed cap at 60 seconds for chaotic finale
            if frame_count == target_frames:
                print(f"Removing speed cap at {video_time:.0f}s for chaotic finale...")
                original_max_speed = config.MAX_POINT_SPEED
                config.MAX_POINT_SPEED = float('inf')
                for point in simulation.points:
                    point.max_speed = float('inf')
            
            # Run physics at normal speed (no acceleration)
            simulation.update_physics(base_dt)
            
            # Render to our surface
            simulation.render()
            
            # Convert surface to array and write to video
            frame_array = self.surface_to_array(self.screen)
            video_writer.write(frame_array)
            
            frame_count += 1
            
            # Print progress every 10 seconds of video time
            if frame_count % (self.fps * 10) == 0:
                print(f"Recording progress: {video_time:.0f}s video time, {len(simulation.points)} points remaining")
        
        elapsed_real_time = time.time() - start_time
        print(f"✓ Video recording completed: {self.final_duration:.0f}s video (real time: {elapsed_real_time:.1f}s)")
        return True
    
    def extract_simulation_state(self, simulation):
        """Extract the exact state of a simulation for replication."""
        state = {
            'points': [],
            'energy_factor': simulation.energy_factor,
            'width': simulation.width,
            'height': simulation.height
        }
        
        for point in simulation.points:
            state['points'].append((point.x, point.y, point.vx, point.vy, point.color))
        
        return state

    def generate_video(self, num_points=5, attempt_number=1, use_solution_space=True):
        """
        Generate a single video file using two-phase approach:
        1. Test simulation viability without recording
        2. Record video only if simulation is viable
        
        If use_solution_space=True and solution space is available, uses optimized conditions.
        Otherwise falls back to random generation.
        
        Returns the filename if successful, None if simulation ended early.
        """
        print(f"\nAttempt {attempt_number}: ", end="")
        
        # Choose generation method and get simulation state
        if use_solution_space and self.solution_space:
            print("Using optimized solution from solution space...")
            # Select random solution from solution space
            solution_data = random.choice(self.solution_space)
            simulation_state = solution_data  # Already in the right format
            
            # Phase 1: Test with solution space data (should pass, but verify)
            test_simulation = self.create_simulation_from_solution(solution_data)
            is_viable = self.test_simulation_viability(test_simulation)
            
            if not is_viable:
                print(f"✗ Unexpected: Solution space condition failed test")
                return None
            
        else:
            print("Testing random simulation viability...")
            # Phase 1: Test simulation without recording - extract its exact state
            test_simulation = self.create_simulation(num_points)
            simulation_state = self.extract_simulation_state(test_simulation)
            
            is_viable = self.test_simulation_viability(test_simulation)
            
            if not is_viable:
                print(f"✗ Random simulation test failed, skipping video recording")
                return None
        
        print(f"✓ Simulation test passed, proceeding to record video...")
        
        # Phase 2: Create identical simulation for recording
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        method = "optimized" if (use_solution_space and self.solution_space) else "random"
        filename = f"simulation_{timestamp}_attempt{attempt_number}_{method}_ef{self.energy_factor:.3f}_p{num_points}.mp4"
        filepath = os.path.join(self.recordings_dir, filename)
        
        print(f"Target file: {filename}")
        
        # Create video writer with higher quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {filepath}")
            return None
        
        try:
            # Create recording simulation using EXACT same state as test simulation
            recording_simulation = self.create_simulation_from_solution(simulation_state)
            
            success = self.run_simulation_for_video(recording_simulation, video_writer)
            
            # Clean up
            video_writer.release()
            
            if success:
                print(f"✓ Successfully generated: {filename}")
                return filename
            else:
                print(f"✗ Video recording failed, deleting: {filename}")
                # Delete the incomplete video file
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return None
                
        except Exception as e:
            print(f"Error during video recording: {e}")
            video_writer.release()
            # Clean up incomplete file
            try:
                os.remove(filepath)
            except OSError:
                pass
            return None
    
    def generate_videos(self, num_videos=1, num_points=5, max_attempts_per_video=10, use_solution_space=True):
        """
        Generate multiple videos, retrying until successful or max attempts reached.
        
        If use_solution_space=True and solution space is loaded, uses optimized conditions
        for much higher success rate. Otherwise uses random generation.
        """
        successful_videos = []
        
        print(f"Starting video generation process...")
        print(f"Target: {num_videos} successful videos")
        print(f"Parameters: {num_points} points, energy factor {self.energy_factor}")
        
        if use_solution_space and self.solution_space:
            print(f"Using optimized solution space with {len(self.solution_space)} solutions")
            print(f"Expected success rate: ~95%+ (vs ~5% for random)")
        else:
            print(f"Using random generation (low success rate expected)")
        
        print(f"Duration: {self.target_duration}s normal + {self.final_duration - self.target_duration}s boosted = {self.final_duration}s total")
        print(f"Output directory: {os.path.abspath(self.recordings_dir)}")
        print("-" * 60)
        
        for video_num in range(1, num_videos + 1):
            print(f"\n=== VIDEO {video_num}/{num_videos} ===")
            
            success = False
            for attempt in range(1, max_attempts_per_video + 1):
                filename = self.generate_video(num_points, attempt, use_solution_space)
                
                if filename:
                    successful_videos.append(filename)
                    success = True
                    break
                else:
                    if use_solution_space and self.solution_space:
                        print(f"Attempt {attempt} failed unexpectedly (solution space should have high success rate)")
                    else:
                        print(f"Attempt {attempt} failed, retrying...")
            
            if not success:
                print(f"Failed to generate video {video_num} after {max_attempts_per_video} attempts")
        
        print("\n" + "=" * 60)
        print(f"GENERATION COMPLETE")
        print(f"Successful videos: {len(successful_videos)}/{num_videos}")
        
        if successful_videos:
            print("\nGenerated files:")
            for filename in successful_videos:
                filepath = os.path.join(self.recordings_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"  - {filename} ({file_size:.1f} MB)")
        
        return successful_videos

def main():
    """Main function to run video generation."""
    # Configuration parameters
    ENERGY_FACTOR = 1.1  # Match the manual simulation that achieved 60+ seconds
    NUM_POINTS = 5
    NUM_VIDEOS = 10  # Number of successful videos to generate
    VIDEO_WIDTH = 1920   # Full HD width for crisp quality
    VIDEO_HEIGHT = 1080  # Full HD height for crisp quality
    VIDEO_FPS = 60
    MAX_ATTEMPTS_PER_VIDEO = 1000  # Should be much lower with correct energy factor
    SIMULATION_SPEED_MULTIPLIER = 4.0  # How much faster to run TESTING (video recording is always real-time)
    
    # Solution space file (set to None to use random generation)
    SOLUTION_SPACE_FILE = "solution_space/solution_space_latest.json"  # Update this path as needed
    
    print("Point Collision Simulation - Video Generator")
    print("=" * 50)
    print("Two-phase generation process:")
    print("1. Test simulation viability at high speed (no recording)")
    print("2. Record video in REAL TIME for viable simulations")
    
    # Check for solution space
    if SOLUTION_SPACE_FILE and os.path.exists(SOLUTION_SPACE_FILE):
        print("3. Using OPTIMIZED solution space for high success rate")
        use_solution_space = True
    else:
        print("3. Using RANDOM generation (low success rate expected)")
        use_solution_space = False
        if SOLUTION_SPACE_FILE:
            print(f"   (Solution space file not found: {SOLUTION_SPACE_FILE})")
        print(f"   Run optimizer.py first to generate solution space!")
    
    print("Success criteria:")
    print("- Must survive 60+ seconds with at least 2 points")
    print("- Speed cap removed at 60s for final 10s of chaos")
    print("- Simulations with <2 points before 60s are discarded")
    print(f"- Testing speed: {SIMULATION_SPEED_MULTIPLIER}x (video recording: 1x real-time)")
    print("-" * 50)
    
    # Create video generator
    generator = VideoGenerator(
        energy_factor=ENERGY_FACTOR,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        fps=VIDEO_FPS,
        simulation_speed_multiplier=SIMULATION_SPEED_MULTIPLIER,
        solution_space_file=SOLUTION_SPACE_FILE if use_solution_space else None
    )
    
    # Generate videos
    successful_videos = generator.generate_videos(
        num_videos=NUM_VIDEOS,
        num_points=NUM_POINTS,
        max_attempts_per_video=MAX_ATTEMPTS_PER_VIDEO,
        use_solution_space=use_solution_space
    )
    
    if successful_videos:
        print(f"\n🎉 Successfully generated {len(successful_videos)} videos!")
        if use_solution_space:
            print(f"   Using optimized solution space significantly improved success rate!")
    else:
        print(f"\n❌ No videos were successfully generated.")
        if not use_solution_space:
            print(f"   Consider running optimizer.py first to generate a solution space.")
    
    print("\nVideo generation complete.")
    
    return successful_videos

if __name__ == "__main__":
    main()
