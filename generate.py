# Video Generation Script for Point Collision Simulation
# Generates videos of simulations without UI, with specific parameters and conditions

import pygame
import cv2
import numpy as np
import os
import time
import datetime
from main import CircleSimulation, Point
import config

class VideoGenerator:
    """
    Generates videos of the simulation without UI display.
    """
    def __init__(self, energy_factor=1.01, width=800, height=600, fps=60, simulation_speed_multiplier=4.0):
        self.energy_factor = energy_factor
        self.width = width
        self.height = height
        self.fps = fps
        self.simulation_speed_multiplier = simulation_speed_multiplier  # How much faster to run simulation
        self.target_duration = 60  # seconds to run before speed boost
        self.final_duration = 70   # total duration including speed boost
        
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
        """
        start_time = time.time()
        frame_count = 0
        speed_boosted = False
        
        # Calculate time step for accelerated simulation
        base_dt = 1.0 / self.fps
        
        # Calculate total frames needed for target video duration
        target_frames = int(self.target_duration * self.fps)
        final_frames = int(self.final_duration * self.fps)
        
        print(f"Testing simulation viability with {len(simulation.points)} points, energy factor {self.energy_factor}")
        print(f"Simulation speed: {self.simulation_speed_multiplier}x (testing {self.final_duration}s in {self.final_duration/self.simulation_speed_multiplier:.1f}s real time)")
        
        while frame_count < final_frames:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_time
            
            # Calculate current video time based on frame count
            video_time = frame_count / self.fps
            
            # Check if all points are gone (simulation ended early)
            if len(simulation.points) == 0:
                print(f"Simulation test failed at {video_time:.1f}s video time - all points eliminated (real time: {elapsed_real_time:.1f}s)")
                return False
            
            # Check if only one point remains before reaching target duration (boring scenario)
            if len(simulation.points) == 1 and video_time < self.target_duration:
                print(f"Simulation test failed at {video_time:.1f}s video time - only one point remaining (boring) (real time: {elapsed_real_time:.1f}s)")
                return False
            
            # Apply speed boost at target duration
            if frame_count >= target_frames and not speed_boosted:
                print(f"Applying speed boost at {video_time:.1f}s video time (real time: {elapsed_real_time:.1f}s)")
                # Set maximum speed to infinity (remove speed limit)
                for point in simulation.points:
                    point.max_speed = float('inf')
                # Also update the config maximum speed
                original_max_speed = config.MAX_POINT_SPEED
                config.MAX_POINT_SPEED = float('inf')
                speed_boosted = True
            
            # Run multiple physics updates per frame to achieve acceleration
            for _ in range(int(self.simulation_speed_multiplier)):
                simulation.update_physics(base_dt)
            
            frame_count += 1
            
            # Print progress every 15 seconds of video time (less frequent for testing)
            if frame_count % (self.fps * 15) == 0:
                actual_speedup = video_time / elapsed_real_time if elapsed_real_time > 0 else 0
                print(f"Test progress: {video_time:.1f}s video time ({elapsed_real_time:.1f}s real, {actual_speedup:.1f}x speed), Points: {len(simulation.points)}")
        
        # Restore original max speed if it was changed
        if speed_boosted:
            config.MAX_POINT_SPEED = original_max_speed
        
        print(f"Simulation test PASSED: {self.final_duration:.1f}s video time (real time: {elapsed_real_time:.1f}s)")
        return True
    
    def run_simulation_for_video(self, simulation, video_writer):
        """
        Run a simulation and record it to video.
        This assumes the simulation has already been tested and is viable.
        Returns True when completed successfully.
        """
        start_time = time.time()
        frame_count = 0
        speed_boosted = False
        
        # Calculate time step
        base_dt = 1.0 / self.fps
        
        # Calculate total frames needed
        target_frames = int(self.target_duration * self.fps)
        final_frames = int(self.final_duration * self.fps)
        
        print(f"Recording simulation to video...")
        print(f"Expected recording time: {self.final_duration / self.simulation_speed_multiplier:.1f}s real time")
        
        while frame_count < final_frames:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_time
            
            # Calculate current video time based on frame count
            video_time = frame_count / self.fps
            
            # Apply speed boost at target duration
            if frame_count >= target_frames and not speed_boosted:
                print(f"Applying speed boost for recording at {video_time:.1f}s video time")
                # Set maximum speed to infinity (remove speed limit)
                for point in simulation.points:
                    point.max_speed = float('inf')
                # Also update the config maximum speed
                original_max_speed = config.MAX_POINT_SPEED
                config.MAX_POINT_SPEED = float('inf')
                speed_boosted = True
            
            # Run multiple physics updates per frame to achieve acceleration
            for _ in range(int(self.simulation_speed_multiplier)):
                simulation.update_physics(base_dt)
            
            # Render to our surface (only once per frame, not per physics update)
            simulation.render()
            
            # Convert surface to array and write to video
            frame_array = self.surface_to_array(self.screen)
            video_writer.write(frame_array)
            
            frame_count += 1
            
            # Print progress every 20 seconds of video time
            if frame_count % (self.fps * 20) == 0:
                actual_speedup = video_time / elapsed_real_time if elapsed_real_time > 0 else 0
                print(f"Recording progress: {video_time:.1f}s video time ({elapsed_real_time:.1f}s real, {actual_speedup:.1f}x speed)")
        
        # Restore original max speed if it was changed
        if speed_boosted:
            config.MAX_POINT_SPEED = original_max_speed
        
        print(f"Video recording completed: {self.final_duration:.1f}s video time (real time: {elapsed_real_time:.1f}s)")
        return True
    
    def generate_video(self, num_points=5, attempt_number=1):
        """
        Generate a single video file using two-phase approach:
        1. Test simulation viability without recording
        2. Record video only if simulation is viable
        Returns the filename if successful, None if simulation ended early.
        """
        print(f"\nAttempt {attempt_number}: Testing simulation viability...")
        
        # Phase 1: Test simulation without recording
        test_simulation = self.create_simulation(num_points)
        is_viable = self.test_simulation_viability(test_simulation)
        
        if not is_viable:
            print(f"✗ Simulation test failed, skipping video recording")
            # return None
        
        print(f"✓ Simulation test passed, proceeding to record video...")
        
        # Phase 2: Create new simulation and record video
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}_attempt{attempt_number}_ef{self.energy_factor:.3f}_p{num_points}.mp4"
        filepath = os.path.join(self.recordings_dir, filename)
        
        print(f"Target file: {filename}")
        
        # Create video writer with higher quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {filepath}")
            return None
        
        try:
            # Create a fresh simulation for recording (same seed would be ideal but not critical)
            recording_simulation = self.create_simulation(num_points)
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
    
    def generate_videos(self, num_videos=1, num_points=5, max_attempts_per_video=10):
        """
        Generate multiple videos, retrying until successful or max attempts reached.
        """
        successful_videos = []
        
        print(f"Starting video generation process...")
        print(f"Target: {num_videos} successful videos")
        print(f"Parameters: {num_points} points, energy factor {self.energy_factor}")
        print(f"Duration: {self.target_duration}s normal + {self.final_duration - self.target_duration}s boosted = {self.final_duration}s total")
        print(f"Output directory: {os.path.abspath(self.recordings_dir)}")
        print("-" * 60)
        
        for video_num in range(1, num_videos + 1):
            print(f"\n=== VIDEO {video_num}/{num_videos} ===")
            
            success = False
            for attempt in range(1, max_attempts_per_video + 1):
                filename = self.generate_video(num_points, attempt)
                
                if filename:
                    successful_videos.append(filename)
                    success = True
                    break
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
    ENERGY_FACTOR = 1.1  # Can be modified as needed
    NUM_POINTS = 5
    NUM_VIDEOS = 100  # Number of successful videos to generate
    VIDEO_WIDTH = 1920   # Full HD width for crisp quality
    VIDEO_HEIGHT = 1080  # Full HD height for crisp quality
    VIDEO_FPS = 60
    MAX_ATTEMPTS_PER_VIDEO = 1000
    SIMULATION_SPEED_MULTIPLIER = 4.0  # How much faster to run simulation (4x = 17.5 min real time for 70s video)
    
    print("Point Collision Simulation - Video Generator")
    print("=" * 50)
    print("Two-phase generation process:")
    print("1. Test simulation viability at high speed (no recording)")
    print("2. Record video only for viable simulations")
    print("Success criteria:")
    print("- Must survive 60+ seconds with at least 2 points")
    print("- Speed cap removed at 60s for final 10s of chaos")
    print("- Simulations with <2 points before 60s are discarded")
    print(f"- Simulation speed: {SIMULATION_SPEED_MULTIPLIER}x (faster generation)")
    print("-" * 50)
    
    # Create video generator
    generator = VideoGenerator(
        energy_factor=ENERGY_FACTOR,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        fps=VIDEO_FPS,
        simulation_speed_multiplier=SIMULATION_SPEED_MULTIPLIER
    )
    
    # Generate videos
    successful_videos = generator.generate_videos(
        num_videos=NUM_VIDEOS,
        num_points=NUM_POINTS,
        max_attempts_per_video=MAX_ATTEMPTS_PER_VIDEO
    )
    
    if successful_videos:
        print(f"\n🎉 Successfully generated {len(successful_videos)} videos!")
    else:
        print(f"\n❌ No videos were successfully generated.")
    
    print("\nVideo generation complete.")

if __name__ == "__main__":
    main()
