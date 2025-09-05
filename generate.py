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
    def __init__(self, energy_factor=1.01, width=800, height=600, fps=60):
        self.energy_factor = energy_factor
        self.width = width
        self.height = height
        self.fps = fps
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
        # Temporarily override point radius to make points 75% bigger
        original_point_radius = config.POINT_RADIUS
        config.POINT_RADIUS = int(config.POINT_RADIUS * 1.75)
        
        # Create a modified simulation that doesn't show UI
        simulation = CircleSimulation(
            width=self.width,
            height=self.height,
            num_points=num_points,
            energy_factor=self.energy_factor
        )
        
        # Restore original point radius
        config.POINT_RADIUS = original_point_radius
        
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
    
    def run_simulation_for_video(self, simulation, video_writer):
        """
        Run a simulation and record it to video.
        Returns True if simulation lasted the full duration, False otherwise.
        """
        start_time = time.time()
        frame_count = 0
        speed_boosted = False
        
        print(f"Starting simulation with {len(simulation.points)} points, energy factor {self.energy_factor}")
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if we should stop (simulation ended early or reached final duration)
            if elapsed_time >= self.final_duration:
                print(f"Simulation completed full duration: {elapsed_time:.1f}s")
                return True
            
            # Check if all points are gone (simulation ended early)
            if len(simulation.points) == 0:
                print(f"Simulation ended early at {elapsed_time:.1f}s - all points eliminated")
                return False
            
            # Check if only one point remains before reaching target duration (boring scenario)
            if len(simulation.points) == 1 and elapsed_time < self.target_duration:
                print(f"Simulation ended early at {elapsed_time:.1f}s - only one point remaining (boring)")
                return False
            
            # Apply speed boost at target duration
            if elapsed_time >= self.target_duration and not speed_boosted:
                print(f"Applying speed boost at {elapsed_time:.1f}s")
                # Set maximum speed to infinity (remove speed limit)
                for point in simulation.points:
                    point.max_speed = float('inf')
                # Also update the config maximum speed
                original_max_speed = config.MAX_POINT_SPEED
                config.MAX_POINT_SPEED = float('inf')
                speed_boosted = True
            
            # Update simulation physics
            dt = self.clock.tick(self.fps) / 1000.0
            simulation.update_physics(dt)
            
            # Render to our surface
            simulation.render()
            
            # Convert surface to array and write to video
            frame_array = self.surface_to_array(self.screen)
            video_writer.write(frame_array)
            
            frame_count += 1
            
            # Print progress every 5 seconds
            if frame_count % (self.fps * 5) == 0:
                print(f"Progress: {elapsed_time:.1f}s, Points: {len(simulation.points)}, Lines: {sum(1 for line in simulation.lines.values() if line.active)}")
        
        # Restore original max speed if it was changed
        if speed_boosted:
            config.MAX_POINT_SPEED = original_max_speed
    
    def generate_video(self, num_points=5, attempt_number=1):
        """
        Generate a single video file.
        Returns the filename if successful, None if simulation ended early.
        """
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}_attempt{attempt_number}_ef{self.energy_factor:.3f}_p{num_points}.mp4"
        filepath = os.path.join(self.recordings_dir, filename)
        
        print(f"\nAttempt {attempt_number}: Starting video generation...")
        print(f"Target file: {filename}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {filepath}")
            return None
        
        try:
            # Create and run simulation
            simulation = self.create_simulation(num_points)
            success = self.run_simulation_for_video(simulation, video_writer)
            
            # Clean up
            video_writer.release()
            
            if success:
                print(f"✓ Successfully generated: {filename}")
                return filename
            else:
                print(f"✗ Simulation ended early, deleting: {filename}")
                # Delete the incomplete video file
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return None
                
        except Exception as e:
            print(f"Error during video generation: {e}")
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
    ENERGY_FACTOR = 1.05  # Can be modified as needed
    NUM_POINTS = 5
    NUM_VIDEOS = 100  # Number of successful videos to generate
    VIDEO_WIDTH = 800
    VIDEO_HEIGHT = 600
    VIDEO_FPS = 60
    MAX_ATTEMPTS_PER_VIDEO = 100
    
    print("Point Collision Simulation - Video Generator")
    print("=" * 50)
    print("Success criteria:")
    print("- Must survive 60+ seconds with at least 2 points")
    print("- Speed cap removed at 60s for final 10s of chaos")
    print("- Simulations with <2 points before 60s are discarded")
    print("-" * 50)
    
    # Create video generator
    generator = VideoGenerator(
        energy_factor=ENERGY_FACTOR,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        fps=VIDEO_FPS
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
