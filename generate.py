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
        
        # Calculate time step for accelerated simulation
        base_dt = 1.0 / self.fps  # Normal time step
        accelerated_dt = base_dt * self.simulation_speed_multiplier
        
        # Calculate total frames needed for target video duration
        target_frames = int(self.target_duration * self.fps)  # 60s * 60fps = 3600 frames
        final_frames = int(self.final_duration * self.fps)    # 70s * 60fps = 4200 frames
        
        print(f"Starting simulation with {len(simulation.points)} points, energy factor {self.energy_factor}")
        print(f"Simulation speed: {self.simulation_speed_multiplier}x (generating {self.final_duration}s video in {self.final_duration/self.simulation_speed_multiplier:.1f}s real time)")
        
        while frame_count < final_frames:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_time
            
            # Calculate current video time based on frame count
            video_time = frame_count / self.fps
            
            # Check if all points are gone (simulation ended early)
            if len(simulation.points) == 0:
                print(f"Simulation ended early at {video_time:.1f}s video time - all points eliminated (real time: {elapsed_real_time:.1f}s)")
                return False
            
            # Check if only one point remains before reaching target duration (boring scenario)
            if len(simulation.points) == 1 and video_time < self.target_duration:
                print(f"Simulation ended early at {video_time:.1f}s video time - only one point remaining (boring) (real time: {elapsed_real_time:.1f}s)")
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
            
            # Update simulation physics with accelerated time step
            simulation.update_physics(accelerated_dt)
            
            # Render to our surface
            simulation.render()
            
            # Convert surface to array and write to video
            frame_array = self.surface_to_array(self.screen)
            video_writer.write(frame_array)
            
            frame_count += 1
            
            # Print progress every 5 seconds of video time
            if frame_count % (self.fps * 5) == 0:
                real_time_ratio = video_time / elapsed_real_time if elapsed_real_time > 0 else 0
                print(f"Progress: {video_time:.1f}s video time ({elapsed_real_time:.1f}s real, {real_time_ratio:.1f}x speed), Points: {len(simulation.points)}, Lines: {sum(1 for line in simulation.lines.values() if line.active)}")
        
        # Restore original max speed if it was changed
        if speed_boosted:
            config.MAX_POINT_SPEED = original_max_speed
        
        print(f"Simulation completed full duration: {self.final_duration:.1f}s video time (real time: {elapsed_real_time:.1f}s)")
        return True
    
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
    ENERGY_FACTOR = 1.1  # Can be modified as needed
    NUM_POINTS = 5
    NUM_VIDEOS = 100  # Number of successful videos to generate
    VIDEO_WIDTH = 800
    VIDEO_HEIGHT = 600
    VIDEO_FPS = 60
    MAX_ATTEMPTS_PER_VIDEO = 100
    SIMULATION_SPEED_MULTIPLIER = 4.0  # How much faster to run simulation (4x = 17.5 min real time for 70s video)
    
    print("Point Collision Simulation - Video Generator")
    print("=" * 50)
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
