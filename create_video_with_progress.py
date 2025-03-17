import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import subprocess
import glob
import tempfile
import shutil
from typing import Tuple, List, Optional
import json


def create_progress_frame(base_image: Image.Image, progress_x: int, 
                         line_color: Tuple[int, int, int] = (255, 0, 0), 
                         line_width: int = 2) -> Image.Image:
    """
    Create a frame with a vertical progress line at the specified x position
    
    Args:
        base_image: Base image to draw on
        progress_x: X-coordinate of progress line
        line_color: RGB color tuple for the line
        line_width: Width of the progress line
    
    Returns:
        Frame with progress line as PIL Image
    """
    # Create a copy of the base image to avoid modifying the original
    frame = base_image.copy()
    draw = ImageDraw.Draw(frame)
    
    # Draw vertical line at progress_x
    draw.line([(progress_x, 0), (progress_x, base_image.height)], 
             fill=line_color, width=line_width)
    
    return frame


def generate_video_with_progress(image_path: str, audio_path: str, 
                               display_start: int, display_end: int, 
                               output_path: str, fps: int = 30, 
                               line_color: Tuple[int, int, int] = (255, 0, 0),
                               line_width: int = 2) -> bool:
    """
    Generate a video with a moving progress line synchronized with audio
    
    Args:
        image_path: Path to the visualization image
        audio_path: Path to the audio file
        display_start: Starting x-coordinate for progress line (pixels)
        display_end: Ending x-coordinate for progress line (pixels)
        output_path: Path to save the output video
        fps: Frames per second for the output video
        line_color: RGB color tuple for the progress line
        line_width: Width of the progress line
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Load base image
        base_image = Image.open(image_path)
        
        # Get audio duration using FFprobe
        duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            audio_path
        ]
        
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration_data = json.loads(result.stdout)
        audio_duration = float(duration_data['format']['duration'])
        
        # Calculate total number of frames
        total_frames = int(audio_duration * fps)
        
        # Calculate the progress coordinates for each frame
        progress_range = display_end - display_start
        
        # Create temporary directory to store frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate frames with progress line
            for frame_idx in range(total_frames):
                # Calculate current time position
                current_time = frame_idx / fps
                
                # Calculate current x position for the progress line
                progress_ratio = current_time / audio_duration
                progress_x = int(display_start + progress_ratio * progress_range)
                
                # Create frame with progress line
                frame = create_progress_frame(base_image, progress_x, line_color, line_width)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                frame.save(frame_path)
            
            # Create video using FFmpeg with NVIDIA hardware acceleration if available
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-i', audio_path,
                '-c:v', 'h264_nvenc',  # NVIDIA hardware encoding
                '-preset', 'slow',
                '-b:v', '5M',  # Video bitrate
                '-c:a', 'aac',  # Audio codec
                '-b:a', '192k',  # Audio bitrate
                '-shortest',
                output_path
            ]
            
            # Check for NVENC availability before running
            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if 'h264_nvenc' not in check_result.stdout:
                # Fallback to CPU encoding if NVENC is not available
                ffmpeg_cmd[ffmpeg_cmd.index('-c:v') + 1] = 'libx264'
            
            # Run FFmpeg command quietly
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
            
    except Exception as e:
        # Silent fail for multiprocessing
        return False


def load_coordinates_from_log(log_file: str) -> dict:
    """
    Load display coordinates from a log file
    
    Args:
        log_file: Path to log file containing coordinates
    
    Returns:
        Dictionary with filenames as keys and coordinate data as values
    """
    coordinates = {}
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith("File:"):
                    parts = line.strip().split(", ")
                    if len(parts) >= 4:
                        filename = parts[0].replace("File: ", "")
                        start = int(parts[1].replace("Start: ", ""))
                        end = int(parts[2].replace("End: ", ""))
                        duration = float(parts[3].replace("Duration: ", "").replace("s", ""))
                        coordinates[filename] = {
                            "start": start,
                            "end": end,
                            "duration": duration
                        }
    except Exception as e:
        print(f"Error loading coordinates from log file: {e}")
    
    return coordinates


def batch_process_files(base_dir: str, log_file: Optional[str] = None, fps: int = 30) -> None:
    """
    Process all visualization images and audio files in a directory
    
    Args:
        base_dir: Base directory containing visualization images and audio files
        log_file: Path to log file containing coordinates (if None, use default values)
        fps: Frames per second for output videos
    """
    # Load coordinates if log file is provided
    coordinates = {}
    if log_file and os.path.exists(log_file):
        coordinates = load_coordinates_from_log(log_file)
    
    # Find all visualization images
    viz_files = glob.glob(os.path.join(base_dir, "result_compare_plt", "*_comparison.png"))
    
    # Create output directory
    output_dir = os.path.join(base_dir, "result_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    for viz_file in viz_files:
        try:
            # Extract base filename
            base_name = os.path.basename(viz_file).replace("_comparison.png", "")
            
            # Look for corresponding audio file
            audio_file = os.path.join(base_dir, "target_audio", f"{base_name}.wav")
            if not os.path.exists(audio_file):
                print(f"Audio file not found for {base_name}")
                continue
            
            # Get display coordinates
            if base_name in coordinates:
                display_start = coordinates[base_name]["start"]
                display_end = coordinates[base_name]["end"]
            else:
                # Use default values if not in log file
                print(f"Coordinates not found for {base_name}, using default values")
                display_start = 100
                display_end = 1800
            
            # Output video path
            output_video = os.path.join(output_dir, f"{base_name}_video.mp4")
            
            # Generate video
            generate_video_with_progress(
                viz_file, audio_file, 
                display_start, display_end, 
                output_video, fps
            )
            
            print(f"Processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {viz_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos with progress line using NVIDIA encoding")
    parser.add_argument("--image", help="Path to visualization image")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--start", type=int, help="Start x-coordinate (pixels)")
    parser.add_argument("--end", type=int, help="End x-coordinate (pixels)")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--batch", action="store_true", help="Process all files in directory")
    parser.add_argument("--base-dir", default=".", help="Base directory for batch processing")
    parser.add_argument("--log-file", help="Path to log file containing coordinates")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process_files(args.base_dir, args.log_file, args.fps)
    else:
        if not all([args.image, args.audio, args.start, args.end, args.output]):
            print("Missing required arguments for single file processing")
            parser.print_help()
            return
        
        generate_video_with_progress(
            args.image, args.audio, 
            args.start, args.end, 
            args.output, args.fps
        )


if __name__ == "__main__":
    main()
