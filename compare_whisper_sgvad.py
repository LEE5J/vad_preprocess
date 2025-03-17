import os
import glob
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sgvad import SGVAD
import textgrid
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from create_video_with_progress import generate_video_with_progress
import multiprocessing
from tqdm import tqdm
from functools import partial

def get_wav_files(directory):
    """Get all WAV files from the target directory"""
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    return wav_files

def read_audio_file(file_path):
    """Read audio file and return waveform and sample rate"""
    waveform, sample_rate = librosa.load(file_path, sr=None)
    return waveform, sample_rate

def read_textgrid_file(file_path):
    """Read TextGrid file and extract speech intervals"""
    try:
        tg = textgrid.TextGrid.fromFile(file_path)
        speech_intervals = []
        
        # Find the tier that contains speech segments
        for tier in tg.tiers:
            if tier.name.lower() in ["speech", "words", "phones", "segments"]:
                for interval in tier.intervals:
                    if interval.mark.strip():  # If the label is not empty
                        speech_intervals.append((interval.minTime, interval.maxTime))
        
        return speech_intervals
    except Exception as e:
        print(f"Error reading TextGrid file {file_path}: {e}")
        return []

def process_vad_results(audio_path, sgvad_model, A_percentile=85, B_percentile=60):
    """Process audio with SGVAD and get speech segments based on percentile thresholds"""
    # Load audio
    audio = sgvad_model.load_audio(audio_path)
    
    # Get SGVAD probabilities
    probs = sgvad_model.predict(audio, smooth=21)
    
    # Calculate percentile thresholds
    B_threshold = np.percentile(probs, B_percentile)
    A_threshold = np.percentile(probs, A_percentile)
    
    # Adjust thresholds according to requirements
    B_threshold = max(min(B_threshold, 0.49), 0.3)  # B_threshold between 0.3 and 0.49
    A_threshold = max(A_threshold, 0.5)   # A_threshold (higher) should be >= 0.5
    
    # Find speech segments
    candidate_labels = (np.array(probs) > B_threshold).astype(int)
    speech_segments = []
    
    i = 0
    while i < len(candidate_labels):
        if candidate_labels[i] == 1:
            start_frame_idx = i
            while i < len(candidate_labels) and candidate_labels[i] == 1:
                i += 1
            end_frame_idx = i - 1
            
            # Check if any probability in this segment exceeds the A_threshold
            candidate_segment = probs[start_frame_idx:end_frame_idx+1]
            if np.any(np.array(candidate_segment) >= A_threshold):
                # Convert frame indices to time (assuming 10ms per frame)
                start_time = start_frame_idx * 0.01  # 10ms per frame
                end_time = (end_frame_idx + 1) * 0.01
                speech_segments.append((start_time, end_time))
        else:
            i += 1
    
    return probs, B_threshold, A_threshold, speech_segments

def plot_comparison(waveform, sample_rate, probs, B_threshold, A_threshold, A_percentile, B_percentile,
                   sgvad_segments, textgrid_segments, title, output_path=None):
    """Plot two-panel visualization comparing waveform with speech segments and VAD probabilities"""
    # Set fixed figure size in inches for 1920x1080 pixels at 100 dpi
    dpi = 100
    figsize = (1920/dpi, 1080/dpi)  # Convert pixels to inches
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, gridspec_kw={'height_ratios': [1, 1]})
    fig.set_size_inches(figsize)
    
    # Time axis for waveform
    audio_duration = len(waveform) / sample_rate
    audio_time = np.linspace(0, audio_duration, len(waveform))
    
    # Time axis for probabilities (assuming 10ms per frame)
    prob_duration = len(probs) * 0.01  # 10ms per frame
    prob_time = np.linspace(0, prob_duration, len(probs))
    
    # 1. Audio waveform with speech segments overlay
    ax1.plot(audio_time, waveform, color='black', alpha=0.9)
    
    # Plot TextGrid speech segments in blue (more transparent)
    for start, end in textgrid_segments:
        ax1.axvspan(start, end, alpha=0.2, color='blue')
    
    # Plot SGVAD speech segments in red (more transparent)
    for start, end in sgvad_segments:
        ax1.axvspan(start, end, alpha=0.15, color='red')
    
    # Add legend with custom patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.2, label='TextGrid Speech'),
        Patch(facecolor='red', alpha=0.15, label='SGVAD Speech')
    ]
    ax1.legend(handles=legend_elements)
    
    ax1.set_title(f"Audio Waveform with Speech Segments: {title}")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis limits to exact duration of audio with no margin
    ax1.set_xlim(0, audio_duration)
    
    # 2. SGVAD probabilities
    ax2.plot(prob_time, probs, color='green')
    ax2.axhline(y=B_threshold, color='orange', linestyle='--', 
               label=f'{B_percentile}th percentile (B_threshold): {B_threshold:.2f}')
    ax2.axhline(y=A_threshold, color='red', linestyle='--', 
               label=f'{A_percentile}th percentile (A_threshold): {A_threshold:.2f}')
    ax2.set_title("SGVAD Speech Probabilities")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Probability")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set x-axis limits to exact duration of audio with no margin
    ax2.set_xlim(0, audio_duration)
    
    plt.tight_layout()
    
    # Calculate pixel positions
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the display coordinates of the data points
    display_start = ax1.transData.transform((0, 0))[0]
    display_end = ax1.transData.transform((audio_duration, 0))[0]
    
    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()
    
    # Return display coordinates and audio duration
    return display_start, display_end, audio_duration

def process_single_file(wav_file, output_dir, video_output_dir, textgrid_dir, A_percentile, B_percentile, fps):
    """Process a single WAV file - to be run in a separate process"""
    try:
        # Initialize SGVAD model for this process
        sgvad_model = SGVAD.init_from_ckpt()
        
        # Read audio file
        waveform, sample_rate = read_audio_file(wav_file)
        
        # Get file name without extension
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        
        # Look for corresponding TextGrid file
        textgrid_path = None
        if textgrid_dir:
            textgrid_path = os.path.join(textgrid_dir, f"{base_name}.TextGrid")
            if not os.path.exists(textgrid_path):
                textgrid_path = os.path.join(textgrid_dir, f"{base_name}.textgrid")
                if not os.path.exists(textgrid_path):
                    textgrid_path = None
        
        # Read TextGrid file if available
        textgrid_segments = []
        if textgrid_path and os.path.exists(textgrid_path):
            textgrid_segments = read_textgrid_file(textgrid_path)
        
        # Process with SGVAD using configurable percentiles
        probs, B_threshold, A_threshold, sgvad_segments = process_vad_results(
            wav_file, sgvad_model, A_percentile, B_percentile)
        
        # Create visualization and get display coordinates
        plot_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        display_start, display_end, audio_duration = plot_comparison(
            waveform, 
            sample_rate, 
            probs, 
            B_threshold, 
            A_threshold,
            A_percentile,
            B_percentile,
            sgvad_segments, 
            textgrid_segments, 
            base_name, 
            plot_path
        )
        
        # Generate video with progress bar
        video_path = os.path.join(video_output_dir, f"{base_name}_video.mp4")
        success = generate_video_with_progress(
            plot_path,
            wav_file,
            int(display_start),
            int(display_end),
            video_path,
            fps=fps
        )
        
        return base_name, True
        
    except Exception as e:
        return os.path.basename(wav_file), False

def main():
    # Configuration
    input_dir = "target_audio" 
    output_dir = "result_compare_plt"
    video_output_dir = "result_videos"
    textgrid_dir = "result_tg"
    A_percentile = 85
    B_percentile = 60
    fps = 30
    num_processes = 8  # Fixed 8 processes as requested
    
    # Path to directories
    target_dir = os.path.join(os.path.dirname(__file__), input_dir)
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    video_output_dir = os.path.join(os.path.dirname(__file__), video_output_dir)
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Check if target directory exists
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} not found.")
        return
    
    # Get all WAV files
    wav_files = get_wav_files(target_dir)
    
    if not wav_files:
        print(f"No WAV files found in {target_dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Define process_function with fixed parameters
    process_func = partial(
        process_single_file,
        output_dir=output_dir,
        video_output_dir=video_output_dir,
        textgrid_dir=textgrid_dir,
        A_percentile=A_percentile,
        B_percentile=B_percentile,
        fps=fps
    )
    
    # Initialize multiprocessing pool with 8 processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process files in parallel with tqdm progress bar
        results = list(tqdm(
            pool.imap(process_func, wav_files),
            total=len(wav_files),
            desc="Processing files"
        ))
    
    # Report results
    success_count = sum(1 for _, success in results if success)
    print(f"Successfully processed {success_count} out of {len(wav_files)} files")
    
    # List any files that failed
    failed_files = [name for name, success in results if not success]
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for file in failed_files:
            print(f"  - {file}")

if __name__ == "__main__":
    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
