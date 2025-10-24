import os
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# Configuration
CSV_PATH = "descs_ret_test_didemo.csv"
INPUT_DIR = "didemo_new/videos"
OUTPUT_DIR = "corrected_didemo"

def re_encode_video(args):
    """
    Re-encode a single video using ffmpeg to fix corruption issues.
    
    Args:
        args: tuple of (input_path, output_path, video_id)
    
    Returns:
        tuple: (video_id, success, error_message)
    """
    input_path, output_path, video_id = args
    
    try:
        # FFmpeg command to re-encode the video
        # -y: overwrite output file
        # -i: input file
        # -c:v libx264: use H.264 codec (most compatible)
        # -preset medium: balance between speed and compression
        # -crf 23: quality (lower = better, 23 is default)
        # -c:a aac: use AAC audio codec
        # -b:a 128k: audio bitrate
        # -movflags +faststart: optimize for streaming/playback
        # -pix_fmt yuv420p: pixel format for maximum compatibility
        
        # Fix for odd dimensions: scale to even dimensions
        # This uses floor division to round down to nearest even number
        # trunc(iw/2)*2 = round down width to nearest even number
        # trunc(ih/2)*2 = round down height to nearest even number
        # This crops 1 pixel max if dimensions are odd (minimal impact)
        vf_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', input_path,  # Input file
            '-vf', vf_filter,  # Video filter to fix odd dimensions
            '-c:v', 'libx264',  # Video codec
            '-preset', 'medium',  # Encoding speed/quality tradeoff
            '-crf', '23',  # Quality (lower = better)
            '-c:a', 'aac',  # Audio codec
            '-b:a', '128k',  # Audio bitrate
            '-movflags', '+faststart',  # Optimize for playback
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-loglevel', 'error',  # Only show errors
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout per video
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            return (video_id, True, None)
        else:
            error_msg = result.stderr.decode('utf-8') if result.stderr else "Unknown error"
            return (video_id, False, error_msg)
            
    except subprocess.TimeoutExpired:
        return (video_id, False, "Timeout (>5 minutes)")
    except Exception as e:
        return (video_id, False, str(e))

def main():
    print("=" * 70)
    print("DiDeMo Video Re-encoding Script")
    print("=" * 70)
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✓ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: FFmpeg not found. Please install ffmpeg first:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Conda: conda install -c conda-forge ffmpeg")
        return
    
    # Load CSV
    print(f"\n📄 Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} entries")
    
    # Get unique video IDs
    if 'video_id' not in df.columns:
        print("✗ ERROR: 'video_id' column not found in CSV")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    unique_video_ids = df['video_id'].unique()
    print(f"✓ Found {len(unique_video_ids)} unique videos")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Created output directory: {OUTPUT_DIR}")
    
    # Prepare tasks
    tasks = []
    missing_videos = []
    
    for video_id in unique_video_ids:
        input_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")
        
        if not os.path.exists(input_path):
            missing_videos.append(video_id)
            continue
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
            
        tasks.append((input_path, output_path, video_id))
    
    if missing_videos:
        print(f"\n⚠ Warning: {len(missing_videos)} videos not found in {INPUT_DIR}")
        print(f"First few missing: {missing_videos[:5]}")
    
    already_done = len(unique_video_ids) - len(tasks) - len(missing_videos)
    if already_done > 0:
        print(f"✓ {already_done} videos already re-encoded, skipping...")
    
    if len(tasks) == 0:
        print("\n✓ All videos already processed!")
        return
    
    print(f"\n🎬 Re-encoding {len(tasks)} videos...")
    print("=" * 70)
    
    # Use multiprocessing for faster processing
    num_workers = min(mp.cpu_count(), 4)  # Use up to 4 workers
    print(f"Using {num_workers} parallel workers\n")
    
    # Process videos
    success_count = 0
    failed_videos = []
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(re_encode_video, tasks),
            total=len(tasks),
            desc="Processing",
            unit="video"
        ))
    
    # Collect results
    for video_id, success, error_msg in results:
        if success:
            success_count += 1
        else:
            failed_videos.append((video_id, error_msg))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully re-encoded: {success_count}/{len(tasks)} videos")
    
    if failed_videos:
        print(f"✗ Failed: {len(failed_videos)} videos")
        print("\nFailed videos:")
        for video_id, error_msg in failed_videos[:10]:  # Show first 10
            print(f"  - {video_id}: {error_msg[:100]}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Save failed videos list if any
    if failed_videos:
        failed_df = pd.DataFrame(failed_videos, columns=['video_id', 'error'])
        failed_df.to_csv('failed_videos.csv', index=False)
        print(f"✓ Failed videos list saved to: failed_videos.csv")

if __name__ == "__main__":
    main()