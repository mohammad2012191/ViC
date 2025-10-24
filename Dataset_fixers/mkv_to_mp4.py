import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_mkv_to_mp4(input_path, output_path):
    """
    Convert a single MKV file to MP4 using ffmpeg.
    
    Args:
        input_path: Path to input .mkv file
        output_path: Path to output .mp4 file
    """
    try:
        # Using ffmpeg with copy codec for fast conversion (no re-encoding)
        # If you need re-encoding, replace 'copy' with actual codecs like 'libx264'
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c', 'copy',  # Copy streams without re-encoding (fast)
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")
        return False

def convert_all_mkv_in_directory(directory):
    """
    Convert all .mkv files in a directory to .mp4 format.
    
    Args:
        directory: Path to directory containing .mkv files
    """
    directory = Path(directory)
    
    # Find all .mkv files
    mkv_files = list(directory.glob("*.mkv"))
    
    if not mkv_files:
        print(f"No .mkv files found in {directory}")
        return
    
    print(f"Found {len(mkv_files)} .mkv files to convert")
    
    successful = 0
    failed = 0
    
    for mkv_path in tqdm(mkv_files, desc="Converting videos"):
        # Create output path with .mp4 extension
        mp4_path = mkv_path.with_suffix('.mp4')
        
        # Skip if mp4 already exists
        if mp4_path.exists():
            print(f"Skipping {mkv_path.name} - .mp4 already exists")
            continue
        
        # Convert the file
        if convert_mkv_to_mp4(str(mkv_path), str(mp4_path)):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    

if __name__ == "__main__":
    # Set your video directory here
    VIDEO_DIR = "ActivityNet/val1"  # Change this to your directory
    
    # Check if directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Directory '{VIDEO_DIR}' does not exist")
    else:
        convert_all_mkv_in_directory(VIDEO_DIR)