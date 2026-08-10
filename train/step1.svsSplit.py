import os
import openslide
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def make_parser():
    parser = argparse.ArgumentParser("OpenSlide SVS Patch Extractor - Center Patch Only")
    
    parser.add_argument("-i", "--input_file", 
                        type=str, 
                        default=None,
                        help="Input SVS file path")
    
    parser.add_argument("-s", "--size", default=1024, type=int, help="Patch size")
    parser.add_argument("--overlap", default=0.05, type=float, help="Overlap rate between patches")
    parser.add_argument("-o", "--output_dir", 
                        type=str, 
                        default=None,
                        help="Output directory for patches")
    parser.add_argument("--threads", default=1, type=int, help="Number of threads for parallel processing")
    parser.add_argument("--grid_size", default=3, type=int, help="Grid size (e.g., 3 for 3x3 grid)")
    return parser

def extract_patch(slide, x, y, patch_size, row, col, output_folder):
    """Extract a single patch from the slide"""
    start_time = time.time()
    
    # Read the region from the slide
    region = slide.read_region((x, y), level=0, size=(patch_size, patch_size))
    
    # Convert to RGB numpy array (remove alpha channel)
    region_np = np.array(region)[:, :, :3]
    
    # Convert from RGB to BGR for OpenCV
    region_bgr = cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)
    
    # Save the patch
    output_filename = f'tile_{row}_{col}.png'
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, region_bgr)
    
    end_time = time.time()
    return end_time - start_time

def process_batch(slide, batch_coords, output_folder, patch_size):
    """Process a batch of patches"""
    for x, y, row, col in batch_coords:
        extract_patch(slide, x, y, patch_size, row, col, output_folder)

def extract_center_patches_parallel(slide_path, output_folder, patch_size=1024, overlap_rate=0.05, num_threads=8, grid_size=3):
    """Extract only center patches from each grid"""
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the slide
    print(f"Opening slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)
    
    # Get dimensions of level 0 (highest resolution)
    width, height = slide.dimensions
    print(f"Slide dimensions: {width} x {height}")
    
    # Calculate effective step size with overlap
    overlap_pixels = int(patch_size * overlap_rate)
    step_size = patch_size - overlap_pixels
    
    # Calculate number of patches in each dimension
    num_cols = (width - overlap_pixels) // step_size
    num_rows = (height - overlap_pixels) // step_size
    
    # Calculate number of grids
    grid_rows = (num_rows + grid_size - 1) // grid_size
    grid_cols = (num_cols + grid_size - 1) // grid_size
    
    print(f"Original grid would be {num_rows} x {num_cols} = {num_rows * num_cols} patches")
    print(f"Will extract only center patches: approximately {grid_rows} x {grid_cols} = {grid_rows * grid_cols} patches")
    
    # Generate coordinates for center patches only
    coords = []
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            # Calculate the center patch position in this grid
            center_row = grid_row * grid_size + grid_size // 2
            center_col = grid_col * grid_size + grid_size // 2
            
            # Skip if the center patch is outside the image bounds
            if center_row >= num_rows or center_col >= num_cols:
                continue
                
            # Calculate the actual coordinates
            x = center_col * step_size
            y = center_row * step_size
            coords.append((x, y, center_row, center_col))
    
    # Process patches in parallel
    total_start_time = time.time()
    
    # Split coordinates into batches for each thread
    batch_size = len(coords) // num_threads + 1
    batches = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(process_batch, slide, batch, output_folder, patch_size))
        
        # Show progress
        with tqdm(total=len(coords), desc="Extracting center patches") as pbar:
            for i, future in enumerate(futures):
                result = future.result()
                if i < len(batches) - 1:
                    pbar.update(len(batches[i]))
                else:
                    pbar.update(len(coords) - pbar.n)
    
    total_end_time = time.time()
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Extracted {len(coords)} center patches to {output_folder}")

def process_svs_files(input_path, output_dir, patch_size, overlap_rate, num_threads, grid_size):
    """Process single SVS file or directory of SVS files"""
    if os.path.isdir(input_path):
        # Process all SVS files in directory
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".svs"):
                    svs_file_path = os.path.join(root, file)
                    sample_name = os.path.splitext(file)[0]
                    sample_output_dir = os.path.join(output_dir, sample_name)
                    extract_center_patches_parallel(svs_file_path, sample_output_dir, patch_size, overlap_rate, num_threads, grid_size)
    else:
        # Process single SVS file
        sample_name = os.path.splitext(os.path.basename(input_path))[0]
        sample_output_dir = os.path.join(output_dir, sample_name)
        extract_center_patches_parallel(input_path, sample_output_dir, patch_size, overlap_rate, num_threads, grid_size)

if __name__ == "__main__":
    args = make_parser().parse_args()
    
    if args.input_file is None:
        print("Please specify input SVS file or directory with -i")
        exit(1)
    
    if args.output_dir is None:
        print("Please specify output directory with -o")
        exit(1)
    
    print(f"Processing with patch size: {args.size}, overlap: {args.overlap}, threads: {args.threads}, grid size: {args.grid_size}x{args.grid_size}")
    process_svs_files(args.input_file, args.output_dir, args.size, args.overlap, args.threads, args.grid_size)