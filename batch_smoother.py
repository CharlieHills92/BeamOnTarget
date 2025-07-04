# batch_smoother.py
"""
Batch processing tool to apply smoothing to all .vtp/.vtm simulation results
found in a directory. Smoothed files are saved to a 'SMOOTHED' subfolder
within the input directory.
"""
import pyvista as pv
import os
import glob
from tqdm import tqdm
import argparse

# Import the core smoothing function from our library file
from smooth_results import apply_smoothing

# --- CONFIGURATION PANEL ---
# This now acts as the DEFAULT input directory if none is provided via command line.
DEFAULT_INPUT_DIRECTORY = "OUTPUT"

# The prefix to add to the smoothed output files.
OUTPUT_PREFIX = "smoothed_"

# The number of Laplacian smoothing iterations to apply to every file.
SMOOTHING_ITERATIONS = 20

# --- END OF CONFIGURATION ---

def batch_process_directory(input_dir):
    """
    Finds all result files in the input directory, applies smoothing,
    and saves them to a 'SMOOTHED' subfolder within that directory.
    """
    if not os.path.isdir(input_dir):
        print(f"FATAL ERROR: Input directory '{input_dir}' not found.")
        return
        
    # --- NEW: Dynamically create the output directory path ---
    output_dir = os.path.join(input_dir, "SMOOTHED")
    os.makedirs(output_dir, exist_ok=True)
    
    search_path_vtp = os.path.join(input_dir, '*.vtp')
    search_path_vtm = os.path.join(input_dir, '*.vtm')
    # Important: Exclude files in the SMOOTHED subfolder from being processed again
    files_to_process = [f for f in glob.glob(search_path_vtp) + glob.glob(search_path_vtm) if "SMOOTHED" not in os.path.dirname(f)]
    
    if not files_to_process:
        print(f"No .vtp or .vtm files found in the root of '{input_dir}'. Nothing to do.")
        return

    print(f"--- Starting Batch Smoothing Process ---")
    print(f"Input directory:  '{input_dir}'")
    print(f"Output directory: '{output_dir}'")
    print(f"Found {len(files_to_process)} files to process.")
    
    for input_path in tqdm(files_to_process, desc="Smoothing Files"):
        try:
            dataset = pv.read(input_path)
            dataset_copy = dataset.copy(deep=True)
            
            if isinstance(dataset_copy, pv.MultiBlock):
                processed_dataset = pv.MultiBlock()
                for i in range(dataset_copy.n_blocks):
                    smoothed_block = apply_smoothing(dataset_copy[i], n_iter=SMOOTHING_ITERATIONS)
                    processed_dataset.append(smoothed_block)
                final_dataset = processed_dataset
            else:
                final_dataset = apply_smoothing(dataset_copy, n_iter=SMOOTHING_ITERATIONS)

            original_filename = os.path.basename(input_path)
            output_filename = f"{OUTPUT_PREFIX}{original_filename}"
            full_output_path = os.path.join(output_dir, output_filename)
            
            final_dataset.save(full_output_path, binary=True)
            
        except Exception as e:
            print(f"\n  - An error occurred while processing '{os.path.basename(input_path)}': {e}")
            
    print("\n--- Batch smoothing process finished. ---")

if __name__ == "__main__":
    # The argument parser is now simpler
    parser = argparse.ArgumentParser(description="Batch smooth VTP/VTM simulation results.")
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"Directory containing the input files. Defaults to '{DEFAULT_INPUT_DIRECTORY}'."
    )
    args = parser.parse_args()
    
    # Call the main function with only the input directory
    batch_process_directory(args.input_dir)