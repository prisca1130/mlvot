import os
import sys
import numpy as np
from pathlib import Path
import time
import trackeval
import shutil
import configparser

def copy_sequence_info(src_dir, dst_dir):
    """
    Copy and verify seqinfo.ini to the evaluation directory structure
    """
    src_path = Path(src_dir) / 'seqinfo.ini'
    dst_path = Path(dst_dir) / 'ADL-Rundle-6' / 'seqinfo.ini'
    
    # Create destination directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read and write the file to ensure proper formatting
    config = configparser.ConfigParser()
    config.read(src_path)
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the config to the new location
    with open(dst_path, 'w') as f:
        config.write(f)
    
    return dst_path.exists()

def prepare_data_for_evaluation(tracking_results_path, output_dir, gt_dir):
    """
    Convert tracking results to MOTChallenge format and prepare directory structure.
    """
    # Create required directories using absolute paths
    output_dir = Path(output_dir).resolve()
    seqmap_dir = output_dir / 'seqmaps'
    tracker_dir = output_dir / 'trackers' / 'mot_challenge' / 'MyTracker' / 'data'
    seq_dir = output_dir / 'ADL-Rundle-6'
    
    # Create all necessary directories
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Create seqmap file
    with open(seqmap_dir / 'MOT17-train.txt', 'w') as f:
        f.write('name\n')
        f.write('ADL-Rundle-6\n')

    # Copy seqinfo.ini to the proper location
    if not copy_sequence_info(gt_dir, output_dir):
        print("Failed to copy seqinfo.ini")
        return False

    try:
        # Copy and format tracking results
        results = np.loadtxt(tracking_results_path, delimiter=',')
        formatted_results = []
        
        for row in results:
            frame_id, track_id = int(row[0]), int(row[1])
            x, y, w, h = row[2:6]
            formatted_results.append([frame_id, track_id, x, y, w, h, 1, -1, -1, -1])
        
        # Save formatted results
        result_file = tracker_dir / 'ADL-Rundle-6.txt'
        np.savetxt(
            result_file,
            formatted_results,
            delimiter=',',
            fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d'
        )
        
        # Also copy ground truth to the evaluation directory
        gt_src = Path(gt_dir) / 'gt' / 'gt.txt'
        gt_dst = seq_dir / 'gt' / 'gt.txt'
        gt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_src, gt_dst)
        
        return True
    except Exception as e:
        print(f"Error preparing evaluation data: {str(e)}")
        return False

def evaluate_tracking(gt_folder, results_path, eval_config=None):
    """
    Evaluate tracking results using TrackEval.
    """
    # Convert paths to absolute paths
    gt_base_path = Path(gt_folder).resolve().parent.parent
    results_path = Path(results_path).resolve()
    
    if eval_config is None:
        eval_config = {
            'DISPLAY_LESS_PROGRESS': False,
            'METRICS': ['HOTA', 'CLEAR', 'Identity'],
            'TRACKERS_TO_EVAL': ['MyTracker'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'INPUT_AS_ZIP': False,
            'DO_PREPROC': True,
            'OUTPUT_DETAILED': True,
        }

    # Prepare temporary evaluation directory
    temp_eval_dir = Path('temp_eval').resolve()
    if temp_eval_dir.exists():
        shutil.rmtree(temp_eval_dir)
    temp_eval_dir.mkdir()

    try:
        # Prepare data including copying sequence info
        if not prepare_data_for_evaluation(results_path, temp_eval_dir, gt_base_path):
            raise Exception("Failed to prepare evaluation data")

        # Set up evaluator
        evaluator = trackeval.Evaluator(eval_config)
        dataset_config = {
            'GT_FOLDER': str(temp_eval_dir),  # Changed to use temp directory
            'TRACKERS_FOLDER': str(temp_eval_dir / 'trackers'),
            'OUTPUT_FOLDER': str(temp_eval_dir / 'output'),
            'SEQMAP_FOLDER': str(temp_eval_dir / 'seqmaps'),
            'CLASSES_TO_EVAL': ['pedestrian'],
            'PRINT_CONFIG': True,
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            'SKIP_SPLIT_FOL': True
        }

        # Print debug information
        print("\nDebug Information:")
        print(f"GT Base Path: {temp_eval_dir}")
        print(f"Results Path: {results_path}")
        print(f"Temp Eval Dir: {temp_eval_dir}")
        print(f"Seqinfo.ini location: {temp_eval_dir / 'ADL-Rundle-6' / 'seqinfo.ini'}")

        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

        # Run evaluation
        start_time = time.time()
        results = evaluator.evaluate(dataset_list)
        processing_time = time.time() - start_time

        return results, processing_time

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None, 0
    finally:
        # Clean up
        if temp_eval_dir.exists():
            shutil.rmtree(temp_eval_dir)

def print_metrics(results, processing_time):
    """
    Print the required metrics: HOTA, IDF1, and ID_Switch
    """
    if results is None:
        print("No results to display")
        return

    try:
        metrics = results['MotChallenge2DBox']['ADL-Rundle-6']['MyTracker']
        print("\nTracking Evaluation Results:")
        print(f"HOTA Score: {metrics['HOTA']['HOTA']:.3f}")
        print(f"IDF1 Score: {metrics['Identity']['IDF1']:.3f}")
        print(f"ID Switches: {metrics['Identity']['IDSW']}")
        print(f"Processing Speed: {processing_time:.2f} seconds")
        
        print("\nDetailed Metrics:")
        print(f"Detection Precision: {metrics['CLEAR']['DetPr']:.3f}")
        print(f"Detection Recall: {metrics['CLEAR']['DetRe']:.3f}")
        print(f"MOTA: {metrics['CLEAR']['MOTA']:.3f}")
    except KeyError as e:
        print(f"Error accessing metrics: {str(e)}")
        print("Available metrics structure:", results.keys())

if __name__ == "__main__":
    # Use absolute paths
    current_dir = Path.cwd()
    gt_path = current_dir / "ADL-Rundle-6/gt/gt.txt"
    results_path = current_dir / "results_TP3.txt"
    
    print("\nStarting evaluation with paths:")
    print(f"Ground truth path: {gt_path}")
    print(f"Results path: {results_path}")
    
    # Run evaluation
    results, processing_time = evaluate_tracking(gt_path, results_path)
    
    # Print metrics
    print_metrics(results, processing_time)