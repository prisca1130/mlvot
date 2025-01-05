import os
import sys
import numpy as np
from pathlib import Path
import time
import trackeval
import shutil

def prepare_data_for_evaluation(tracking_results_path, output_dir):
    """
    Convert tracking results to MOTChallenge format and prepare directory structure
    """
    # Create required directories
    output_dir = Path(output_dir)
    seqmap_dir = output_dir / 'seqmaps'
    tracker_dir = output_dir / 'trackers' / 'mot_challenge' / 'MyTracker' / 'data'
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    
    # Create seqmap file
    with open(seqmap_dir / 'MOT17-train.txt', 'w') as f:
        f.write('name\n')
        f.write('ADL-Rundle-6\n')
    
    # Copy and format tracking results
    results = np.loadtxt(tracking_results_path, delimiter=',')
    formatted_results = []
    for row in results:
        frame_id, track_id = int(row[0]), int(row[1])
        x, y, w, h = row[2:6]
        formatted_results.append([frame_id, track_id, x, y, w, h, 1, -1, -1, -1])
    
    # Save formatted results
    np.savetxt(tracker_dir / 'ADL-Rundle-6.txt', formatted_results, 
               delimiter=',', fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d')

def evaluate_tracking(gt_path, results_path, eval_config=None):
    """
    Evaluate tracking results using TrackEval
    """
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
    temp_eval_dir = Path('temp_eval')
    if temp_eval_dir.exists():
        shutil.rmtree(temp_eval_dir)
    temp_eval_dir.mkdir()
    
    # Prepare data
    prepare_data_for_evaluation(results_path, temp_eval_dir)
    
    # Set up evaluator
    evaluator = trackeval.Evaluator(eval_config)
    dataset_config = {
        'GT_FOLDER': gt_path,
        'TRACKERS_FOLDER': str(temp_eval_dir / 'trackers'),
        'OUTPUT_FOLDER': str(temp_eval_dir / 'output'),
        'SEQMAP_FOLDER': str(temp_eval_dir / 'seqmaps'),
        'CLASSES_TO_EVAL': ['pedestrian'],
    }
    print(f"Expected path to seqinfo.ini: {Path('ADL-Rundle-6/seqinfo.ini').resolve()}")
    print(f"GT_LOC_FORMAT: {dataset_config['GT_FOLDER']}")
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate(dataset_list)
    processing_time = time.time() - start_time
    
    # Clean up
    shutil.rmtree(temp_eval_dir)
    
    return results, processing_time

if __name__ == "__main__":
    # Example usage
    gt_path = "ADL-Rundle-6/gt/gt.txt"
    results_path = "results_TP3.txt"
    
    # Run evaluation
    results, processing_time = evaluate_tracking(gt_path, results_path)
    
    # Print metrics
    metrics = results['MotChallenge2DBox']['ADL-Rundle-6']['MyTracker']
    print("\nTracking Evaluation Results:")
    print(f"HOTA: {metrics['HOTA']['HOTA']:.2f}")
    print(f"IDF1: {metrics['Identity']['IDF1']:.2f}")
    print(f"ID Switches: {metrics['Identity']['IDSW']}")
    print(f"\nProcessing Time: {processing_time:.2f} seconds")