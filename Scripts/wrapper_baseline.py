#!/usr/bin/env python
"""
Wrapper script for ensemble retrieval methods
Supports CombSum, CombMNZ, and RRF for both T2V and V2T retrieval
"""

import argparse
import numpy as np
import pandas as pd
import sys
from baseline import (
    combsum_fusion,
    combmnz_fusion,
    rrf_fusion,
    evaluate_retrieval,
    load_similarity_matrices,
    print_results
)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Methods for Text-to-Video and Video-to-Text Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CombSum for T2V retrieval
  python wrapper_baseline.py --method combsum --mode t2v \\
    --sim_paths model1.npy model2.npy model3.npy \\
    --csv_path data.csv --video_dir /path/to/videos \\
    --output_path combsum_t2v.npy --evaluate

  # CombMNZ for V2T retrieval with custom weights
  python wrapper_baseline.py --method combmnz --mode v2t \\
    --sim_paths model1.npy model2.npy \\
    --csv_path data.csv --video_dir /path/to/videos \\
    --weights 0.6 0.4 --evaluate

  # RRF with custom k value
  python wrapper_baseline.py --method rrf --mode t2v \\
    --sim_paths model1.npy model2.npy model3.npy \\
    --csv_path data.csv --video_dir /path/to/videos \\
    --k 100 --evaluate
        """
    )
    
    # Required arguments
    parser.add_argument("--method", type=str, required=True,
                        choices=["combsum", "combmnz", "rrf"],
                        help="Ensemble method to use")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["t2v", "v2t"],
                        help="Retrieval mode: t2v (text-to-video) or v2t (video-to-text)")
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths (.npy files)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    
    # Optional arguments
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save fused similarity matrix (default: auto-generated)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate retrieval performance on the fused scores")
    
    # CombSum/CombMNZ arguments
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore", "none"],
                        help="Score normalization method (default: minmax) [CombSum/CombMNZ only]")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional weights for each matrix (must sum to 1.0) [CombSum/CombMNZ only]")
    
    # CombMNZ-specific arguments
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Threshold for non-zero detection (default: 1e-6) [CombMNZ only]")
    
    # RRF-specific arguments
    parser.add_argument("--k", type=int, default=60,
                        help="RRF constant k (default: 60) [RRF only]")
    
    # Other options
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = f"{args.method}_fused_{args.mode}.npy"
    
    # Validate weights
    if args.weights is not None:
        if len(args.weights) != len(args.sim_paths):
            parser.error(f"Number of weights ({len(args.weights)}) must match number of matrices ({len(args.sim_paths)})")
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("=" * 80)
        method_names = {
            'combsum': 'CombSum (Score Fusion)',
            'combmnz': 'CombMNZ (Combined Multiple Non-Zero)',
            'rrf': 'RRF (Reciprocal Rank Fusion)'
        }
        mode_names = {
            't2v': 'Text-to-Video',
            'v2t': 'Video-to-Text'
        }
        print(f"{method_names[args.method]} for {mode_names[args.mode]} Retrieval")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Method: {args.method}")
        print(f"  Mode: {args.mode}")
        print(f"  Similarity matrices: {len(args.sim_paths)}")
        for i, path in enumerate(args.sim_paths):
            weight_str = f" (weight: {args.weights[i]:.3f})" if args.weights else ""
            print(f"    {i+1}. {path}{weight_str}")
        print(f"  CSV path: {args.csv_path}")
        print(f"  Video directory: {args.video_dir}")
        
        if args.method in ['combsum', 'combmnz']:
            print(f"  Normalization: {args.normalization}")
        if args.method == 'combmnz':
            print(f"  Non-zero threshold: {args.threshold}")
        if args.method == 'rrf':
            print(f"  RRF constant k: {args.k}")
        
        print(f"  Output path: {args.output_path}")
        print()
    
    # Load CSV
    if verbose:
        print("Loading CSV...")
    df = pd.read_csv(args.csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if verbose:
        if args.mode == 't2v':
            print(f"Loaded {len(df)} queries")
        else:
            print(f"Loaded {len(df)} video-text pairs")
    
    # Load similarity matrices
    sim_matrices = load_similarity_matrices(args.sim_paths, df, mode=args.mode, verbose=verbose)
    
    # Perform fusion
    if verbose:
        print("\n" + "=" * 80)
    
    if args.method == 'combsum':
        fused_scores = combsum_fusion(
            sim_matrices,
            normalization=args.normalization,
            weights=args.weights,
            verbose=verbose
        )
    
    elif args.method == 'combmnz':
        fused_scores = combmnz_fusion(
            sim_matrices,
            normalization=args.normalization,
            weights=args.weights,
            threshold=args.threshold,
            verbose=verbose
        )
    
    elif args.method == 'rrf':
        fused_scores = rrf_fusion(
            sim_matrices,
            k=args.k,
            verbose=verbose
        )
    
    else:
        print(f"Error: Unknown method '{args.method}'")
        sys.exit(1)
    
    # Print statistics
    if verbose:
        print(f"\nFused scores statistics:")
        print(f"  Score range: [{np.min(fused_scores):.4f}, {np.max(fused_scores):.4f}]")
        print(f"  Score mean: {np.mean(fused_scores):.4f}, std: {np.std(fused_scores):.4f}")
    
    # Save fused scores
    if verbose:
        print(f"\nSaving fused similarity matrix to {args.output_path}...")
    np.save(args.output_path, fused_scores)
    if verbose:
        print("Saved!")
    
    # Evaluate if requested
    if args.evaluate:
        if verbose:
            print("\n" + "=" * 80)
        
        results = evaluate_retrieval(
            fused_scores,
            df,
            video_dir=args.video_dir,
            mode=args.mode,
            verbose=verbose
        )
        
        print_results(results)
    
    if verbose:
        print("\nDone!")


if __name__ == "__main__":
    main()
