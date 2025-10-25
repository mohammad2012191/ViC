#!/usr/bin/env python
"""
CombMNZ (Combined Multiple Non-Zero) for VATEX Text-to-Video Retrieval
No GPU required - pure numpy implementation
Evaluates ensemble results directly without VLM reranking
"""

import os
import numpy as np
import pandas as pd
import argparse


def normalize_scores(scores, method='minmax'):
    """Normalize scores for a single query."""
    if method == 'none':
        return scores
    
    elif method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-8:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_score) / (max_score - min_score)
    
    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score < 1e-8:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - mean_score) / std_score
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def combmnz_fusion(sim_matrices, normalization='minmax', weights=None, threshold=1e-6):
    """
    Perform CombMNZ fusion on multiple similarity matrices.
    
    Args:
        sim_matrices: List of numpy arrays
        normalization: 'minmax', 'zscore', or 'none'
        weights: Optional list of weights
        threshold: Threshold for non-zero detection
    
    Returns:
        fused_scores: numpy array with CombMNZ scores
    """
    shape = sim_matrices[0].shape
    
    # Validate all matrices have same shape
    for i, mat in enumerate(sim_matrices):
        if mat.shape != shape:
            raise ValueError(f"Matrix {i} has shape {mat.shape}, expected {shape}")
    
    # Set up weights
    if weights is None:
        weights = np.ones(len(sim_matrices)) / len(sim_matrices)
    else:
        weights = np.array(weights, dtype=np.float32)
        if len(weights) != len(sim_matrices):
            raise ValueError(f"Number of weights must match number of matrices")
        if not np.isclose(weights.sum(), 1.0):
            print(f"Warning: weights sum to {weights.sum()}, normalizing to 1.0")
            weights = weights / weights.sum()
    
    # Initialize fused scores and non-zero counts
    fused_scores = np.zeros(shape, dtype=np.float32)
    nonzero_counts = np.zeros(shape, dtype=np.int32)
    
    print(f"Performing CombMNZ with normalization='{normalization}'...")
    print(f"Shape: {shape[0]} queries x {shape[1]} items")
    print(f"Weights: {weights}")
    print(f"Non-zero threshold: {threshold}")
    
    num_queries = shape[0]
    
    # For each query
    for q_idx in range(num_queries):
        if (q_idx + 1) % 100 == 0:
            print(f"Processing query {q_idx + 1}/{num_queries}...")
        
        # For each similarity matrix
        for mat_idx, mat in enumerate(sim_matrices):
            query_scores = mat[q_idx].astype(np.float32)
            
            # Normalize scores
            normalized_scores = normalize_scores(query_scores, method=normalization)
            
            # Add weighted normalized scores
            fused_scores[q_idx] += weights[mat_idx] * normalized_scores
            
            # Count non-zero scores
            nonzero_mask = normalized_scores > threshold
            nonzero_counts[q_idx] += nonzero_mask.astype(np.int32)
    
    # Multiply by non-zero counts (MNZ part)
    print("\nApplying MNZ multiplier (consensus voting)...")
    fused_scores = fused_scores * nonzero_counts.astype(np.float32)
    
    # Statistics
    avg_consensus = np.mean(nonzero_counts[nonzero_counts > 0])
    max_consensus = np.max(nonzero_counts)
    print(f"Consensus statistics:")
    print(f"  Average non-zero count: {avg_consensus:.2f}")
    print(f"  Maximum consensus: {max_consensus}")
    print(f"  Items with full consensus: {np.sum(nonzero_counts == len(sim_matrices))}")
    
    print("CombMNZ fusion complete!")
    return fused_scores


def evaluate_retrieval(fused_scores, df, video_dir):
    """
    Evaluate retrieval performance using Recall@1, Recall@5, Recall@10
    
    Args:
        fused_scores: numpy array of shape (num_queries, num_videos) or (num_queries, num_unique_videos)
        df: pandas DataFrame with columns [video_id, sentence]
        video_dir: path to video directory
    
    Returns:
        dict with recall metrics
    """
    num_queries = len(df)
    
    # Check if we have a rectangular matrix (queries x unique_videos)
    rectangular = (fused_scores.shape[1] != len(df))
    
    if rectangular:
        # Build unique video list (preserves first-seen order)
        unique_videos = list(dict.fromkeys(df['video_id'].tolist()))
        print(f"Rectangular matrix detected: {fused_scores.shape[0]} queries x {len(unique_videos)} unique videos")
        if len(unique_videos) != fused_scores.shape[1]:
            raise ValueError(f"Unique videos ({len(unique_videos)}) != matrix columns ({fused_scores.shape[1]})")
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    print("\nEvaluating retrieval performance...")
    
    for q_idx in range(num_queries):
        if (q_idx + 1) % 100 == 0:
            print(f"Evaluating query {q_idx + 1}/{num_queries}...")
        
        # Ground truth video for this query
        gt_video_id = df.iloc[q_idx]['video_id']
        
        # Get scores for this query
        query_scores = fused_scores[q_idx]
        
        # Rank by score (higher is better)
        ranked_indices = np.argsort(-query_scores)
        
        # Get predicted video IDs
        if rectangular:
            # Map column indices to video IDs
            predicted_video_ids = [unique_videos[idx] for idx in ranked_indices]
        else:
            # Map row indices to video IDs
            predicted_video_ids = [df.iloc[idx]['video_id'] for idx in ranked_indices]
        
        # Find rank of ground truth
        if gt_video_id in predicted_video_ids:
            gt_rank = predicted_video_ids.index(gt_video_id) + 1
        else:
            gt_rank = num_queries + 1  # not found
        
        # Update recall counts
        if gt_rank <= 1:
            recall_at_1 += 1
        if gt_rank <= 5:
            recall_at_5 += 1
        if gt_rank <= 10:
            recall_at_10 += 1
    
    # Calculate recall percentages
    results = {
        'recall_at_1': recall_at_1 / num_queries,
        'recall_at_5': recall_at_5 / num_queries,
        'recall_at_10': recall_at_10 / num_queries,
        'total_queries': num_queries
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CombMNZ for VATEX Text-to-Video Retrieval (No VLM)")
    
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_path", type=str, default="combmnz_fused_vatex.npy",
                        help="Path to save fused similarity matrix")
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore", "none"],
                        help="Score normalization method (default: minmax)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional weights for each matrix (must sum to 1.0)")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Threshold for non-zero detection (default: 1e-6)")
    
    args = parser.parse_args()
    
    # Validate weights
    if args.weights is not None:
        if len(args.weights) != len(args.sim_paths):
            raise ValueError(f"Number of weights must match number of matrices")
    
    print("=" * 80)
    print("CombMNZ for VATEX T2V Retrieval (No VLM Reranking)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Similarity matrices: {len(args.sim_paths)}")
    for i, path in enumerate(args.sim_paths):
        weight_str = f" (weight: {args.weights[i]:.3f})" if args.weights else ""
        print(f"    {i+1}. {path}{weight_str}")
    print(f"  CSV path: {args.csv_path}")
    print(f"  Video directory: {args.video_dir}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Non-zero threshold: {args.threshold}")
    print(f"  Output path: {args.output_path}")
    print()
    
    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(args.csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} queries")
    
    # Load similarity matrices
    print("\nLoading similarity matrices...")
    sim_matrices = []
    for i, path in enumerate(args.sim_paths):
        print(f"  Loading {i+1}/{len(args.sim_paths)}: {path}")
        mat = np.load(path)
        print(f"    Shape: {mat.shape}")
        print(f"    Score range: [{np.min(mat):.4f}, {np.max(mat):.4f}]")
        sim_matrices.append(mat)
    
    # Perform CombMNZ
    print("\n" + "=" * 80)
    fused_scores = combmnz_fusion(sim_matrices, 
                                   normalization=args.normalization,
                                   weights=args.weights,
                                   threshold=args.threshold)
    
    print(f"\nFused scores statistics:")
    print(f"  Score range: [{np.min(fused_scores):.4f}, {np.max(fused_scores):.4f}]")
    print(f"  Score mean: {np.mean(fused_scores):.4f}, std: {np.std(fused_scores):.4f}")
    
    # Save fused scores
    print(f"\nSaving fused similarity matrix to {args.output_path}...")
    np.save(args.output_path, fused_scores)
    print("Saved!")
    
    # Evaluate
    print("\n" + "=" * 80)
    results = evaluate_retrieval(fused_scores, df, args.video_dir)
    
    print("\n" + "=" * 80)
    print("RESULTS (CombMNZ Only - No VLM)")
    print("=" * 80)
    print(f"Recall@1:  {results['recall_at_1']:.4f} ({results['recall_at_1']*100:.2f}%)")
    print(f"Recall@5:  {results['recall_at_5']:.4f} ({results['recall_at_5']*100:.2f}%)")
    print(f"Recall@10: {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%)")
    print(f"Total queries: {results['total_queries']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
