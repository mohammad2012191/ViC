#!/usr/bin/env python
"""
CombSum (Score Fusion) for Video-to-Text Retrieval (V2T)
No GPU required - pure numpy implementation

CombSum: Simply sums the normalized similarity scores from multiple systems
Formula: score(text) = sum over all systems of: normalized_score_in_system

V2T-specific: For each video query, we rank text candidates
"""

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict


def normalize_scores(scores, method='minmax'):
    """
    Normalize scores for a single query across all candidates.
    
    Args:
        scores: 1D numpy array of scores for one query
        method: 'minmax', 'zscore', or 'none'
    
    Returns:
        normalized scores
    """
    if method == 'none':
        return scores
    
    elif method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-8:  # avoid division by zero
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_score) / (max_score - min_score)
    
    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score < 1e-8:  # avoid division by zero
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - mean_score) / std_score
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def combsum_fusion(sim_matrices, normalization='minmax', weights=None):
    """
    Perform CombSum fusion on multiple similarity matrices for V2T.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_videos, num_texts)
        normalization: 'minmax', 'zscore', or 'none'
        weights: Optional list of weights for each matrix (must sum to 1.0)
    
    Returns:
        fused_scores: numpy array of shape (num_videos, num_texts) with CombSum scores
    """
    num_videos, num_texts = sim_matrices[0].shape
    
    # Validate all matrices have same shape
    for i, mat in enumerate(sim_matrices):
        if mat.shape != (num_videos, num_texts):
            raise ValueError(f"Matrix {i} has shape {mat.shape}, expected {(num_videos, num_texts)}")
    
    # Set up weights
    if weights is None:
        weights = np.ones(len(sim_matrices)) / len(sim_matrices)
    else:
        weights = np.array(weights, dtype=np.float32)
        if len(weights) != len(sim_matrices):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of matrices ({len(sim_matrices)})")
        if not np.isclose(weights.sum(), 1.0):
            print(f"Warning: weights sum to {weights.sum()}, normalizing to 1.0")
            weights = weights / weights.sum()
    
    # Initialize fused scores
    fused_scores = np.zeros((num_videos, num_texts), dtype=np.float32)
    
    print(f"Performing CombSum with normalization='{normalization}' on {len(sim_matrices)} similarity matrices...")
    print(f"Shape: {num_videos} videos (queries) x {num_texts} texts (candidates)")
    print(f"Weights: {weights}")
    
    # For each video query
    for v_idx in range(num_videos):
        if (v_idx + 1) % 100 == 0:
            print(f"Processing video {v_idx + 1}/{num_videos}...")
        
        # For each similarity matrix
        for mat_idx, mat in enumerate(sim_matrices):
            # Get similarity scores for this video across all texts
            video_scores = mat[v_idx].astype(np.float32)
            
            # Normalize scores
            normalized_scores = normalize_scores(video_scores, method=normalization)
            
            # Add weighted normalized scores to fusion
            fused_scores[v_idx] += weights[mat_idx] * normalized_scores
    
    print("CombSum fusion complete!")
    return fused_scores


def evaluate_v2t_retrieval(fused_scores, df, video_dir):
    """
    Evaluate V2T retrieval performance using Recall@1, Recall@5, Recall@10
    
    Args:
        fused_scores: numpy array of shape (num_videos, num_texts)
        df: pandas DataFrame with columns [video_id, sentence]
        video_dir: path to video directory (for validation)
    
    Returns:
        dict with recall metrics
    """
    num_videos = len(df)
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    print("\nEvaluating V2T retrieval performance...")
    
    for v_idx in range(num_videos):
        if (v_idx + 1) % 100 == 0:
            print(f"Evaluating video {v_idx + 1}/{num_videos}...")
        
        # Ground truth text for this video
        gt_text = df.iloc[v_idx]['sentence']
        
        # Get CombSum scores for this video query
        video_scores = fused_scores[v_idx]
        
        # Rank texts by CombSum score (higher is better)
        ranked_indices = np.argsort(-video_scores)
        
        # Get predicted texts
        predicted_texts = [df.iloc[idx]['sentence'] for idx in ranked_indices]
        
        # Find rank of ground truth
        if gt_text in predicted_texts:
            gt_rank = predicted_texts.index(gt_text) + 1  # rank starts at 1
        else:
            gt_rank = num_videos + 1  # not found
        
        # Update recall counts
        if gt_rank <= 1:
            recall_at_1 += 1
        if gt_rank <= 5:
            recall_at_5 += 1
        if gt_rank <= 10:
            recall_at_10 += 1
    
    # Calculate recall percentages
    results = {
        'recall_at_1': recall_at_1 / num_videos,
        'recall_at_5': recall_at_5 / num_videos,
        'recall_at_10': recall_at_10 / num_videos,
        'total_videos': num_videos
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CombSum Fusion for Video-to-Text Retrieval (V2T)")
    
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths (e.g., clip4clip.npy GRAM.npy)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_path", type=str, default="combsum_fused_similarity_v2t.npy",
                        help="Path to save fused similarity matrix")
    parser.add_argument("--normalization", type=str, default="minmax",
                        choices=["minmax", "zscore", "none"],
                        help="Score normalization method (default: minmax)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional weights for each matrix (must sum to 1.0)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate retrieval performance on the fused scores")
    
    args = parser.parse_args()
    
    # Validate weights
    if args.weights is not None:
        if len(args.weights) != len(args.sim_paths):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of matrices ({len(args.sim_paths)})")
    
    print("=" * 80)
    print("CombSum (Score Fusion) for Video-to-Text Retrieval (V2T)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Similarity matrices: {len(args.sim_paths)}")
    for i, path in enumerate(args.sim_paths):
        weight_str = f" (weight: {args.weights[i]:.3f})" if args.weights else ""
        print(f"    {i+1}. {path}{weight_str}")
    print(f"  CSV path: {args.csv_path}")
    print(f"  Video directory: {args.video_dir}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Output path: {args.output_path}")
    print()
    
    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(args.csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} video-text pairs")
    
    # Load similarity matrices
    print("\nLoading similarity matrices...")
    sim_matrices = []
    for i, path in enumerate(args.sim_paths):
        print(f"  Loading {i+1}/{len(args.sim_paths)}: {path}")
        mat = np.load(path)
        print(f"    Shape: {mat.shape}")
        print(f"    Score range: [{np.min(mat):.4f}, {np.max(mat):.4f}]")
        print(f"    Score mean: {np.mean(mat):.4f}, std: {np.std(mat):.4f}")
        
        # Validate shape - for V2T, rows are videos, columns are texts
        if mat.shape[0] != len(df) or mat.shape[1] != len(df):
            raise ValueError(f"Matrix {path} has shape {mat.shape}, expected ({len(df)}, {len(df)})")
        
        sim_matrices.append(mat.T)
    
    # Perform CombSum
    print("\n" + "=" * 80)
    fused_scores = combsum_fusion(sim_matrices, normalization=args.normalization, weights=args.weights)
    
    print(f"\nFused scores statistics:")
    print(f"  Score range: [{np.min(fused_scores):.4f}, {np.max(fused_scores):.4f}]")
    print(f"  Score mean: {np.mean(fused_scores):.4f}, std: {np.std(fused_scores):.4f}")
    
    # Save fused scores
    print(f"\nSaving fused similarity matrix to {args.output_path}...")
    np.save(args.output_path, fused_scores)
    print("Saved!")
    
    # Evaluate if requested
    if args.evaluate:
        print("\n" + "=" * 80)
        results = evaluate_v2t_retrieval(fused_scores, df, args.video_dir)
        
        print("\n" + "=" * 80)
        print("RESULTS (V2T)")
        print("=" * 80)
        print(f"Recall@1:  {results['recall_at_1']:.4f} ({results['recall_at_1']*100:.2f}%)")
        print(f"Recall@5:  {results['recall_at_5']:.4f} ({results['recall_at_5']*100:.2f}%)")
        print(f"Recall@10: {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%)")
        print(f"Total videos: {results['total_videos']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
