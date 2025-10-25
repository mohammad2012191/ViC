#!/usr/bin/env python
"""
Reciprocal Rank Fusion (RRF) for Video-to-Text Retrieval (V2T)
No GPU required - pure numpy implementation

RRF Formula: score(text) = sum over all systems of: 1 / (k + rank_in_system)
where k is a constant (typically 60) and rank starts from 1

V2T-specific: For each video query, we rank text candidates
"""

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict


def reciprocal_rank_fusion(sim_matrices, k=60):
    """
    Perform Reciprocal Rank Fusion on multiple similarity matrices for V2T.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_videos, num_texts)
        k: RRF constant (default 60, as used in the original paper)
    
    Returns:
        fused_scores: numpy array of shape (num_videos, num_texts) with RRF scores
    """
    num_videos, num_texts = sim_matrices[0].shape
    
    # Validate all matrices have same shape
    for i, mat in enumerate(sim_matrices):
        if mat.shape != (num_videos, num_texts):
            raise ValueError(f"Matrix {i} has shape {mat.shape}, expected {(num_videos, num_texts)}")
    
    # Initialize fused scores
    fused_scores = np.zeros((num_videos, num_texts), dtype=np.float32)
    
    print(f"Performing RRF with k={k} on {len(sim_matrices)} similarity matrices...")
    print(f"Shape: {num_videos} videos (queries) x {num_texts} texts (candidates)")
    
    # For each video query
    for v_idx in range(num_videos):
        if (v_idx + 1) % 100 == 0:
            print(f"Processing video {v_idx + 1}/{num_videos}...")
        
        # For each similarity matrix
        for mat in sim_matrices:
            # Get similarity scores for this video across all texts
            video_scores = mat[v_idx]
            
            # Rank texts by similarity (higher is better)
            # argsort in descending order
            ranked_text_indices = np.argsort(-video_scores)
            
            # Assign RRF scores
            for rank_position, text_idx in enumerate(ranked_text_indices):
                # rank_position starts at 0, but rank should start at 1
                rank = rank_position + 1
                rrf_score = 1.0 / (k + rank)
                fused_scores[v_idx, text_idx] += rrf_score
    
    print("RRF fusion complete!")
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
        
        # Get RRF scores for this video query
        video_scores = fused_scores[v_idx]
        
        # Rank texts by RRF score (higher is better)
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
    parser = argparse.ArgumentParser(description="Reciprocal Rank Fusion for Video-to-Text Retrieval (V2T)")
    
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths (e.g., clip4clip.npy GRAM.npy)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_path", type=str, default="rrf_fused_similarity_v2t.npy",
                        help="Path to save fused similarity matrix")
    parser.add_argument("--k", type=int, default=60,
                        help="RRF constant k (default: 60)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate retrieval performance on the fused scores")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Reciprocal Rank Fusion (RRF) for Video-to-Text Retrieval (V2T)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Similarity matrices: {len(args.sim_paths)}")
    for i, path in enumerate(args.sim_paths):
        print(f"    {i+1}. {path}")
    print(f"  CSV path: {args.csv_path}")
    print(f"  Video directory: {args.video_dir}")
    print(f"  RRF constant k: {args.k}")
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
        
        # Validate shape - for V2T, rows are videos, columns are texts
        if mat.shape[0] != len(df) or mat.shape[1] != len(df):
            raise ValueError(f"Matrix {path} has shape {mat.shape}, expected ({len(df)}, {len(df)})")
        
        sim_matrices.append(mat.T)
    
    # Perform RRF
    print("\n" + "=" * 80)
    fused_scores = reciprocal_rank_fusion(sim_matrices, k=args.k)
    
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
