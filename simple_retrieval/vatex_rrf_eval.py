#!/usr/bin/env python
"""
Reciprocal Rank Fusion (RRF) for VATEX Text-to-Video Retrieval
No GPU required - pure numpy implementation
Evaluates ensemble results directly without VLM reranking
"""

import os
import numpy as np
import pandas as pd
import argparse


def reciprocal_rank_fusion(sim_matrices, k=60):
    """
    Perform Reciprocal Rank Fusion on multiple similarity matrices.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_queries, num_videos)
        k: RRF constant (default 60)
    
    Returns:
        fused_scores: numpy array with RRF scores
    """
    shape = sim_matrices[0].shape
    
    # Validate all matrices have same shape
    for i, mat in enumerate(sim_matrices):
        if mat.shape != shape:
            raise ValueError(f"Matrix {i} has shape {mat.shape}, expected {shape}")
    
    # Initialize fused scores
    fused_scores = np.zeros(shape, dtype=np.float32)
    
    print(f"Performing RRF with k={k} on {len(sim_matrices)} similarity matrices...")
    print(f"Shape: {shape[0]} queries x {shape[1]} items")
    
    num_queries = shape[0]
    
    # For each query
    for q_idx in range(num_queries):
        if (q_idx + 1) % 100 == 0:
            print(f"Processing query {q_idx + 1}/{num_queries}...")
        
        # For each similarity matrix
        for mat in sim_matrices:
            # Get similarity scores for this query
            query_scores = mat[q_idx]
            
            # Rank items by similarity (higher is better)
            ranked_indices = np.argsort(-query_scores)
            
            # Assign RRF scores
            for rank_position, item_idx in enumerate(ranked_indices):
                rank = rank_position + 1
                rrf_score = 1.0 / (k + rank)
                fused_scores[q_idx, item_idx] += rrf_score
    
    print("RRF fusion complete!")
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
    parser = argparse.ArgumentParser(description="RRF for VATEX Text-to-Video Retrieval (No VLM)")
    
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_path", type=str, default="rrf_fused_vatex.npy",
                        help="Path to save fused similarity matrix")
    parser.add_argument("--k", type=int, default=60,
                        help="RRF constant k (default: 60)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RRF for VATEX T2V Retrieval (No VLM Reranking)")
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
    print(f"Loaded {len(df)} queries")
    
    # Load similarity matrices
    print("\nLoading similarity matrices...")
    sim_matrices = []
    for i, path in enumerate(args.sim_paths):
        print(f"  Loading {i+1}/{len(args.sim_paths)}: {path}")
        mat = np.load(path)
        print(f"    Shape: {mat.shape}")
        sim_matrices.append(mat)
    
    # Perform RRF
    print("\n" + "=" * 80)
    fused_scores = reciprocal_rank_fusion(sim_matrices, k=args.k)
    
    # Save fused scores
    print(f"\nSaving fused similarity matrix to {args.output_path}...")
    np.save(args.output_path, fused_scores)
    print("Saved!")
    
    # Evaluate
    print("\n" + "=" * 80)
    results = evaluate_retrieval(fused_scores, df, args.video_dir)
    
    print("\n" + "=" * 80)
    print("RESULTS (RRF Only - No VLM)")
    print("=" * 80)
    print(f"Recall@1:  {results['recall_at_1']:.4f} ({results['recall_at_1']*100:.2f}%)")
    print(f"Recall@5:  {results['recall_at_5']:.4f} ({results['recall_at_5']*100:.2f}%)")
    print(f"Recall@10: {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%)")
    print(f"Total queries: {results['total_queries']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
