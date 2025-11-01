#!/usr/bin/env python
"""
Ensemble Methods for Text-to-Video and Video-to-Text Retrieval
No GPU required - pure numpy implementation

Supports:
- CombSum (Score Fusion)
- CombMNZ (Combined Multiple Non-Zero)
- RRF (Reciprocal Rank Fusion)

Both T2V and V2T modes supported
"""

import numpy as np
import pandas as pd


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


def combsum_fusion(sim_matrices, normalization='minmax', weights=None, verbose=True):
    """
    Perform CombSum fusion on multiple similarity matrices.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_queries, num_items)
        normalization: 'minmax', 'zscore', or 'none'
        weights: Optional list of weights for each matrix (must sum to 1.0)
        verbose: Whether to print progress
    
    Returns:
        fused_scores: numpy array with CombSum scores
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
            raise ValueError(f"Number of weights ({len(weights)}) must match number of matrices ({len(sim_matrices)})")
        if not np.isclose(weights.sum(), 1.0):
            if verbose:
                print(f"Warning: weights sum to {weights.sum()}, normalizing to 1.0")
            weights = weights / weights.sum()
    
    # Initialize fused scores
    fused_scores = np.zeros(shape, dtype=np.float32)
    
    if verbose:
        print(f"Performing CombSum with normalization='{normalization}'...")
        print(f"Shape: {shape[0]} queries x {shape[1]} items")
        print(f"Weights: {weights}")
    
    num_queries = shape[0]
    
    # For each query
    for q_idx in range(num_queries):
        if verbose and (q_idx + 1) % 100 == 0:
            print(f"Processing query {q_idx + 1}/{num_queries}...")
        
        # For each similarity matrix
        for mat_idx, mat in enumerate(sim_matrices):
            query_scores = mat[q_idx].astype(np.float32)
            
            # Normalize scores
            normalized_scores = normalize_scores(query_scores, method=normalization)
            
            # Add weighted normalized scores
            fused_scores[q_idx] += weights[mat_idx] * normalized_scores
    
    if verbose:
        print("CombSum fusion complete!")
    
    return fused_scores


def combmnz_fusion(sim_matrices, normalization='minmax', weights=None, threshold=1e-6, verbose=True):
    """
    Perform CombMNZ fusion on multiple similarity matrices.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_queries, num_items)
        normalization: 'minmax', 'zscore', or 'none'
        weights: Optional list of weights for each matrix (must sum to 1.0)
        threshold: Threshold for considering a score as "non-zero"
        verbose: Whether to print progress
    
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
            raise ValueError(f"Number of weights ({len(weights)}) must match number of matrices ({len(sim_matrices)})")
        if not np.isclose(weights.sum(), 1.0):
            if verbose:
                print(f"Warning: weights sum to {weights.sum()}, normalizing to 1.0")
            weights = weights / weights.sum()
    
    # Initialize fused scores and non-zero counts
    fused_scores = np.zeros(shape, dtype=np.float32)
    nonzero_counts = np.zeros(shape, dtype=np.int32)
    
    if verbose:
        print(f"Performing CombMNZ with normalization='{normalization}'...")
        print(f"Shape: {shape[0]} queries x {shape[1]} items")
        print(f"Weights: {weights}")
        print(f"Non-zero threshold: {threshold}")
    
    num_queries = shape[0]
    
    # For each query
    for q_idx in range(num_queries):
        if verbose and (q_idx + 1) % 100 == 0:
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
    
    # Multiply by non-zero counts (this is the MNZ part)
    if verbose:
        print("\nApplying MNZ multiplier (consensus voting)...")
    
    fused_scores = fused_scores * nonzero_counts.astype(np.float32)
    
    # Statistics on consensus
    if verbose:
        avg_consensus = np.mean(nonzero_counts[nonzero_counts > 0])
        max_consensus = np.max(nonzero_counts)
        print(f"Consensus statistics:")
        print(f"  Average non-zero count: {avg_consensus:.2f}")
        print(f"  Maximum consensus: {max_consensus}")
        print(f"  Items with full consensus: {np.sum(nonzero_counts == len(sim_matrices))}")
        print("CombMNZ fusion complete!")
    
    return fused_scores


def rrf_fusion(sim_matrices, k=60, verbose=True):
    """
    Perform Reciprocal Rank Fusion on multiple similarity matrices.
    
    Args:
        sim_matrices: List of numpy arrays, each of shape (num_queries, num_items)
        k: RRF constant (default 60, as used in the original paper)
        verbose: Whether to print progress
    
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
    
    if verbose:
        print(f"Performing RRF with k={k} on {len(sim_matrices)} similarity matrices...")
        print(f"Shape: {shape[0]} queries x {shape[1]} items")
    
    num_queries = shape[0]
    
    # For each query
    for q_idx in range(num_queries):
        if verbose and (q_idx + 1) % 100 == 0:
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
    
    if verbose:
        print("RRF fusion complete!")
    
    return fused_scores


def evaluate_retrieval(fused_scores, df, video_dir=None, mode='t2v', verbose=True):
    """
    Evaluate retrieval performance using Recall@1, Recall@5, Recall@10
    
    Args:
        fused_scores: numpy array of shape (num_queries, num_items)
        df: pandas DataFrame with columns [video_id, sentence]
        video_dir: path to video directory (optional, for validation)
        mode: 't2v' for text-to-video or 'v2t' for video-to-text
        verbose: Whether to print progress
    
    Returns:
        dict with recall metrics
    """
    num_queries = len(df)
    
    # Check if we have a rectangular matrix (queries x unique_items)
    rectangular = (fused_scores.shape[1] != len(df))
    
    if rectangular:
        # Build unique video list (preserves first-seen order)
        unique_videos = list(dict.fromkeys(df['video_id'].tolist()))
        if verbose:
            print(f"Rectangular matrix detected: {fused_scores.shape[0]} queries x {len(unique_videos)} unique videos")
        if len(unique_videos) != fused_scores.shape[1]:
            raise ValueError(f"Unique videos ({len(unique_videos)}) != matrix columns ({fused_scores.shape[1]})")
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    if verbose:
        print(f"\nEvaluating {mode.upper()} retrieval performance...")
    
    for q_idx in range(num_queries):
        if verbose and (q_idx + 1) % 100 == 0:
            print(f"Evaluating query {q_idx + 1}/{num_queries}...")
        
        # Ground truth
        if mode == 't2v':
            gt_item = df.iloc[q_idx]['video_id']
        else:  # v2t
            gt_item = df.iloc[q_idx]['sentence']
        
        # Get scores for this query
        query_scores = fused_scores[q_idx]
        
        # Rank by score (higher is better)
        ranked_indices = np.argsort(-query_scores)
        
        # Get predicted items
        if mode == 't2v':
            if rectangular:
                # Map column indices to video IDs
                predicted_items = [unique_videos[idx] for idx in ranked_indices]
            else:
                # Map row indices to video IDs
                predicted_items = [df.iloc[idx]['video_id'] for idx in ranked_indices]
        else:  # v2t
            predicted_items = [df.iloc[idx]['sentence'] for idx in ranked_indices]
        
        # Find rank of ground truth
        if gt_item in predicted_items:
            gt_rank = predicted_items.index(gt_item) + 1
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
        'total_queries': num_queries,
        'mode': mode
    }
    
    return results


def load_similarity_matrices(sim_paths, df, mode='t2v', verbose=True):
    """
    Load similarity matrices from disk and validate their shapes.
    
    Args:
        sim_paths: List of paths to .npy files
        df: pandas DataFrame for validation
        mode: 't2v' or 'v2t'
        verbose: Whether to print progress
    
    Returns:
        List of numpy arrays (transposed for v2t mode)
    """
    if verbose:
        print("\nLoading similarity matrices...")
    
    sim_matrices = []
    for i, path in enumerate(sim_paths):
        if verbose:
            print(f"  Loading {i+1}/{len(sim_paths)}: {path}")
        
        mat = np.load(path)
        
        if verbose:
            print(f"    Shape: {mat.shape}")
            print(f"    Score range: [{np.min(mat):.4f}, {np.max(mat):.4f}]")
            print(f"    Score mean: {np.mean(mat):.4f}, std: {np.std(mat):.4f}")
        
        # Validate shape
        if mode == 't2v':
            if mat.shape[0] != len(df):
                raise ValueError(f"Matrix {path} has {mat.shape[0]} rows, expected {len(df)}")
        else:  # v2t
            # For V2T, we expect square matrices and will transpose them
            if mat.shape[0] != len(df) or mat.shape[1] != len(df):
                raise ValueError(f"Matrix {path} has shape {mat.shape}, expected ({len(df)}, {len(df)})")
            mat = mat.T  # Transpose for V2T
        
        sim_matrices.append(mat)
    
    return sim_matrices


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 80)
    print(f"RESULTS ({results['mode'].upper()})")
    print("=" * 80)
    print(f"Recall@1:  {results['recall_at_1']:.4f} ({results['recall_at_1']*100:.2f}%)")
    print(f"Recall@5:  {results['recall_at_5']:.4f} ({results['recall_at_5']*100:.2f}%)")
    print(f"Recall@10: {results['recall_at_10']:.4f} ({results['recall_at_10']*100:.2f}%)")
    print(f"Total queries: {results['total_queries']}")
    print("=" * 80)
