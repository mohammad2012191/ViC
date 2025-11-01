#!/usr/bin/env python
import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import transformers
import warnings
import logging
import numpy as np
import pandas as pd
from VATEX_retrieval import run_distributed_inference
from VATEX_retrieval import Config

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
logging.getLogger("torch.cuda").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Video-Text Retrieval with InternVL")
    
    # Required arguments
    parser.add_argument("--retrieval_mode", type=str, required=True,
                        choices=["t2v", "v2t"],
                        help="Retrieval mode: t2v (text-to-video) or v2t (video-to-text)")
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths (e.g., clip4clip_msrvtt.npy GRAM_msrvtt.npy) - ORDER MATTERS!!!")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: key, vid_key, video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., OpenGVLab/InternVL3_5-38B)")
    parser.add_argument("--grid_size", type=int, required=True,
                        help="Grid size for video frame sampling (e.g., 3 for 3x3, 4 for 4x4)")
    
    # Mode-specific arguments
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of video candidates to select (for t2v mode)")
    parser.add_argument("--num_captions", type=int, default=None,
                        help="Number of text candidates to select (for v2t mode)")
    
    # Ensemble mode argument
    parser.add_argument("--ensemble_mode", type=str, default="none",
                        choices=["ViC_duplicate", "ViC_unique" , "none"],
                        help="Ensemble mode: ViC_duplicate, ViC_unique, or none")
    
    # Ensemble weights (optional)
    parser.add_argument("--ensemble_weights", nargs="+", type=float, default=None,
                        help="Weights for soft ensemble mode (must match number of sim_paths)")
    
    # Per-model top-k for hard ensemble modes
    parser.add_argument("--per_model_topk", type=int, default=4,
                        help="Number of candidates to take from each model in hard ensemble modes")
    
    # Subtitle arguments
    parser.add_argument("--use_subs", action="store_true",
                        help="Whether to use subtitles in prompts")
    parser.add_argument("--subtitle_json", type=str, default=None,
                        help="Path to subtitle JSON file (required if --use_subs is set)")
    
    args = parser.parse_args()
    
    # Validation
    if args.retrieval_mode == "t2v" and args.num_images is None:
        parser.error("--num_images is required for t2v mode")
    if args.retrieval_mode == "v2t" and args.num_captions is None:
        parser.error("--num_captions is required for v2t mode")
    
    if args.use_subs and args.subtitle_json is None:
        parser.error("--subtitle_json is required when --use_subs is set")
    
    if args.ensemble_mode == "soft" and args.ensemble_weights is not None:
        if len(args.ensemble_weights) != len(args.sim_paths):
            parser.error(f"Number of ensemble_weights ({len(args.ensemble_weights)}) must match number of sim_paths ({len(args.sim_paths)})")
    
    return args


def main():
    # Parse arguments
    args = parse_args()
    
    ##################################################
    # 1) Init process group (distributed)
    ##################################################
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        print("="*80)
        print("Configuration:")
        print(f"  Retrieval Mode: {args.retrieval_mode}")
        print(f"  CSV Path: {args.csv_path}")
        print(f"  Video Directory: {args.video_dir}")
        print(f"  Model: {args.model_name}")
        print(f"  Grid Size: {args.grid_size}")
        print(f"  Similarity Matrices: {args.sim_paths}")
        print(f"  Ensemble Mode: {args.ensemble_mode}")
        if args.ensemble_mode != "none":
            print(f"  Ensemble Weights: {args.ensemble_weights}")
            if args.ensemble_mode in ["hard_dup", "hard_unique"]:
                print(f"  Per-Model Top-K: {args.per_model_topk}")
        if args.retrieval_mode == "t2v":
            print(f"  Num Images (candidates): {args.num_images}")
        else:
            print(f"  Num Captions (candidates): {args.num_captions}")
        print(f"  Use Subtitles: {args.use_subs}")
        if args.use_subs:
            print(f"  Subtitle JSON: {args.subtitle_json}")
        print("="*80)

    ##################################################
    # 2) Set Config from args
    ##################################################
    Config.video_dir = args.video_dir
    Config.model_name = args.model_name
    Config.grid_size = args.grid_size
    Config.num_images = args.num_images
    Config.num_captions = args.num_captions
    Config.ensemble_mode = args.ensemble_mode if args.ensemble_mode != "none" else None
    Config.ensemble_weights = args.ensemble_weights
    Config.per_model_topk = args.per_model_topk
    Config.use_subs = args.use_subs
    Config.subtitle_json = args.subtitle_json

    ##################################################
    # 3) Load CSV + similarity matrices
    ##################################################
    df = pd.read_csv(args.csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Load similarity matrices
    sim_mats = [np.load(p) for p in args.sim_paths]
    
    # Sanity check
    for p, m in zip(args.sim_paths, sim_mats):
        if m.shape[0] != len(df) and m.shape[1] != len(df):
            raise ValueError(f"{p} shape {m.shape} incompatible with CSV len={len(df)}")
    
    if rank == 0:
        print(f"Loaded {len(sim_mats)} similarity matrices")
        for i, (p, m) in enumerate(zip(args.sim_paths, sim_mats)):
            print(f"  Matrix {i+1}: {p} - shape {m.shape}")

    ##################################################
    # 4) Determine slice of data
    ##################################################
    N = len(df)
    all_indices = list(range(N))
    data_slice = all_indices[rank::world_size]
    
    if rank == 0:
        print(f"\nTotal items: {N}")
        print(f"World size: {world_size}")
    print(f"[Rank {rank}] Processing {len(data_slice)} items")

    ##################################################
    # 5) Run the inference on this slice
    ##################################################
    c1, c5, c10, local_count = run_distributed_inference(
        local_rank, data_slice, df, sim_mats, args.retrieval_mode
    )

    # Convert partial stats to a tensor
    part_tensor = torch.tensor([c1, c5, c10, local_count], dtype=torch.float32, device=local_rank)

    ##################################################
    # 6) All-reduce the sums
    ##################################################
    dist.all_reduce(part_tensor, op=dist.ReduceOp.SUM)

    # If rank=0, print final results
    if rank == 0:
        total_c1, total_c5, total_c10, total_cnt = part_tensor.cpu().tolist()
        recall_1 = total_c1 / total_cnt if total_cnt > 0 else 0.0
        recall_5 = total_c5 / total_cnt if total_cnt > 0 else 0.0
        recall_10 = total_c10 / total_cnt if total_cnt > 0 else 0.0

        print("\n" + "="*80)
        print("Final Results")
        print("="*80)
        print(f"Recall@1:  {recall_1:.4f}")
        print(f"Recall@5:  {recall_5:.4f}")
        print(f"Recall@10: {recall_10:.4f}")
        print(f"Total processed: {int(total_cnt)}/{N}")
        print("="*80)

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
