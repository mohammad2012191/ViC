#!/usr/bin/env python
import os
import json
import torch
import torch.distributed as dist
import transformers
import warnings
import logging
import numpy as np
import pandas as pd
import argparse
from ensemble_msrvtt_activitynet_didemo_v2t import run_distributed_inference
from ensemble_msrvtt_activitynet_didemo_v2t import Config

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
logging.getLogger("torch.cuda").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Video-to-Text Retrieval with InternVL")
    
    # Required arguments
    parser.add_argument("--sim_paths", nargs="+", required=True,
                        help="List of similarity matrix paths (e.g., clip4clip_msrvtt.npy GRAM_msrvtt_tvas.npy) - ORDER MATTERS!!!")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: key, vid_key, video_id, sentence")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--num_captions", type=int, required=True,
                        help="Number of text candidates to select from similarity matrix")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., OpenGVLab/InternVL3_5-38B)")
    parser.add_argument("--grid_size", type=int, required=True,
                        help="Grid size for video frame sampling (e.g., 3 for 3x3, 4 for 4x4)")
    
    
    # Subtitle arguments
    parser.add_argument("--use_subs", action="store_true",
                        help="Whether to use subtitles in prompts")
    parser.add_argument("--subtitle_json", type=str, default=None,
                        help="Path to subtitle JSON file (required if --use_subs is set)")
    
    parser.add_argument("--ensemble_mode", type=str, default="hard_dup",
                    choices=["soft", "hard_dup", "hard_unique", "none"],
                    help="Ensemble mode: soft, hard_dup, hard_unique, or none")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate subtitle arguments
    if args.use_subs and args.subtitle_json is None:
        raise ValueError("--subtitle_json must be provided when --use_subs is set")
    
    ##################################################
    # 1) Init process group (distributed)
    ##################################################
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    ##################################################
    # 2) Update Config with command-line arguments
    ##################################################
    Config.num_captions = args.num_captions
    Config.video_dir = args.video_dir
    Config.model_name = args.model_name
    Config.grid_size = args.grid_size
    Config.use_subs = args.use_subs
    Config.subtitle_json = args.subtitle_json
    Config.ensemble_mode = args.ensemble_mode
    
    ##################################################
    # 3) Load CSV + similarity matrix
    ##################################################
    df = pd.read_csv(args.csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
        
    sim_mats = [np.load(p).T for p in args.sim_paths]
    
    # Sanity check: CSV length must match first dimension of each mat
    for p, m in zip(args.sim_paths, sim_mats):
        if m.shape[0] != len(df) and m.shape[1] != len(df):
            raise ValueError(f"{p} shape {m.shape} incompatible with CSV len={len(df)}")
    
    if rank == 0:
        print(f"Loaded {len(sim_mats)} similarity matrices")
        print(f"Model: {args.model_name}")
        print(f"Grid size: {args.grid_size}x{args.grid_size}")
        print(f"Ensemble mode: {args.ensemble_mode}")  # ADD THIS LINE
        print(f"Use subtitles: {args.use_subs}")
        if args.use_subs:
            print(f"Subtitle JSON: {args.subtitle_json}")
    
    ##################################################
    # 4) Determine slice of data
    ##################################################
    N = len(df)
    all_indices = list(range(N))
    data_slice = all_indices[rank::world_size]
    print(f"[Rank {rank}] local_rank={local_rank}, we have {len(data_slice)} items to process.")
    
    ##################################################
    # 5) Run the inference on this slice
    ##################################################
    c1, c5, c10, local_count = run_distributed_inference(local_rank, data_slice, df, sim_mats) 
    
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
        print("\n==================== Final Results ====================")
        print(f"Recall@1:  {recall_1:.4f}")
        print(f"Recall@5:  {recall_5:.4f}")
        print(f"Recall@10: {recall_10:.4f}")
        print(f"Total processed: {int(total_cnt)}/{N}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()