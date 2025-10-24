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
from msrvtt_activitynet_didemo_v2t import run_distributed_inference
from msrvtt_activitynet_didemo_v2t import Config

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
logging.getLogger("torch.cuda").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description='Run distributed video-to-text inference')
    parser.add_argument('--sim_path', type=str, required=True,
                        help='Path to similarity matrix (.npy file)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with video descriptions')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--num_captions', type=int, default=20,
                        help='Number of text candidates to select (default: 20)')
    parser.add_argument('--model_name', type=str, default='OpenGVLab/InternVL3_5-38B',
                        help='Model name/path (default: OpenGVLab/InternVL3_5-38B)')
    parser.add_argument('--grid_size', type=int, default=3,
                        help='Grid size for frame sampling (default: 3 for 3x3 grid)')
    parser.add_argument('--use_subs', action='store_true',
                        help='Enable subtitle usage')
    parser.add_argument('--subtitle_json', type=str, default=None,
                        help='Path to subtitle JSON file (required if --use_subs is set)')
    return parser.parse_args()

def main():
    ##################################################
    # Parse command-line arguments
    ##################################################
    args = parse_args()
    
    # Validate subtitle arguments
    if args.use_subs and args.subtitle_json is None:
        raise ValueError("--subtitle_json must be provided when --use_subs is enabled")
    
    # Set config values from arguments
    Config.num_captions = args.num_captions
    Config.video_dir = args.video_dir
    Config.model_name = args.model_name
    Config.grid_size = args.grid_size
    Config.use_subs = args.use_subs
    Config.subtitle_json = args.subtitle_json
    
    ##################################################
    # 1) Init process group (distributed)
    ##################################################
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    ##################################################
    # 2) Load CSV + similarity matrix
    #    We do it on every rank for simplicity.
    ##################################################
    csv_path = args.csv_path
    sim_path = args.sim_path
    
    df = pd.read_csv(csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Load similarity matrix
    sim_matrix = np.load(sim_path).T  # shape (N, N)
    N = len(df)
    if sim_matrix.shape[0] != N:
        raise ValueError("Sim matrix shape mismatch with CSV length!")
    
    if rank == 0:
        print(f"[Rank {rank}] Loaded CSV of size {N}, sim_matrix {sim_matrix.shape}.")
        print(f"[Rank {rank}] Configuration:")
        print(f"  - Model: {Config.model_name}")
        print(f"  - Grid size: {Config.grid_size}x{Config.grid_size}")
        print(f"  - Num captions: {Config.num_captions}")
        print(f"  - Video dir: {Config.video_dir}")
        print(f"  - Use subtitles: {Config.use_subs}")
        if Config.use_subs:
            print(f"  - Subtitle JSON: {Config.subtitle_json}")

    if N == 0:
        print(f"[Rank {rank}] No data, exiting.")
        dist.destroy_process_group()
        return

    ##################################################
    # 3) Determine slice of data
    #    We'll distribute row indices among ranks.
    ##################################################
    all_indices = list(range(N))
    # e.g. rank0 => 0, world_size, 2*world_size, ...
    data_slice = all_indices[rank::world_size]
    print(f"[Rank {rank}] local_rank={local_rank}, we have {len(data_slice)} items to process.")

    ##################################################
    # 4) Run the inference on this slice
    ##################################################
    c1, c5, c10, local_count = run_distributed_inference(local_rank, data_slice, df, sim_matrix)

    # Convert partial stats to a tensor
    part_tensor = torch.tensor([c1, c5, c10, local_count], dtype=torch.float32, device=local_rank)

    ##################################################
    # 5) All-reduce the sums
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