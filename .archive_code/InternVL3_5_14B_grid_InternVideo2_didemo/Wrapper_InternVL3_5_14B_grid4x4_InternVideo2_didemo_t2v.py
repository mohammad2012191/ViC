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
from InternVL3_5_14B_grid4x4_InternVideo2_didemo_t2v import run_distributed_inference
from InternVL3_5_14B_grid4x4_InternVideo2_didemo_t2v import Config

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
logging.getLogger("torch.cuda").setLevel(logging.ERROR)

def main():
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
    csv_path = "descs_ret_test_didemo.csv"  # The CSV with columns: key, vid_key, video_id, sentence
    # sim_path = "sim_matrix_clip4clip.npy"  # NxN similarity matrix
    # sim_path = "sim_matrix_Vast_tvas.npy"  # NxN similarity matrix
    sim_path = "InternVideo2_didemo.npy"  # NxN similarity matrix

    df = pd.read_csv(csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    ###################################################################################################################

    #### FOR OTHER MODELS #############################################################################################
    # # Load CSV
    # df = pd.read_csv(csv_path)  # or whatever your CSV delimiter is
    # # The DataFrame must have columns named: "video_id", "sentence", etc.
    # # If your CSV has headers: ["key","vid_key","video_id","sentence"], 
    # # then df["video_id"] and df["sentence"] are valid.
    ###################################################################################################################

    # Load similarity matrix
    sim_matrix = np.load(sim_path)  # shape (N, N)
    N = len(df)
    if sim_matrix.shape[0] != N:
        raise ValueError("Sim matrix shape mismatch with CSV length!")
    print(f"[Rank {rank}] Loaded CSV of size {N}, sim_matrix {sim_matrix.shape}.")

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
