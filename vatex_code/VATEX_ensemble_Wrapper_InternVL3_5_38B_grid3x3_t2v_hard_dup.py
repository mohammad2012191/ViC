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
from VATEX_ensemble_InternVL3_5_38B_grid3x3_t2v_hard_dup import run_distributed_inference
from VATEX_ensemble_InternVL3_5_38B_grid3x3_t2v_hard_dup import Config

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
    csv_path = "descs_ret_test_vatex.csv"  # The CSV with columns: key, vid_key, video_id, sentence
    df = pd.read_csv(csv_path)
    df.sort_values('video_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    ###################################################################################################################
    sim_paths = [
                # "InternVideo/InternVideo2/multi_modality/sim_TxV.npy",
                 # "VAST/vast_sim_pruned_12520x1252.npy",
                 # "similarity_matrices/clip4clip_activitynet.npy",
                 "VAST_vatex_final.npy",
                 # "similarity_matrices/GRAM_activitynet_tva.npy",
                 "InternVideo2_vatex_final.npy"
                    ]  # pass list via env or CLI
    sim_mats  = [np.load(p) for p in sim_paths]
    # sanity: CSV length must match first dimension of each mat (after any .T you do)
    for p, m in zip(sim_paths, sim_mats):
        if m.shape[0] != len(df) and m.shape[1] != len(df):
            raise ValueError(f"{p} shape {m.shape} incompatible with CSV len={len(df)}")
    print(f"[Rank {rank}] loaded {len(sim_mats)} similarity matrices")

    
    ##################################################
    # 3) Determine slice of data
    #    We'll distribute row indices among ranks.
    ##################################################
    N = len(df)
    all_indices = list(range(N))
    # e.g. rank0 => 0, world_size, 2*world_size, ...
    data_slice = all_indices[rank::world_size]
    print(f"[Rank {rank}] local_rank={local_rank}, we have {len(data_slice)} items to process.")

    ##################################################
    # 4) Run the inference on this slice
    ##################################################
    c1, c5, c10, local_count = run_distributed_inference(local_rank, data_slice, df, sim_mats)

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
