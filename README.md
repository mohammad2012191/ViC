# Quick Start

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for distributed training)
    - 1 A100 is enough
- conda or pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/mohammad2012191/Vote-in-Context-Fusion.git
cd Vote-in-Context-Fusion

# Using conda 
conda env create -f environment.yml
conda activate vote-in-context-env
```

## Required Data

1. **CSV File**: Contains video metadata with columns: `key`, `vid_key`, `video_id`, `sentence`
2. **Similarity Matrices**: Pre-computed `.npy` files from retrieval models (e.g., CLIP4Clip, GRAM, InternVideo2)
3. **Video Directory**: Folder containing video files all are mp4 format
4. **Subtitle JSON** (optional): JSON file with video subtitles

##  ViC Usage

### Text-to-Video Retrieval (Ensemble Mode)
```bash
torchrun --nproc_per_node=3 Msrvtt_activitynet_didemo_code\ensemble_wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_paths Similarity_Matrices\clip4clip_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy Similarity_Matrices\InternVideo2_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode ViC_duplicate \
  2>&1 | tee logs/msrvtt_t2v_ensemble.log
```

### Video-to-Text Retrieval (Ensemble Mode)

```bash
torchrun --nproc_per_node=3 Msrvtt_activitynet_didemo_code\ensemble_wrapper_msrvtt_activitynet_didemo_v2t.py \
  --sim_paths Similarity_Matrices\clip4clip_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy Similarity_Matrices\InternVideo2_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_captions 20 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode ViC_duplicate \
  2>&1 | tee logs/msrvtt_v2t_ensemble.log
```

### Single Similarity Matrix Mode

For single similarity matrix (no ensemble):

```bash
torchrun --nproc_per_node=3 Msrvtt_activitynet_didemo_code\wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_path Similarity_Matrices\clip4clip_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  2>&1 | tee logs/msrvtt_t2v_single.log
```
### With Subtitles

```bash
torchrun --nproc_per_node=3 Msrvtt_activitynet_didemo_code\ensemble_wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_paths Similarity_Matrices\clip4clip_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy Similarity_Matrices\InternVideo2_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode ViC_duplicate \
  --use_subs \
  --subtitle_json Subtitles\msrvtt_subtitles.json \
  2>&1 | tee logs/msrvtt_t2v_with_subs.log
```

## ⚙️ Configuration Options

### Text-to-Video (T2V) Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--sim_paths` | List of similarity matrix paths (ensemble mode) - the order matters for ViC_duplicate and ViC_unique | Required | `clip4clip.npy GRAM.npy` |
| `--sim_path` | Single similarity matrix path | Required | `clip4clip.npy` |
| `--csv_path` | Path to CSV with video metadata | Required | `descs_ret_test_msrvtt.csv` |
| `--video_dir` | Directory containing videos | Required | `MSRVTT` |
| `--num_images` | Number of video candidates to rerank | Required | `14` |
| `--model_name` | InternVL model name | Required | `OpenGVLab/InternVL3_5-38B` |
| `--grid_size` | Grid size for frame sampling | Required | `3` (for 3×3 grid) |
| `--ensemble_mode` | Ensemble strategy | `ViC_duplicate` |  `ViC_duplicate`, `ViC_unique`, `none` |
| `--use_subs` | Enable subtitle usage | `False` | flag |
| `--subtitle_json` | Path to subtitle JSON | `None` | `msrvtt_subtitles.json` |

### Video-to-Text (V2T) Parameters

Same as T2V, except:
- Use `--num_captions` instead of `--num_images`


## Baseline Usage


### Text-to-Video Retrieval (CombSum)

```bash
python wrapper_baseline.py \
  --method combsum \
  --mode t2v \
  --sim_paths Similarity_Matrices\clip4clip_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy Similarity_Matrices\InternVideo2_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --output_path Similarity_Matrices\combsum_fused_msrvtt_t2v.npy \
  --normalization minmax \
  --evaluate
```

## Configuration Options

### Common Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--method` | Ensemble method to use | Required | `combsum`, `combmnz`, `rrf` |
| `--mode` | Retrieval mode | Required | `t2v`, `v2t` |
| `--sim_paths` | List of similarity matrix paths (.npy files) | Required | `clip4clip.npy GRAM.npy` |
| `--csv_path` | Path to CSV file (columns: video_id, sentence) | Required | `descs_ret_test_msrvtt.csv` |
| `--video_dir` | Directory containing video files | Required | `MSRVTT` |
| `--output_path` | Path to save fused similarity matrix | `{method}_fused_{mode}.npy` | `combsum_result.npy` |
| `--evaluate` | Evaluate and print Recall@1/5/10 metrics | `False` | flag |
| `--quiet` | Suppress progress output | `False` | flag |

### CombSum / CombMNZ Parameters

| Parameter | Description | Default | Methods | Example |
|-----------|-------------|---------|---------|---------|
| `--normalization` | Score normalization method | `minmax` | CombSum, CombMNZ | `minmax`, `zscore`, `none` |
| `--weights` | Custom weights for each matrix (must sum to 1.0) | Equal weights | CombSum, CombMNZ | `0.5 0.3 0.2` |
| `--threshold` | Threshold for non-zero detection | `1e-6` | CombMNZ only | `1e-5` |

### RRF Parameters

| Parameter | Description | Default | Methods | Example |
|-----------|-------------|---------|---------|---------|
| `--k` | RRF constant (rank offset parameter) | `60` | RRF only | `100` |


## Using Hard-Coded Versions

For specific model and dataset combinations, hard-coded wrapper scripts are available that don't require command-line arguments.

### VATEX Dataset
```bash
torchrun --nproc_per_node=2 --master_port=29501 Vatex_code\VATEX_Wrapper_InternVL3_5_38B_grid3x3_VAST_t2v.py \
  2>&1 | tee logs/VATEX_InternVL3_5_38B_grid3x3_VAST_t2v.log
```

### Qwen Model (MSRVTT Dataset)
```bash
torchrun --nproc_per_node=2 --master_port=29501 Qwen3_VL_30B_A3B_Instruct_grid_InternVideo2_msrvtt\Wrapper_Qwen3_VL_30B_A3B_Instruct_grid3x3_InternVideo2_msrvtt_t2v_30_grids.py \
  2>&1 | tee logs/Qwen3_VL_30B_A3B_Instruct_grid3x3_InternVideo2_msrvtt_t2v_30_grids.log
```

### Gemma Model (MSRVTT Dataset)
```bash
torchrun --nproc_per_node=2 --master_port=29501 Gemma_3_27b_it_grid_InternVideo2_msrvtt_code\Wrapper_Gemma_3_27b_it_grid3x3_InternVideo2_msrvtt_v2t_20_captions.py \
  2>&1 | tee logs/Gemma_3_27b_it_grid3x3_InterVideo2_msrvtt_v2t_20_caption.log
```

**Note**: These scripts have all parameters (paths, model names, grid sizes, etc.) configured internally. Simply run the command without additional arguments. More versions are available inside each folder and can be configured there to use new paths etc.