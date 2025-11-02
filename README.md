# 🎥 Vote-in-Context (ViC): Turning VLMs into Robust Zero-Shot Ranked-List Fusers


[![arXiv](https://img.shields.io/badge/arXiv-2508.18265-b31b1b)](https://arxiv.org/abs/)


---

> **“Abstract”**
In the retrieval domain, candidates’ fusion from heteroge-
neous retrievers is a long-standing challenge, particularly
for complex, multi-modal data such as videos. While typ-
ical fusion techniques are training-free; they rely solely
on rank or score signals, disregarding candidates’ repre-
sentations. This work introduces Vote-in-Context (ViC), a
generalized, training-free framework that re-thinks list-wise
reranking and fusion as a zero-shot reasoning task for a
Vision-Language Model (VLM). The core insight is to seri-
alize both content evidence and retriever metadata directly
within the VLM’s prompt, allowing the model to adap-
tively weigh retriever consensus against visual–linguistic
content. We demonstrate the generality of this framework
by applying it to the challenging domain of cross-modal
video retrieval. To this end, we introduce the S-Grid, a
compact serialization map that represents each video as
an image grid, optionally paired with subtitles to enable
list-wise reasoning over video candidates. ViC is evalu-
ated both as a single-list reranker, where it dramatically
improves the precision of individual retrievers, and as an
ensemble fuser, where it consistently outperforms strong
baselines like CombSUM. Across video retrieval bench-
marks including ActivityNet and VATEX, the framework
establishes new state-of-the-art zero-shot retrieval perfor-
mance, demonstrating its effectiveness in handling complex
visual and temporal signals alongside text. In zero-shot
settings, ViC achieves Recall@1 scores of 87.1% (t2v) /
89.0% (v2t) on MSR-VTT and 99.6% (v2t) on VATEX, rep-
resenting massive gains of up to +40 Recall@1 over pre-
vious state-of-the-art baselines. We present ViC as a sim-
ple, reproducible, and highly effective recipe for turning
modern VLMs into powerful zero-shot rerankers and fusers.

---

## 🎬 Overview

<div align="center">
  <img src="fig 1.png" width="800">
  <p><em>ViC: List-wise reranking and fusion as a reasoning task for a frozen Vision-Language Model (VLM).</em></p>
</div>

---

## ✨ Highlights

- 🔄 **Unified, Training-Free Framework:**  
  Converts any Vision-Language Model into a **content-aware reranker** and **fusion engine**.

- 🧱 **S-Grid Serialization:**  
  Represents videos as compact image grids with optional subtitles, enabling **multimodal reasoning** at constant cost.

- 📈 **New Zero-Shot SOTA:**  
  +40 Recall@1 improvement on MSR-VTT, VATEX, and ActivityNet benchmarks.

- 🧮 **Two-Stage Generality:**  
  Works as both a **single-list reranker (M=1)** and **multi-retriever fuser (M>1)**.

- ⚙️ **Plug-and-Play Integration:**  
  Compatible with CLIP4Clip, VAST, GRAM, InternVideo2, and other first-stage retrievers.

---


















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

1. **CSV File**: Contains video metadata with columns: `video_id`, `sentence`, found inside Groud_Truth folder
2. **Similarity Matrices**: Pre-computed `.npy` files from retrieval models (e.g., CLIP4Clip, VAST, GRAM, InternVideo2), found inside Similarity_Matrices folder
3. **Video Directory**: Folder containing video files that are in mp4 format, found [here](https://drive.google.com/drive/folders/1HUYL_0UO_006Ux4xqoLoXANQCDH6iMt8)
4. **Subtitle JSON** (optional): JSON file with videos ids and subtitles, found inside Subtitles folder

## ViC Usage Guide

### Text-to-Video Retrieval (T2V)

```bash
torchrun --nproc_per_node=4 Scripts\ensemble_wrapper.py \
  --retrieval_mode t2v \
  --sim_paths Similarity_Matrices\InternVideo2_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy \
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


```bash
torchrun --nproc_per_node=4 Scripts\ensemble_wrapper.py \
  --retrieval_mode t2v \
  --sim_paths Similarity_Matrices\InternVideo2_msrvtt.npy  \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode none \
  2>&1 | tee logs/msrvtt_t2v_single.log
```


## Video-to-Text Retrieval (V2T)


```bash
torchrun --nproc_per_node=4 Scripts\ensemble_wrapper.py \
  --retrieval_mode v2t \
  --sim_paths  Similarity_Matrices\InternVideo2_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy Similarity_Matrices\clip4clip_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_captions 20 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode ViC_duplicate \
  2>&1 | tee logs/msrvtt_v2t_ensemble.log
```


```bash
torchrun --nproc_per_node=4 Scripts\ensemble_wrapper.py \
  --retrieval_mode v2t \
  --sim_paths Similarity_Matrices\InternVideo2_msrvtt.npy Similarity_Matrices\GRAM_msrvtt.npy \
  --csv_path Groud_Truth\descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_captions 20 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode ViC_duplicate \
  --use_subs \
  --subtitle_json Subtitles\msrvtt_subtitles.json \
  2>&1 | tee logs/msrvtt_v2t_with_subs.log
```

## Configuration Parameters

### Required Arguments

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--retrieval_mode` | Task type: `t2v` or `v2t` | `t2v` |
| `--sim_paths` | List of similarity matrix paths | `clip4clip.npy GRAM.npy` |
| `--csv_path` | Path to CSV with metadata | `Groud_Truth\descs_ret_test_msrvtt.csv` |
| `--video_dir` | Directory containing videos | `MSRVTT` |
| `--model_name` | InternVL model variant | `OpenGVLab/InternVL3_5-38B` |
| `--grid_size` | Frame sampling grid size | `3` (for 3×3) |

### Mode-Specific Arguments

| Parameter | Mode | Description | Example |
|-----------|------|-------------|---------|
| `--num_images` | T2V | Number of video candidates | `14` |
| `--num_captions` | V2T | Number of text candidates | `20` |

### Optional Arguments

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--ensemble_mode` | Ensemble strategy | `ViC_duplicate` | `ViC_duplicate`, `ViC_unique`, `none` |
| `--use_subs` | Enable subtitles | `False` | Flag (no value) |
| `--subtitle_json` | Path to subtitle JSON | `None` | `Subtitles\msrvtt_subtitles.json` |

## Notes

- **Order matters**: Similarity matrix paths order matters
- **GPU count**: Adjust `--nproc_per_node` based on available GPUs
- **VATEX**: When working with the VATEX dataset you need to use VATEX_wrapper.py instead.


## Baseline Usage

```bash
python Scripts\wrapper_baseline.py \
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
| `--csv_path` | Path to CSV file (columns: video_id, sentence) | Required | `Groud_Truth\descs_ret_test_msrvtt.csv` |
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
