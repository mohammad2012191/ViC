
## 📁 Project Structure

```
S-Grid-Video-Reranker/
├── .archive_code/              # Legacy code versions
├── Dataset_fixers/             # Dataset preprocessing utilities
├── Gemma_3_27b_it_grid_InternVideo2_msrvtt_code/
├── Groud_Truth/                # Ground truth annotations
├── Paper_logs/                 # Experiment logs
├── Qwen3_VL_30B_A3B_Instruct_grid_InternVideo2_msrvtt/
├── Similarity_Matrices/        # Pre-computed similarity matrices
├── Subtitles/                  # Video subtitle files
├── Vatex_code/                 # VATEX dataset specific code
├── Msrvtt_activitynet_didemo_code/
│   ├── ensemble_msrvtt_activitynet_didemo_t2v.py        # Ensemble T2V inference
│   ├── ensemble_msrvtt_activitynet_didemo_v2t.py        # Ensemble V2T inference
│   ├── msrvtt_activitynet_didemo_t2v.py                 # Single-model T2V inference
│   ├── msrvtt_activitynet_didemo_v2t.py                 # Single-model V2T inference
│   ├── ensemble_wrapper_msrvtt_activitynet_didemo_t2v.py  # Ensemble T2V wrapper
│   ├── ensemble_wrapper_msrvtt_activitynet_didemo_v2t.py  # Ensemble V2T wrapper
│   ├── wrapper_msrvtt_activitynet_didemo_t2v.py           # Single-model T2V wrapper
│   └── wrapper_msrvtt_activitynet_didemo_v2t.py           # Single-model V2T wrapper
│
├── How_to_use.ipynb           # Usage examples
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for distributed training)
    - 1 A100 is enough
- conda or pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/mohammad2012191/S-Grid-Video-Reranker.git
cd S-Grid-Video-Reranker

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate s-grid-env

# Option 2: Using pip
pip install -r requirements.txt # Not recommended, I don't even know if it works.
```


### Required Data

1. **CSV File**: Contains video metadata with columns: `key`, `vid_key`, `video_id`, `sentence`
2. **Similarity Matrices**: Pre-computed `.npy` files from retrieval models (e.g., CLIP4Clip, GRAM, InternVideo2)
3. **Video Directory**: Folder containing video files all are mp4 format
    > If you want access to the encoded mp4 datasets please raise a request and the kaggle access will be given
4. **Subtitle JSON** (optional): JSON file with video subtitles

## 📖 Usage

### Text-to-Video Retrieval (Ensemble Mode)

```bash
torchrun --nproc_per_node=3 ensemble_wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_paths clip4clip_msrvtt.npy GRAM_msrvtt_tvas.npy InternVideo2_msrvtt.npy \
  --csv_path descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode hard_dup \
  2>&1 | tee logs/msrvtt_t2v_ensemble.log
```

### Video-to-Text Retrieval (Ensemble Mode)

```bash
torchrun --nproc_per_node=3 ensemble_wrapper_msrvtt_activitynet_didemo_v2t.py \
  --sim_paths clip4clip_msrvtt.npy GRAM_msrvtt_tvas.npy InternVideo2_msrvtt.npy \
  --csv_path descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_captions 20 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode hard_dup \
  2>&1 | tee logs/msrvtt_v2t_ensemble.log
```

### Single Similarity Matrix Mode

For single similarity matrix (no ensemble):

```bash
torchrun --nproc_per_node=3 wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_path clip4clip_msrvtt.npy \
  --csv_path descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  2>&1 | tee logs/msrvtt_t2v_single.log
```

### With Subtitles

```bash
torchrun --nproc_per_node=3 ensemble_wrapper_msrvtt_activitynet_didemo_t2v.py \
  --sim_paths clip4clip_msrvtt.npy GRAM_msrvtt_tvas.npy InternVideo2_msrvtt.npy \
  --csv_path descs_ret_test_msrvtt.csv \
  --video_dir MSRVTT \
  --num_images 14 \
  --model_name OpenGVLab/InternVL3_5-38B \
  --grid_size 3 \
  --ensemble_mode hard_dup \
  --use_subs \
  --subtitle_json msrvtt_subtitles.json \
  2>&1 | tee logs/msrvtt_t2v_with_subs.log
```

## ⚙️ Configuration Options

### Text-to-Video (T2V) Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--sim_paths` | List of similarity matrix paths (ensemble mode) - the order matters for hard_dup and hard_unique | Required | `clip4clip.npy GRAM.npy` |
| `--sim_path` | Single similarity matrix path | Required | `clip4clip.npy` |
| `--csv_path` | Path to CSV with video metadata | Required | `descs_ret_test_msrvtt.csv` |
| `--video_dir` | Directory containing videos | Required | `MSRVTT` |
| `--num_images` | Number of video candidates to rerank | Required | `14` |
| `--model_name` | InternVL model name | Required | `OpenGVLab/InternVL3_5-38B` |
| `--grid_size` | Grid size for frame sampling | Required | `3` (for 3×3 grid) |
| `--ensemble_mode` | Ensemble strategy | `hard_dup` | `soft`, `hard_dup`, `hard_unique`, `none` |
| `--use_subs` | Enable subtitle usage | `False` | flag |
| `--subtitle_json` | Path to subtitle JSON | `None` | `msrvtt_subtitles.json` |

### Video-to-Text (V2T) Parameters

Same as T2V, except:
- Use `--num_captions` instead of `--num_images` (typically 20)

## 🔧 Ensemble Modes

The system supports multiple ensemble strategies for combining similarity matrices:

### 1. **Soft Ensemble** (`soft`)
- Normalizes each similarity matrix row-wise using min-max scaling
- Computes weighted sum of normalized scores
- Returns top-k candidates based on fused scores

### 2. **Hard Duplicate** (`hard_dup`)
- Takes top-k candidates from each similarity matrix
- Performs round-robin selection (allows duplicates)
    - We found this gives the best results in general.

### 3. **Hard Unique** (`hard_unique`)
- Similar to hard_dup but ensures unique candidates
- Falls back to soft voting if insufficient unique candidates

### 4. **None** (`none`)
- Uses only the first similarity matrix
- No ensemble computation

## 📊 Evaluation Metrics

The system computes standard retrieval metrics:
- **Recall@1**: Percentage of queries where the correct item is ranked first
- **Recall@5**: Percentage of queries where the correct item is in top-5
- **Recall@10**: Percentage of queries where the correct item is in top-10

Example output:
```
==================== Final Results ====================
Recall@1:  0.4523
Recall@5:  0.7834
Recall@10: 0.8912
Total processed: 1000/1000
```

## 🗂️ Dataset

###  Datasets That we worked with 
1. **MSR-VTT**
2. **DiDeMo**
3. **ActivityNet**
4. **VATEX**

> The same scripts should work on any other dataset as long as you follow the same stcture, for example MSVD

### Note on VATEX
VATEX dataset requires different data loading mechanism and is handled separately in the `Vatex_code/` directory. You can find how to use this in the How to use notebook.

## 🔬 Experiments

See Paper_logs for comprehensive experiments including:
- Different ensemble modes
- Various grid sizes
- Model size comparisons (8B, 14B, 38B)
- With/without subtitles
- Different similarity matrix combinations

## 💾 Model Cache

Models are cached in `./models` by default. Update the `cache_dir` parameter in the code to change the location.

