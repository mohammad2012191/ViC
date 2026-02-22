# [CVPR 2026] Vote-in-Context (ViC): Turning VLMs into Zero-Shot Rank Fusers
**Authors:** Mohamed Eltahir, Ali Habibullah, Lama Ayash, Tanveer Hussain and Naeemullah Khan.
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.01617-b31b1b)](https://arxiv.org/abs/2511.01617)

</div>

<div align="center">
  <img src="Figures/fig 1.png" width="1000">
  <p><em> Left: R@1 for T2V/V2T on MSR-VTT, DiDeMo, VATEX, and ActivityNet versus strong baselines. Right: Qualitative example
where multi-retriever outputs are fused and re-ranked (ViC) to obtain the final list.</em></p>
</div>


---

##  Highlights
- Joint Content & Consensus Fusion: A unified, training-free framework that turns any VLM into a robust fuser/reranker by jointly encoding content (what it sees) and retriever consensus (metadata like duplicates) in its prompt.
- S-Grid Representation: A compact serialization map that represents entire videos as a single VLM-readable image grid (+ optional subtitles).
- New Zero-Shot SOTA: Achieves massive gains of up to +40 R@1 over SOTA baselines and saturates benchmarks.
- Plug-and-Play: Works out-of-the-box with any first-stage retriever (CLIP4Clip, VAST, GRAM, InternVideo2, etc.).
---

##  Methodology 
<div align="center">
  <img src="Figures/fig 2.png" width="700">
  <p><em> The Vote-in-Context (ViC) framework applied for Text-to-Video (t2v, top) and Video-to-Text (v2t, bottom).</em></p>
</div>

---
## 🎞️ Results

<div align="center">
  <img src="Figures/fig 3.png" width="500">
  <p><em> Efficiency vs. Performance Trade-off. Time per query vs. Avg Recall@1 for t2v retrieval over the benchmarks MSR-VTT, DiDeMo and ActivityNet in zero-shot settings. Marker size represents model parameters. The Pareto frontier highlights optimal trade-offs. Latency is measured on a single NVIDIA A100 80GB GPU, averaged over 50 queries for a 1k video retrieval task.</em></p>
</div>

<div align="center">
  <img src="Figures/fig 4.png" width="700">
  <p><em> (a) Effect of reranker scale (InternVL 3.5, 3×3 grid) on t2v Recall@1. (b) Impact of grid size on t2v performance, using
InternVideo2-6B and InternVL 3.5-38B.</em></p>
</div>

# Guide for ViC

## 🧩 Prerequisites
- Python **3.8+**
- CUDA-compatible GPU (**A100** or better recommended)
- `conda` or `pip` package manager
- At least **40 GB VRAM** for large VLMs

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/mohammad2012191/ViC.git
cd ViC

# Create and activate the environment
conda env create -f environment.yml
conda activate vic-env
```
---
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


## 📝 Citation

If you use ViC in your research, please cite:

```bibtex
@misc{eltahir2025vic,
      title={Vote-in-Context: Turning VLMs into Zero-Shot Rank Fusers}, 
      author={Mohamed Eltahir and Ali Habibullah and Lama Ayash and Tanveer Hussain and Naeemullah Khan},
      year={2025},
      eprint={2511.01617},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.01617}, 
}

```
