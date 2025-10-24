# my_worker.py

import os
import re
import cv2
import torch
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
import flash_attn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import bitsandbytes as bnb


warnings.filterwarnings("ignore", message="The use of `x.T` on tensors")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

#####################################################
# CONFIG + CONSTANTS
#####################################################
class Config:
    num_captions = 20  # how many text candidates we select from the similarity row
    video_dir = "MSRVTT"
    ensemble_mode = "hard_dup"      # "soft" | "hard_dup" | "hard_unique" | None
    ensemble_weights = None     # e.g., [0.5, 0.3, 0.2]; same len as # of sim_mats
    per_model_topk = 4          # k picked from each system when using hard_* modes


#####################################################
# VLM Worker
#####################################################
class VLMWorker:
    def __init__(self, gpu_id):
        """Initialize the InternVL2 model+tokenizer on a specific GPU."""
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)

        # Model name
        model_name = "OpenGVLab/InternVL3_5-38B"
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir="./models")
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map=f"cuda:{gpu_id}",
            cache_dir="./models"
        ).eval()

        # Build transform once
        self.transform = self._build_transform()


        # self.subs = self._load_subtitles("msrvtt_subtitles.json")
        
        
    def _load_subtitles(self, path):
        import json
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # list[dict{video_id, subtitle}] -> {video_id: subtitle}
        out = {}
        for item in data:
            vid = str(item.get("video_id", "")).strip()
            sub = str(item.get("subtitle", "")).strip()
            if vid:
                out[vid] = sub
        return out
        
    @staticmethod
    def _build_transform(input_size=448):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    #####################################################
    # Image Preprocessing
    #####################################################

    def get_first_frame(self, video_path, grid_size=3):
        """
        Sample grid_size^2 frames uniformly from the given video.
        Each sub-frame is resized to floor(448/grid_size) x floor(448/grid_size).
        Then we paste them into a 448x448 final image (any leftover area remains blank).
    
        :param video_path:  Path to the video file.
        :param grid_size:   2 => (2x2), 3 => (3x3), 4 => (4x4), etc.
                           The final image is always 448x448.
        :return: A PIL Image of size 448x448.
        """
        import numpy as np
    
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        # We'll sample exactly grid_size^2 frames
        n_frames = grid_size * grid_size
    
        # Basic check to ensure the video has enough frames
        if total_frames < n_frames:
            raise ValueError(
                f"Video {video_path} has fewer than {n_frames} frames "
                f"(total: {total_frames}), cannot make a {grid_size}x{grid_size} grid."
            )
    
        # Compute uniform frame indices
        # e.g., if grid_size=2 -> 4 frames across the length of the video
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                raise ValueError(f"Could not read frame {idx} from {video_path}")
            frames.append(frame)
    
        cap.release()
    
        # Each sub-frame is sub_size x sub_size
        sub_size = 448 // grid_size  # e.g. 224 if grid=2, 149 if grid=3, 112 if grid=4, etc.
    
        # Convert frames to PIL, resize
        pil_frames = []
        for frame in frames:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize((sub_size, sub_size), resample=Image.BICUBIC)
            pil_frames.append(pil_img)
    
        # Create the final 448x448 image
        final_image = Image.new("RGB", (448, 448))
    
        # Paste each sub-image in row-major order
        for i, pil_img in enumerate(pil_frames):
            row = i // grid_size
            col = i % grid_size
            x_offset = col * sub_size
            y_offset = row * sub_size
            final_image.paste(pil_img, (x_offset, y_offset))
    
        return final_image


    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """
        Splits an image into multiple patches for the VLM (InternVL2).
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # all possible aspect ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                # tie-breaker
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def load_image_intern(self, image, input_size=448, max_num=6):
        """
        Convert single PIL image -> stacked patches for VLM.
        """
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    #####################################################
    # Reranking for one query
    #####################################################
    def rerank_top_captions(self, query_patches, text_candidates, query_video_id):
        """
        Given a single *video query* (query_patches) and a list of text candidates,
        pass them to model.chat for re-ranking.
        Return a new ranking: e.g. [2,1,4,3,...] meaning text #2 is most relevant, etc.
        """
        num_captions = len(text_candidates)

        prompt_lines = []
        
        # Add the query video
        prompt_lines.append("Query-Video: <image>")
        
        
        # Add text candidates
        for i, text_str in enumerate(text_candidates, start=1):
            prompt_lines.append(f"Candidate-{i}: {text_str}")
    


        prompt_lines.append(
            f"Given the user query (above video) and the {num_captions} text candidates, "
            f"rank these candidates from most to least relevant to the video. "
            f"Output a list like [1,3,2,...,{num_captions}] (just an example). "
            f"Output only the {num_captions}-ranked-elements list instantly."
        )
    
        final_prompt = "\n".join(prompt_lines)
    
        # We feed the video patches as pixel_values, and the prompt that enumerates the text
        # The model needs to interpret <image> as the "query" and the text lines as candidates.
        generation_config = dict(max_new_tokens=256, do_sample=False)
        try:
            response = self.model.chat(
                self.tokenizer,
                query_patches.to(f"cuda:{self.gpu_id}"),
                final_prompt,
                generation_config,
                num_patches_list=[query_patches.size(0)]  # single "image" with multiple patches
            )
        except Exception as e:
            print(f"[GPU {self.gpu_id}] Error in rerank_top_captions: {e}")
            # fallback
            response = "[" + ",".join(str(i) for i in range(1, num_captions + 1)) + "]"
    
        # parse the new ranking
        return self.parse_ranking_from_response(response, num_captions)



    def parse_ranking_from_response(self, response_text, num_items):
        """
        E.g. parse "[1,3,2,5,4]".
        Fallback if not found or invalid = [1..num_items].
        """
        match = re.search(r'\[(.*?)\]', response_text)
        if match:
            content = match.group(1)
            try:
                # e.g. "1,3,2,5,4"
                ranks = [int(x.strip()) for x in content.split(',')]
                                
                # Check if all values are in valid range [1, num_items]
                if not all(1 <= r <= num_items for r in ranks):
                    print(f"[GPU {self.gpu_id}] Invalid ranking values: {ranks}. Expected values in range [1, {num_items}]. Using original order.")
                    return list(range(1, num_items + 1))                
                
                return ranks
            except (ValueError, AttributeError) as e:
                print(f"[GPU {self.gpu_id}] Error parsing ranking: {e}. Using original order.")
                return list(range(1, num_items + 1))
        
        print(f"[GPU {self.gpu_id}] No ranking found in response. Using original order.")
        return list(range(1, num_items + 1))


    #####################################################
    # Ensmebling Utils
    #####################################################   
    def _row_minmax(self, x):
        # safe row-wise min-max to [0,1], fallback to zeros if degenerate
        mn, mx = x.min(), x.max()
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)
        
    def _select_candidates_from_ensembles(self, row_or_col_idx, sim_mats, top_k, mode="soft", weights=None):
        """
        sim_mats:
          - single np.ndarray of shape (Q, N)  -> we take sim_mats[row_or_col_idx]
          - OR list of arrays, each (Q, N), same shape/order.
        Returns: np.ndarray of candidate indices length top_k (with duplicates possible only if mode='hard_dup').
        """
        # normalize to list
        mats = sim_mats if isinstance(sim_mats, (list, tuple)) else [sim_mats]
        assert len(mats) >= 1, "No similarity matrix given."
        # row scores per system
        rows = [m[row_or_col_idx].astype(np.float32, copy=False) for m in mats]  # each shape (N,)
    
        if len(mats) == 1 or mode in (None, "", "none"):
            return np.argsort(-rows[0])[:top_k]
    
        if mode == "soft":
            # per-system row min-max, then weighted sum
            ws = np.asarray(weights, dtype=np.float32) if weights is not None else np.ones(len(rows), dtype=np.float32)
            ws = ws / (ws.sum() + 1e-8)
            fused = np.zeros_like(rows[0], dtype=np.float32)
            for w, r in zip(ws, rows):
                fused += w * self._row_minmax(r)
            return np.argsort(-fused)[:top_k]
    
        # rank lists per system (descending)
        per_k = getattr(Config, "per_model_topk", 4)
        ranked = [np.argsort(-r)[:max(top_k, per_k)] for r in rows]
    
        if mode == "hard_dup":
            bag = []
            # round-robin concat with duplicates
            for rank_pos in range(max(top_k, per_k)):
                for lst in ranked:
                    if rank_pos < len(lst):
                        bag.append(int(lst[rank_pos]))
                        if len(bag) >= top_k:
                            return np.asarray(bag[:top_k], dtype=np.int64)
            # pad if somehow short
            flat = np.concatenate(ranked, axis=0)
            return np.asarray(flat[:top_k], dtype=np.int64)
    
        if mode == "hard_unique":
            chosen, seen = [], set()
            for rank_pos in range(max(top_k, per_k)):
                for lst in ranked:
                    if rank_pos < len(lst):
                        idx = int(lst[rank_pos])
                        if idx not in seen:
                            chosen.append(idx); seen.add(idx)
                            if len(chosen) >= top_k:
                                return np.asarray(chosen, dtype=np.int64)
            # still not enough unique? fill by soft vote fallback
            sv = self._select_candidates_from_ensembles(row_or_col_idx, mats, top_k*2, mode="soft", weights=weights)
            for idx in sv:
                if idx not in seen:
                    chosen.append(int(idx)); seen.add(int(idx))
                if len(chosen) >= top_k:
                    break
            return np.asarray(chosen[:top_k], dtype=np.int64)
    
        raise ValueError(f"Unknown ensemble mode: {mode}")
        
    #####################################################
    # The main loop for data slice
    #####################################################
    def process_data_slice(self, data_slice, df_rows, sim_mats):
        """
        For each item in data_slice (these are row indices for *videos* now):
          1) We get the top K=Config.num_captions from sim_matrix[row_i].
          2) Instead of building video_paths, we build a list of text candidates.
          3) We'll run rerank_top_captions(...) (a new function) passing:
             - the *single video query* as pixel_values
             - the *list of text candidates*
          4) We'll compute recall stats if there's a ground-truth notion,
             e.g. "did we find the correct text among the top re-ranked?"
        """
        n_items = len(data_slice)
        count_recall_1 = 0
        count_recall_5 = 0
        count_recall_10 = 0
    
        for idx in tqdm(data_slice, desc=f"[GPU {self.gpu_id}] Processing"):
            # The user query is the video at row idx
            ground_truth_text = df_rows["sentence"][idx]  # or some ground-truth text
            video_id = df_rows["video_id"][idx]
            video_path = os.path.join(Config.video_dir, f"{video_id}.mp4")
    
            # Load the composite image for the *query video*
            video_query_img = self.get_first_frame(video_path, grid_size=3)
            # Convert to patches for the model
            query_patches = self.load_image_intern(video_query_img, max_num=6).to(torch.bfloat16)

            # ensemble
            top_indices = self._select_candidates_from_ensembles(
                row_or_col_idx=idx,                    # row / column index that corresponds to this video
                sim_mats=sim_mats,               # list-or-array of similarity matrices
                top_k=Config.num_captions,    # how many we want
                mode    = Config.ensemble_mode,
                weights = Config.ensemble_weights,
            )
    
            # build text candidates
            text_candidates = [df_rows["sentence"][j] for j in top_indices]
    
            # Now we call a new function that does the re-ranking of text
            # given the single video query. For example:
            ranking = self.rerank_top_captions(query_patches, text_candidates, video_id) 

            # ranking e.g. [1,3,2,4,...], telling us how to reorder text_candidates
            reordered_texts = [text_candidates[r - 1] for r in ranking]
    
            # Suppose we measure how well the ground_truth_text was ranked.
            if ground_truth_text in reordered_texts:
                rankpos = reordered_texts.index(ground_truth_text) + 1
            else:
                rankpos = len(reordered_texts) + 1
    
            if rankpos <= 1:
                count_recall_1 += 1
            if rankpos <= 5:
                count_recall_5 += 1
            if rankpos <= 10:
                count_recall_10 += 1
    
        return count_recall_1, count_recall_5, count_recall_10, n_items



def run_distributed_inference(local_rank, data_slice, df_rows, sim_mats):
    """
    Worker entry point for a given rank:
      1) Creates VLMWorker(local_rank).
      2) For each row index in data_slice, does the above logic.
      3) Returns partial recall counts + total_count.
    """
    worker = VLMWorker(local_rank)
    c1, c5, c10, total = worker.process_data_slice(data_slice, df_rows, sim_mats)
    return c1, c5, c10, total
