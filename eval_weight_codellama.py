"""
Soft-weight inference for CodeLlama data using pre-trained OCSVM.

Motivation:
  Không phải input nào cũng liên quan đến deprecated API ở mức độ như nhau.
  Script này tính soft-weight w(x) biểu thị mức độ kích hoạt Unlearning cho từng input.

Pipeline:
  1. Load OCSVM (từ HuggingFace: Hoaikkk/codebert-ocsvm) + CodeBERT encoder
  2. Với mỗi deprecated API trong D_forget:
     a. Cho "probing input" + "y_neg" qua CodeBERT → hidden features
     b. Tính Mahalanobis score → OCSVM score → d_H(x) (khoảng cách đến boundary siêu cầu)
     c. Fit phân phối chuẩn N(μ, σ) trên d_H(x) của D_forget
     d. Tìm d_H^0 = μ (= median của phân phối chuẩn, tại P = 0.5)
  3. Với test input x:
     a. Tính p  = P(d_H(x))   = CDF tại d_H(x)
     b. Tính d'_H(x) = 2*d_H^0 - d_H(x)  (đối xứng qua trung vị)
     c. Tính p' = P(d'_H(x))  = CDF tại điểm đối xứng
     d. |p - p'| đo mức lệch khỏi trung tâm forget distribution
     e. w(x) = σ(C * (1 - |p - p'|) - range_th)   (C=10, range_th=2)
  4. Thống kê weight trên D_test_U_dep (deprecated) vs D_test_U_nondep (non-deprecated)
     Mong muốn: D_dep → w cao, D_nondep → w thấp
"""

import os
import json
import argparse
import random
import pickle
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from huggingface_hub import hf_hub_download
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# ---- Functions reused from run_ood.py (L175-222) and eval_o3.py (L40-51) ----
# Kept inline to avoid heavy module-level side effects in those files.

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def weighting_func_gmm(train_in_score, test_in_score):
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    x0 = (mean1 + mean2) / 2
    return gmm, x0


def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)


def obtain_weights(input_x, gmm, x0):
    cp_x = gmm_cdf(input_x, gmm)
    symmetric_x = 2 * x0 - input_x
    cp_symmetric_x = gmm_cdf(symmetric_x, gmm)
    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    cp_sum *= 10  # scaling_factor
    range_th = 2
    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))
    return w_res

# ---- CodeBERT feature extraction ----

def extract_hidden_features(model, tokenizer, texts, max_seq_length=512, batch_size=16):
    """
    Extract per-layer mean hidden features from CodeBERT.
    Returns: list of 13 tensors, each shape (N, 768).
    """
    num_layers = 13  # 1 embedding + 12 transformer layers
    all_layer_features = [[] for _ in range(num_layers)]

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        inputs = tokenizer(
            batch_texts, padding='max_length', truncation=True,
            max_length=max_seq_length, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of 13 tensors

        for i in range(num_layers):
            layer_mean = torch.mean(hidden_states[i], dim=1, keepdim=False).detach().cpu()
            all_layer_features[i].append(layer_mean)

    for i in range(num_layers):
        all_layer_features[i] = torch.cat(all_layer_features[i], dim=0)

    return all_layer_features


def compute_mah_scores(features, cov_estimators):
    """
    Compute Mahalanobis-based scores using pre-computed covariance estimators.
    features: list of 13 tensors, each shape (N, 768)
    cov_estimators: list of 13 EmpiricalCovariance objects

    Returns: np array shape (N, 12) — layers 1..12 (skip layer 0, matching [:, 1:] in original code)
    """
    num_layers = len(features)
    total_scores = []

    for i in range(num_layers):
        mean = torch.from_numpy(cov_estimators[i].location_).float().to(device)
        precision = torch.from_numpy(cov_estimators[i].precision_).float().to(device)
        fea_normalized = F.normalize(features[i], dim=-1)

        out_features = features[i].to(device)
        zero_f = out_features - mean
        gaussian_score = -0.5 * ((zero_f @ precision) @ zero_f.t()).diag()

        # cosine similarity with normalized features (self-referential for single batch)
        cs_score = fea_normalized.to(device) @ fea_normalized.t().to(device)
        cs_score = torch.max(cs_score, dim=1)[0]

        all_score = -cs_score * 1000. + gaussian_score
        total_scores.append(all_score.cpu().numpy())

    # Stack and skip layer 0 (matching [:, 1:] in original code)
    total_scores = np.stack(total_scores, axis=1)
    return total_scores[:, 1:]


def compute_mah_scores_with_ref(features, cov_estimators, ref_features):
    """
    Compute Mahalanobis-based scores using pre-computed covariance estimators
    and reference features (from D_forget) for cosine similarity.
    features: list of 13 tensors, each shape (N, 768)  — test samples
    cov_estimators: list of 13 EmpiricalCovariance objects
    ref_features: list of 13 tensors — from D_forget (for cs_score)

    Returns: np array shape (N, 12) — layers 1..12
    """
    num_layers = len(features)
    total_scores = []

    for i in range(num_layers):
        mean = torch.from_numpy(cov_estimators[i].location_).float().to(device)
        precision = torch.from_numpy(cov_estimators[i].precision_).float().to(device)
        ref_fea_normalized = F.normalize(ref_features[i], dim=-1).to(device)

        out_features = features[i].to(device)
        zero_f = out_features - mean
        gaussian_score = -0.5 * ((zero_f @ precision) @ zero_f.t()).diag()

        out_fea_normalized = F.normalize(out_features, dim=-1)
        cs_score = out_fea_normalized @ ref_fea_normalized.t()
        cs_score = torch.max(cs_score, dim=1)[0]

        all_score = -cs_score * 1000. + gaussian_score
        total_scores.append(all_score.cpu().numpy())

    total_scores = np.stack(total_scores, axis=1)
    return total_scores[:, 1:]


def build_text(sample, mode="probing"):
    """
    Build input text from a sample.
    mode="probing": "probing input" + "y_neg" (direct concat)
    mode="function": "function" field only
    """
    if mode == "probing":
        return sample["probing input"] + sample["y_neg"]
    else:
        return sample["function"]


def get_api_key(sample):
    """Extract the deprecated API name from a sample."""
    api_val = sample.get("deprecated api", "")
    if isinstance(api_val, list):
        return api_val[0] if api_val else ""
    return str(api_val)


def print_weight_stats(weights, label):
    """Print statistics about a list of weights."""
    w_arr = np.array(weights)
    print(f"\n{'='*60}")
    print(f"  Weight Statistics: {label}")
    print(f"{'='*60}")
    print(f"  Count:    {len(w_arr)}")
    print(f"  Mean:     {np.mean(w_arr):.6f}")
    print(f"  Std:      {np.std(w_arr):.6f}")
    print(f"  Min:      {np.min(w_arr):.6f}")
    print(f"  Max:      {np.max(w_arr):.6f}")
    print(f"  Median:   {np.median(w_arr):.6f}")

    # Distribution histogram
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, float('inf')]
    hist, _ = np.histogram(w_arr, bins=bins)
    print(f"\n  Distribution:")
    for j in range(len(hist)):
        lo = bins[j]
        hi = bins[j + 1]
        bar = '#' * hist[j]
        print(f"    [{lo:.1f}, {hi:.1f}): {hist[j]:4d}  {bar}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Soft-weight inference for CodeLlama data')
    parser.add_argument('--ocsvm_repo', type=str,
                        default="Hoaikkk/codebert-ocsvm",
                        help='HuggingFace repo containing OCSVM pkl')
    parser.add_argument('--ocsvm_filename', type=str,
                        default="ocsvm_models_full.pkl",
                        help='Filename of OCSVM pkl in the repo')
    parser.add_argument('--forget_data', type=str,
                        default="./data/codellama/D_forget.json",
                        help='Path to D_forget.json')
    parser.add_argument('--dep_data', type=str,
                        default="./data/codellama/D_test_U_dep.json",
                        help='Path to D_test_U_dep.json')
    parser.add_argument('--nondep_data', type=str,
                        default="./data/codellama/D_test_U_nondep.json",
                        help='Path to D_test_U_nondep.json')
    parser.add_argument('--ood_base_model', type=str,
                        default="tummitum/codebert-deprecated",
                        help='Base model for OOD encoder')
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help='Number of random test samples per dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for feature extraction')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max sequence length for tokenizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_file', type=str,
                        default="./weight_results_codellama.json",
                        help='Output JSON file for results')
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- 1. Load OCSVM models from HuggingFace ----
    print("=" * 60)
    print(f"  Step 1: Loading OCSVM from HuggingFace ({args.ocsvm_repo})")
    print("=" * 60)
    ocsvm_pkl_path = hf_hub_download(
        repo_id=args.ocsvm_repo,
        filename=args.ocsvm_filename,
    )
    with open(ocsvm_pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    cov_estimators = pkl_data["cov_estimators"]  # list of 13 EmpiricalCovariance (mean + precision per layer)
    ocsvm_models = pkl_data["ocsvm_models"]      # dict: api_name -> {'model': OneClassSVM, 'num_samples': int}
    print(f"  Loaded {len(cov_estimators)} covariance estimators (layers)")
    print(f"  Loaded {len(ocsvm_models)} OCSVM models (per deprecated API)")

    # ---- 2. Load CodeBERT ----
    print("\n" + "=" * 60)
    print("  Step 2: Loading CodeBERT model")
    print("=" * 60)
    tokenizer = RobertaTokenizer.from_pretrained(args.ood_base_model)
    model = RobertaModel.from_pretrained(args.ood_base_model, output_hidden_states=True)
    model.to(device)
    model.eval()
    print(f"  Loaded {args.ood_base_model} on {device}")

    # ---- 3. Load D_forget and compute reference features + GMM ----
    print("\n" + "=" * 60)
    print("  Step 3: Processing D_forget for GMM fitting")
    print("=" * 60)
    with open(args.forget_data, encoding='utf-8') as f:
        d_forget = json.load(f)
    print(f"  Loaded {len(d_forget)} samples from D_forget")

    # Build texts
    forget_texts = [build_text(s, mode="probing") for s in d_forget]

    # Extract features
    forget_features = extract_hidden_features(
        model, tokenizer, forget_texts,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    # Compute Mahalanobis scores for D_forget (self-referential cosine similarity)
    forget_mah_scores = compute_mah_scores(forget_features, cov_estimators)
    print(f"  D_forget Mah scores shape: {forget_mah_scores.shape}")

    # Group scores by deprecated API and compute OCSVM scores + fit GMM per API
    api_gmm_data = {}  # api -> {'gmm': gmm, 'x0': x0, 'train_scores': array}
    api_to_indices = {}
    for idx, sample in enumerate(d_forget):
        api = get_api_key(sample)
        if api not in api_to_indices:
            api_to_indices[api] = []
        api_to_indices[api].append(idx)

    print(f"\n  Fitting GMM for each deprecated API...")
    for api, indices in tqdm(api_to_indices.items(), desc="Fitting GMMs"):
        if api not in ocsvm_models:
            continue

        ocsvm_model = ocsvm_models[api]["model"]
        api_mah = forget_mah_scores[indices]
        api_scores = ocsvm_model.score_samples(api_mah)

        if len(api_scores) < 2:
            continue

        gmm, x0 = weighting_func_gmm(api_scores, api_scores)
        api_gmm_data[api] = {
            'gmm': gmm,
            'x0': x0,
            'train_scores': train_scores,
        }

    print(f"  Successfully fitted GMM for {len(api_gmm_data)} APIs")

    # ---- 4. Test on D_test_U_dep (100 random samples) ----
    print("\n" + "=" * 60)
    print("  Step 4: Computing weights on D_test_U_dep")
    print("=" * 60)
    with open(args.dep_data, encoding='utf-8') as f:
        d_dep = json.load(f)
    print(f"  Loaded {len(d_dep)} samples from D_test_U_dep")

    # Filter to samples whose API exists in our OCSVM models
    d_dep_valid = [s for s in d_dep if get_api_key(s) in api_gmm_data]
    print(f"  {len(d_dep_valid)} samples have matching OCSVM+GMM models")

    n_dep = min(args.num_test_samples, len(d_dep_valid))
    dep_samples = random.sample(d_dep_valid, n_dep)
    dep_texts = [build_text(s, mode="probing") for s in dep_samples]

    dep_features = extract_hidden_features(
        model, tokenizer, dep_texts,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    dep_mah_scores = compute_mah_scores_with_ref(dep_features, cov_estimators, forget_features)

    dep_weights = []
    for i, sample in enumerate(dep_samples):
        api = get_api_key(sample)
        ocsvm_model = ocsvm_models[api]["model"]
        score = ocsvm_model.score_samples(dep_mah_scores[i:i+1])
        w = obtain_weights(score[0], api_gmm_data[api]['gmm'], api_gmm_data[api]['x0'])
        dep_weights.append(w)

    print_weight_stats(dep_weights, "D_test_U_dep (deprecated API samples)")

    # ---- 5. Test on D_test_U_nondep (100 random samples) ----
    print("=" * 60)
    print("  Step 5: Computing weights on D_test_U_nondep")
    print("=" * 60)
    with open(args.nondep_data, encoding='utf-8') as f:
        d_nondep = json.load(f)
    print(f"  Loaded {len(d_nondep)} samples from D_test_U_nondep")

    # Filter to samples whose API exists in our OCSVM models
    d_nondep_valid = [s for s in d_nondep if get_api_key(s) in api_gmm_data]
    print(f"  {len(d_nondep_valid)} samples have matching OCSVM+GMM models")

    n_nondep = min(args.num_test_samples, len(d_nondep_valid))
    nondep_samples = random.sample(d_nondep_valid, n_nondep)
    nondep_texts = [build_text(s, mode="function") for s in nondep_samples]

    nondep_features = extract_hidden_features(
        model, tokenizer, nondep_texts,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    nondep_mah_scores = compute_mah_scores_with_ref(nondep_features, cov_estimators, forget_features)

    nondep_weights = []
    for i, sample in enumerate(nondep_samples):
        api = get_api_key(sample)
        ocsvm_model = ocsvm_models[api]["model"]
        score = ocsvm_model.score_samples(nondep_mah_scores[i:i+1])
        w = obtain_weights(score[0], api_gmm_data[api]['gmm'], api_gmm_data[api]['x0'])
        nondep_weights.append(w)

    print_weight_stats(nondep_weights, "D_test_U_nondep (new API samples)")

    # ---- 6. Summary comparison ----
    print("\n" + "=" * 60)
    print("  SUMMARY COMPARISON")
    print("=" * 60)
    print(f"  D_test_U_dep   (deprecated API):  mean={np.mean(dep_weights):.6f}, "
          f"min={np.min(dep_weights):.6f}, max={np.max(dep_weights):.6f}")
    print(f"  D_test_U_nondep (new API):         mean={np.mean(nondep_weights):.6f}, "
          f"min={np.min(nondep_weights):.6f}, max={np.max(nondep_weights):.6f}")
    print("=" * 60)

    # ---- 7. Save results ----
    results = {
        "seed": args.seed,
        "num_test_samples": args.num_test_samples,
        "D_test_U_dep": {
            "count": len(dep_weights),
            "mean": float(np.mean(dep_weights)),
            "std": float(np.std(dep_weights)),
            "min": float(np.min(dep_weights)),
            "max": float(np.max(dep_weights)),
            "median": float(np.median(dep_weights)),
            "weights": [float(w) for w in dep_weights],
        },
        "D_test_U_nondep": {
            "count": len(nondep_weights),
            "mean": float(np.mean(nondep_weights)),
            "std": float(np.std(nondep_weights)),
            "min": float(np.min(nondep_weights)),
            "max": float(np.max(nondep_weights)),
            "median": float(np.median(nondep_weights)),
            "weights": [float(w) for w in nondep_weights],
        },
    }

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
