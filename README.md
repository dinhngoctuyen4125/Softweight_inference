# Soft-Weight Inference for Code API Unlearning

## Installation
```bash
conda create -n sw python=3.10
conda activate sw
pip install -r requirements.txt
```

## Run
```bash
bash eval_weight_codellama.sh
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--ocsvm_pkl` | `./ocsvm/ocsvm_models_full.pkl` | Path to OCSVM pkl file |
| `--forget_data` | `./data/codellama/D_forget.json` | Path to D_forget.json |
| `--dep_data` | `./data/codellama/D_test_U_dep.json` | Path to D_test_U_dep.json |
| `--nondep_data` | `./data/codellama/D_test_U_nondep.json` | Path to D_test_U_nondep.json |
| `--ood_base_model` | `microsoft/codebert-base` | Base encoder model |
| `--num_test_samples` | `100` | Number of random test samples per dataset |
| `--batch_size` | `16` | Batch size for feature extraction |
| `--seed` | `42` | Random seed |
| `--output_file` | `./weight_results_codellama.json` | Output JSON file |
