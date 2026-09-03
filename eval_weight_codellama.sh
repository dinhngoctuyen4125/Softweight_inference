#!/bin/bash
# Soft-weight inference for CodeLlama data

python eval_weight_codellama.py \
    --ocsvm_repo "Hoaikkk/codebert-ocsvm" \
    --forget_data "../Data-Collection/codellama/D_forget.json" \
    --dep_data "../Data-Collection/codellama/D_test_U_dep.json" \
    --nondep_data "../Data-Collection/codellama/D_test_U_nondep.json" \
    --ood_base_model "tummitum/codebert-deprecated" \
    --num_test_samples 100 \
    --batch_size 32 \
    --seed 42 \
    --output_file "./weight_results_codellama.json"
