#!/bin/bash

set -ev

# python3 scripts/run_training.py --data data --input-model weights/for_larger_data_bckp/step_0_StripClasses0.pt --output-model weights/final_models/retrained.pt --epochs 20 --learning-rate 0.001 --batch-size 128 --device cuda 
#
# sleep 10
# python3 scripts/run_training.py --data data --input-model weights/final_models/retrained.pt --output-model weights/final_models/finetuned.pt --epochs 5 --learning-rate 0.0001 --batch-size 128 --device cuda 
#
sleep 10
python3 scripts/run_pruning.py --data data --input-model weights/final_models/finetuned.pt --output-model weights/final_models/pruned_0_6_ap.pt --target-ap 0.6 --device cuda --train-batch-size 64 --learning-rate 0.0001 --prune-ratio 0.2
sleep 10
python3 scripts/run_operator_fusion.py --data data --input-model weights/final_models/pruned_0_6_ap.pt --output-model weights/final_models/fused_0_6_ap.pt

sleep 10
python3 scripts/run_pruning.py --data data --input-model weights/final_models/pruned_0_6_ap.pt --output-model weights/final_models/pruned_0_5_ap.pt --target-ap 0.5 --device cuda --train-batch-size 64 --learning-rate 0.0001 --prune-ratio 0.2
sleep 10
python3 scripts/run_operator_fusion.py --data data --input-model weights/final_models/pruned_0_5_ap.pt --output-model weights/final_models/fused_0_5_ap.pt

sleep 10
python3 scripts/run_pruning.py --data data --input-model weights/final_models/pruned_0_5_ap.pt --output-model weights/final_models/pruned_0_4_ap.pt --target-ap 0.4 --device cuda --train-batch-size 64 --learning-rate 0.0001 --prune-ratio 0.2
sleep 10
python3 scripts/run_operator_fusion.py --data data --input-model weights/final_models/pruned_0_4_ap.pt --output-model weights/final_models/fused_0_4_ap.pt

sleep 10
python3 scripts/run_pruning.py --data data --input-model weights/final_models/pruned_0_4_ap.pt --output-model weights/final_models/pruned_0_3_ap.pt --target-ap 0.3 --device cuda --train-batch-size 64 --learning-rate 0.0001 --prune-ratio 0.2
sleep 10
python3 scripts/run_operator_fusion.py --data data --input-model weights/final_models/pruned_0_3_ap.pt --output-model weights/final_models/fused_0_3_ap.pt
