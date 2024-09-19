#!/usr/bin/env bash

# 5 way 5 shot
python train.py --mode "baseline" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 5 --eval_every_n_iters 5000 --xi_ 1 --lambda_ 1 --result_dir "results/5way5shot_baseline"
python train.py --mode "lrp" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 5 --eval_every_n_iters 5000 --xi_ 1 --lambda_ 1 --result_dir "results/5way5shot_lrp_paper"
python train.py --mode "lrp_variation" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 5 --eval_every_n_iters 5000 --xi_ 1 --lambda_ 1 --result_dir "results/5way5shot_lrp_variation"
python train.py --mode "lrp" --no_base_grad --xi_ 0 --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 5 --eval_every_n_iters 5000 --lambda_ 2 --result_dir "results/5way5shot_ablation"
# higher lambda to compensate for lower gradient and losses

# 5 way 1 shot
python train.py --mode "baseline" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 1 --eval_every_n_iters 5000 --xi_ 1 --lambda_ "0.5" --result_dir "results/5way1shot_baseline"
python train.py --mode "lrp" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 1 --eval_every_n_iters 5000 --xi_ 1 --lambda_ "0.5" --result_dir "results/5way1shot_lrp_paper"
python train.py --mode "lrp_variation" --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 1 --eval_every_n_iters 5000 --xi_ 1 --lambda_ "0.5" --result_dir "results/5way1shot_lrp_variation"
python train.py --mode "lrp" --no_base_grad --xi_ 0 --num_iterations 100000 --batch_size 16 --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 1 --eval_every_n_iters 5000 --lambda_ 2 --result_dir "results/5way1shot_ablation"
# higher lambda to compensate for lower gradient and losses
