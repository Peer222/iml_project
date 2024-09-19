#!/usr/bin/env bash


for dataset in "CarsDataset" "CUBDataset" "Cifar10Dataset"; do
  # 5 Shot
  for result_dir in ./results/5way5shot*; do
    python test.py --dataset "$dataset" --result_dir "$result_dir" --mode "test" --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 5 --batch_size 16
  done
  # 1 Shot
  for result_dir in ./results/5way1shot*; do
    python test.py --dataset "$dataset" --result_dir "$result_dir" --mode "test" --image_transform "resize_and_normalize" --num_fewshot_classes 5 --num_fewshot_instances 1 --batch_size 16
  done
done
