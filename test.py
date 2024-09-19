import argparse
from pathlib import Path
import json

from src.evaluation import evaluate
from src.util import get_device, load_classes, load_model_state

device = get_device()
print(f"{device=}")


def test(args: argparse.Namespace):
    """Evaluate model

    Args:
        args (argparse.Namespace): command line arguments
    """
    embedding_module, relation_module, train_dataset, val_dataset, test_dataset = (
        load_classes(args)
    )
    match args.mode:
        case "val":
            eval_dataset = val_dataset
        case "test":
            eval_dataset = test_dataset
        case _:
            raise NotImplementedError(f'mode "{args.mode}" is not implemented')

    with open(args.result_dir / "args.json", "r") as f:
        train_args = json.load(f)

    relation_module = load_model_state(args.result_dir, relation_module)
    embedding_module.to(device)
    relation_module.to(device)

    evaluation_dict = evaluate(
        embedding_module,
        relation_module,
        eval_dataset,
        args.num_iterations,
        args.batch_size,
        train_args["xi_"],
        train_args["lambda_"],
    )

    # Here we log our results
    eval_dir = (
        args.result_dir
        / f"eval_{args.dataset}_{args.mode}_{args.num_fewshot_classes}way_{args.num_fewshot_instances}shot"
    )
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True)

    with open(eval_dir / "log.json", "w") as f:
        json.dump(evaluation_dict, f)

    with open(eval_dir / "args.json", "w") as f:
        args.result_dir = str(args.result_dir)
        json.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")

    # model args
    parser.add_argument(
        "--model",
        type=str,
        default="RelationNetFewShot",
        choices=["RelationNet", "RelationNetFewShot"],
        help="Model that should be tested.",
    )
    parser.add_argument(
        "--num_hidden_channels",
        type=int,
        default=512,
        help="Number of hidden channels of the convolutional layers in the relation net/module. (64 is standard for DefaultEmbeddings and ResNet TODO)",
    )
    parser.add_argument(
        "--embedding_module",
        type=str,
        default="ResNetEmbeddingModule",
        choices=["ResNetEmbeddingModule", "DefaultEmbeddingModule"],
        help="Module for generation embeddings for the relation net head. (ResNet is pretrained on ImageNet)",
    )

    # data args
    parser.add_argument(
        "--dataset",
        type=str,
        default="MiniImageNetDataset",
        choices=["MiniImageNetDataset", "CarsDataset", "CUBDataset", "Cifar10Dataset"],
        help="Dataset that should be used for evaluation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Used mode of the evaluation dataset.",
    )
    parser.add_argument(
        "--image_transform",
        type=str,
        default="resize_and_normalize",
        choices=["no_transform", "resize_and_normalize"],
        help="torchvision Transforms that should be applied on input images.",
    )
    parser.add_argument(
        "--num_fewshot_classes", type=int, default=5, help="Number of few shot classes."
    )
    parser.add_argument(
        "--num_fewshot_instances",
        type=int,
        default=5,
        help="Number of few shot training instances per class.",
    )

    # evaluation args
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2000,
        help="Number of evaluation iterations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of query images per support set.",
    )

    # util args
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Directory to load model state from and to save evaluation results.",
    )

    args = parser.parse_args()

    test(args)
