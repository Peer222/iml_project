import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
import json

import torch
from torch import nn
import numpy as np

from src.evaluation import validate
from src.util import load_classes, get_device, calc_accuracy, calc_loss
from src.relation_module import RelationModuleFewShot

device = get_device()
print(f"{device=}")


def save_training_results(
    relation_module: RelationModuleFewShot,
    train_losses: dict[str, list],
    val_losses: dict[str, list],
    train_accuracies: dict[str, list],
    val_accuracies: dict[str, list],
    best_i: int,
):
    """ Save the results from training

    Args:
        relation_module: model to save model state
        train_losses: loss types and their corresponding losses
        val_losses: loss types and their corresponding losses
        train_accuracies: accuracy types and their corresponding accuracies
        val_accuracies: accuracy types and their corresponding accuracies
        best_i: training step at which the best result was achieved
    """
    torch.save(relation_module.state_dict(), args.result_dir / "relation_module.pth")
    log = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_i": best_i,
    }
    with open(args.result_dir / "log.json", "w") as f:
        json.dump(log, f)


def train(args: argparse.Namespace):
    """ Full train of model

    Args:
        args: commandline arguments, see `if __name__ == '__main__'`
    """
    if not args.result_dir.exists():
        args.result_dir.mkdir(parents=True)

    embedding_module, relation_module, train_dataset, val_dataset, _ = load_classes(
        args
    )

    optimizer = torch.optim.Adam(relation_module.parameters())
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=args.num_iterations
    )

    ce_loss = nn.CrossEntropyLoss()

    embedding_module.eval()
    relation_module.train()
    embedding_module.to(device)
    relation_module.to(device)

    i = 0
    train_losses = {"pred": [], "pred_lrp": [], "total": [], "step": []}
    val_losses = {"pred": [], "pred_lrp": [], "total": [], "step": []}
    train_accuracies = {"pred": [], "pred_lrp": [], "step": []}
    val_accuracies = {"pred": [], "pred_lrp": [], "step": []}
    best_i = 0
    last_validate_loss = np.inf

    # save training config
    with open(args.result_dir / "args.json", "w") as f:
        args.result_dir = str(args.result_dir)

        # adding additional info to log
        args.optimizer = type(optimizer).__name__
        args.lr_scheduler = type(lr_scheduler).__name__
        args.loss_function = type(ce_loss).__name__

        json.dump(vars(args), f)
        args.result_dir = Path(args.result_dir)

    with tqdm(total=args.num_iterations, desc="training") as pbar:
        while i < args.num_iterations:
            supportset, (query_images, query_labels) = train_dataset.sample_batch(
                args.batch_size
            )

            for key, value in supportset.items():
                supportset[key] = value.to(device)

            # compute embeddings
            with torch.no_grad():
                supportset_embedding = embedding_module.get_supportset_embeddings(
                    supportset
                )
                query_embedding: torch.Tensor = embedding_module(
                    query_images.to(device)
                )

            relation_module.update_supportset(supportset_embedding)
            label_indices = relation_module.get_label_indices(query_labels).to(device)
            pred, pred_lrp = relation_module.forward_pass(
                query_embedding, label_indices, use_base_grad=not args.no_base_grad
            )

            pred_loss, pred_lrp_loss, loss = calc_loss(
                ce_loss, pred, pred_lrp, label_indices, args.xi_, args.lambda_
            )

            # step updates
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            i += 1
            pbar.update(1)

            # logging
            train_losses["step"].append(i)
            train_losses["pred"].append(pred_loss.item())
            train_losses["pred_lrp"].append(pred_lrp_loss.item())
            train_losses["total"].append(loss.item())

            train_accuracies["step"].append(i)
            train_accuracies["pred"].append(calc_accuracy(pred, label_indices))
            train_accuracies["pred_lrp"].append(calc_accuracy(pred_lrp, label_indices))

            if i % args.eval_every_n_iters == 0 or i == 1:
                info = validate(
                    embedding_module,
                    relation_module,
                    val_dataset,
                    args.num_val_iterations,
                    args.val_batch_size,
                    args.xi_,
                    args.lambda_,
                )
                relation_module.train()

                val_losses["step"].append(i)
                val_losses["pred"].append(info["losses"]["pred"])
                val_losses["pred_lrp"].append(info["losses"]["pred_lrp"])
                val_losses["total"].append(info["losses"]["total"])
                val_accuracies["step"].append(i)
                val_accuracies["pred"].append(info["accuracies"]["pred"])
                val_accuracies["pred_lrp"].append(info["accuracies"]["pred_lrp"])

                if info["losses"]["total"] < last_validate_loss:
                    last_validate_loss = info["losses"]["total"]
                    best_i = i
                    torch.save(
                        relation_module.state_dict(),
                        args.result_dir / "relation_module_best.pth",
                    )

                # saving results and model
                save_training_results(
                    relation_module,
                    train_losses,
                    val_losses,
                    train_accuracies,
                    val_accuracies,
                    best_i,
                )

            if i >= args.num_iterations:
                break

    # saving results and model
    save_training_results(
        relation_module,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        best_i,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script trains the explanation guided relation net module with the provided arguments"
    )

    # model args
    parser.add_argument(
        "--model",
        type=str,
        default="RelationNetFewShot",
        choices=["RelationNet", "RelationNetFewShot"],
        help="Model that should be trained.",
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
    parser.add_argument(
        "--mode",
        type=str,
        default="lrp",
        choices=["lrp", "baseline", "lrp_variation"],
        help="Set mode to use",
    )
    parser.add_argument(
        "--state_dict_path",
        type=Path,
        default=None,
        help="If provided, model is initialized accordingly",
    )

    # data args
    parser.add_argument(
        "--dataset",
        type=str,
        default="MiniImageNetDataset",
        choices=["MiniImageNetDataset"],
        help="Dataset that should be used for training.",
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

    # training args
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--eval_every_n_iters",
        type=int,
        default=10000,
        help="Evaluation frequency during training.",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.001,
        help="Initial/maximum learning rate of the learning rate scheduler.",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=0.0001,
        help="Final learning rate of the learning rate scheduler. Unused!",
    )  # TODO
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size. Default in paper code: 16",
    )

    parser.add_argument(
        "--no_base_grad",
        action="store_true",
        help="If set, only the gradient from lrp will be used in backpropagation",
    )
    parser.add_argument(
        "--xi_",
        type=float,
        default=1,
        help="Weighting factor for loss of first forward pass. (Paper uses 1 for 5 shot and 1 for 1 shot)",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1,
        help="Weighting factor for LRP loss. (Paper uses 1 for 5 shot and 0.5 for 1 shot)",
    )

    # validation args
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="Batch size to use in validation (Number of images per support set)",
    )
    parser.add_argument(
        "--num_val_iterations",
        type=int,
        default=1000,
        help="Number of iterations in validation run (Number of different support sets)",
    )

    # meta args
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path("./results") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
        help="Directory for the training results (model state_dict, args.json). Default: `[timestamp]`",
    )

    args = parser.parse_args()

    train(args)
