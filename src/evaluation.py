from pathlib import Path
import json
from typing import Any
from tqdm import tqdm
import torch
from torch import nn

from .util import load_classes, get_device, calc_accuracy, calc_loss  # type: ignore
from .dataset import FewShotDataset  # type: ignore
from .embedding_module import EmbeddingModule  # type: ignore
from .relation_module import RelationModuleFewShot  # type: ignore


def validate(
    embedding_module: EmbeddingModule,
    relation_module: RelationModuleFewShot,
    test_dataset: FewShotDataset,
    num_iterations: int,
    batch_size: int,
    xi_: float,
    lambda_: float,
) -> dict[str, Any]:
    """For validation during training. Returns averaged losses and accuracies.

    Args:
        embedding_module (EmbeddingModule): Embedding module
        relation_module (RelationModuleFewShot): Relation module
        test_dataset (FewShotDataset): Dataset to use for validation
        num_iterations (int): Number of iterations of the validation run / different support sets
        batch_size (int): Batch size / Number of images per support set
        xi_ (float): Weighting of the default forward pass loss
        lambda_ (float): Weighting of the lrp forward pass loss

    Returns:
        dict[str, Any]: log dictionary with aggregated metrics (mean).
    """
    evaluation_dict = evaluate(
        embedding_module,
        relation_module,
        test_dataset,
        num_iterations,
        batch_size,
        xi_,
        lambda_,
    )
    return {
        "losses": {
            "pred": sum(evaluation_dict["losses"]["pred"])
            / len(evaluation_dict["losses"]["pred"]),
            "pred_lrp": sum(evaluation_dict["losses"]["pred_lrp"])
            / len(evaluation_dict["losses"]["pred_lrp"]),
            "total": sum(evaluation_dict["losses"]["total"])
            / len(evaluation_dict["losses"]["total"]),
        },
        "accuracies": {
            "pred": sum(evaluation_dict["accuracies"]["pred"])
            / len(evaluation_dict["accuracies"]["pred"]),
            "pred_lrp": sum(evaluation_dict["accuracies"]["pred_lrp"])
            / len(evaluation_dict["accuracies"]["pred_lrp"]),
        },
    }


def evaluate(
    embedding_module: EmbeddingModule,
    relation_module: RelationModuleFewShot,
    test_dataset: FewShotDataset,
    num_iterations: int,
    batch_size: int,
    xi_: float,
    lambda_: float,
) -> dict[str, Any]:
    """Runs full evaluation with detailed log.

    Args:
        embedding_module (EmbeddingModule): Embedding module
        relation_module (RelationModuleFewShot): Relation module
        test_dataset (FewShotDataset): Dataset to use for validation
        num_iterations (int): Number of iterations of the validation run / different support sets
        batch_size (int): Batch size / Number of images per support set
        xi_ (float): Weighting of the default forward pass loss
        lambda_ (float): Weighting of the lrp forward pass loss

    Returns:
        dict[str, Any]: log dictionary
    """
    device = get_device()

    embedding_module.eval()
    relation_module.eval()
    ce_loss = nn.CrossEntropyLoss()

    losses = {"pred": [], "pred_lrp": [], "total": [], "step": []}
    accuracies = {"pred": [], "pred_lrp": [], "step": []}

    confusion_matrix = {"step": [], "pred": [], "ground_truth": [], "labels": []}

    for i in tqdm(range(num_iterations), desc="evaluation", leave=False):
        # compute embeddings
        with torch.no_grad():
            supportset, (query_images, query_labels) = test_dataset.sample_batch(
                batch_size
            )

            for key, value in supportset.items():
                supportset[key] = value.to(device)

            # compute embeddings
            supportset_embedding = embedding_module.get_supportset_embeddings(
                supportset
            )
            query_embedding: torch.Tensor = embedding_module(query_images.to(device))

            relation_module.update_supportset(supportset_embedding)
            label_indices = relation_module.get_label_indices(query_labels).to(device)
            pred, pred_lrp = relation_module.forward_pass(
                query_embedding, label_indices
            )

            pred_loss, pred_lrp_loss, loss = calc_loss(
                ce_loss, pred, pred_lrp, label_indices, xi_, lambda_
            )

            # logging
            losses["step"].append(i)
            losses["pred"].append(pred_loss.item())
            losses["pred_lrp"].append(pred_lrp_loss.item())
            losses["total"].append(loss.item())

            accuracies["step"].append(i)
            accuracies["pred"].append(calc_accuracy(pred, label_indices))
            accuracies["pred_lrp"].append(calc_accuracy(pred_lrp, label_indices))

            confusion_matrix["step"].append(i)
            confusion_matrix["pred"].append(torch.argmax(pred, dim=1).tolist())
            confusion_matrix["ground_truth"].append(label_indices.tolist())
            confusion_matrix["labels"].append(relation_module.labels)

    return {
        "losses": losses,
        "accuracies": accuracies,
        "confusion_matrix": confusion_matrix,
    }
