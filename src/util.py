from __future__ import annotations

from pathlib import Path
import argparse
import torch
from torch import Tensor, nn
import torchvision.transforms as T  # type: ignore
from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule, IdentityRule, GammaRule  # type: ignore

import src.relation_module
from .embedding_module import ResNetEmbeddingModule, DefaultEmbeddingModule, EmbeddingModule  # type: ignore
from .dataset import FewShotDataset, MiniImageNetDataset, CarsDataset, CUBDataset, Cifar10Dataset  # type: ignore


def get_device() -> torch.device:
    """Returns most advanced available device from {cpu, mps, cuda}.

    Returns:
        torch.device: device
    """
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    device = torch.device("cuda") if torch.cuda.is_available() else device
    return device


def normalize_lrp_weighting(lrp_weightings: Tensor) -> Tensor:
    """ Normalize lrp weightins
    Divides by max absolute value so that weightings are in [-1, 1] for every channel

    Args:
        lrp_weightings: Tensor of the weightings (including all channels)

    Returns:
        normalized lrp weightings per channel
    """
    max_weighting_per_channel, _ = torch.max(
        torch.abs(lrp_weightings), dim=1, keepdim=True
    )
    # don't divide by zero for values that evaluate to zero anyways
    max_weighting_per_channel[max_weighting_per_channel == 0] = 1
    return lrp_weightings / max_weighting_per_channel


def calc_loss(
    loss_fn: torch.nn.modules.loss._Loss,
    predictions: Tensor,
    lrp_predictions: Tensor,
    label_indices: Tensor,
    xi_: float,
    lambda_: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate losses

    Args:
        loss_fn (torch.nn.modules.loss._Loss): Loss function from torch.nn.
        predictions (Tensor): Predictions from first forward pass.
        lrp_predictions (Tensor): Predictions from second forward pass.
        label_indices (Tensor): Indices of ground truth labels.
        xi_ (float): Weighting factor for loss of first forward pass.
        lambda_ (float): Weighting factor for lrp loss of second forward pass.

    Returns:
        tuple[Tensor, Tensor, Tensor]: prediction loss, lrp prediction loss, total loss
    """
    pred_loss = loss_fn(predictions, label_indices)
    pred_lrp_loss = loss_fn(lrp_predictions, label_indices)
    loss = (
        xi_ * pred_loss + lambda_ * pred_lrp_loss
    )  # TODO check if separate backward passes work better (paper combines losses as well)
    return pred_loss, pred_lrp_loss, loss


def calc_accuracy(predictions: Tensor, gt_label_indices: Tensor) -> float:
    """Calculates the accuracy of a prediction batch.

    Args:
        predictions (torch.Tensor): Predictions/ relation scores of the relation module.
        gt_label_indices (torch.Tensor): Ground truth indices of the labels.

    Returns:
        float: Accuracy of the predictions
    """
    return (predictions.argmax(dim=1) == gt_label_indices).float().mean().item()


def load_classes(
    args: argparse.Namespace,
) -> tuple[
    EmbeddingModule,
    src.RelationModuleFewShot,
    FewShotDataset,
    FewShotDataset,
    FewShotDataset,
]:
    """loads model and dataset classes from provided arguments

    Args:
        args (argparse.Namespace): parsed command line arguments

    Raises:
        NotImplementedError: Raised when class is not present in argument choices.

    Returns:
        tuple[nn.Module, nn.Module, Dataset, Dataset]: embedding_module, relation_module, train_dataset, validation_dataset
    """

    match args.embedding_module:
        case "ResNetEmbeddingModule":
            embedding_module = ResNetEmbeddingModule()
            padding = 1
            conv_out_channels = args.num_hidden_channels
        case "DefaultEmbeddingModule":
            embedding_module = DefaultEmbeddingModule()
            padding = 0
            conv_out_channels = args.num_hidden_channels * 9
        case _:
            raise NotImplementedError(
                "The provided embedding module type is not supported."
            )

    # TODO move to embedding module classes (freeze() member function or in init)
    for param in embedding_module.parameters():
        param.requires_grad = False

    match args.model:
        case "RelationNet":
            # TODO not supported for current training loop
            relation_module = src.RelationModule(
                padding,
                embedding_module.output_channels,
                hidden_channels=args.num_hidden_channels,
                conv_out_channels=conv_out_channels,
            )
        case "RelationNetFewShot":
            relation_module = src.RelationModuleFewShot(
                padding,
                embedding_module.output_channels * 2,
                hidden_channels=args.num_hidden_channels,
                conv_out_channels=conv_out_channels,
            )
        case _:
            raise NotImplementedError("The provided model type is not supported.")

    transform = load_transform(args.image_transform)
    match args.dataset:
        case "MiniImageNetDataset":
            train_dataset = MiniImageNetDataset(
                transform,
                mode="train",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            val_dataset = MiniImageNetDataset(
                transform,
                mode="val",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            test_dataset = MiniImageNetDataset(
                transform,
                mode="test",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
        case "CarsDataset":
            train_dataset = CarsDataset(
                transform,
                mode="train",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            val_dataset = CarsDataset(
                transform,
                mode="val",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            test_dataset = CarsDataset(
                transform,
                mode="test",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
        case "CUBDataset":
            train_dataset = CUBDataset(
                transform,
                mode="train",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            val_dataset = CUBDataset(
                transform,
                mode="val",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            test_dataset = CUBDataset(
                transform,
                mode="test",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
        case "Cifar10Dataset":
            # it is not possible to train on cifar10 due to the low number of classes
            train_dataset = None  # this will not actually be used
            val_dataset = Cifar10Dataset(
                transform,
                mode="val",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
            test_dataset = Cifar10Dataset(
                transform,
                mode="test",
                n_way=args.num_fewshot_classes,
                k_shot=args.num_fewshot_instances,
            )
        case _:
            raise NotImplementedError("The provided dataset type is not supported.")

    return embedding_module, relation_module, train_dataset, val_dataset, test_dataset


def load_transform(transform_name: str) -> T.Compose:
    """Loads torchvision transformations for input image transformations from provided name.

    Args:
        transform_name (str): Name of transformation from: {'no_transform', 'resize_and_normalize'}

    Raises:
        NotImplementedError: Provided name has no matching implemented transformation.

    Returns:
        T.Compose: Image transformations.
    """
    match transform_name:
        case "no_transform":
            transformations = [
                T.ToTensor(),
            ]
        case "resize_and_normalize":
            transformations = [
                T.Lambda(lambda x: x.convert("RGB")),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        case _:
            raise NotImplementedError(
                "The provided transformation name is not supported"
            )

    return T.Compose(transformations)


def set_lrp_rules(module: nn.Module, mode: str = "lrp"):
    """Set lrp propagation rules on modules
    This is needed because captum deletes rules after every usage (bug?)

    Args:
        module: A torch module
    """
    lrp_rules_paper = {
        "Conv2d_LRP": Alpha1_Beta0_Rule(),
        "Linear_LRP": EpsilonRule(epsilon=0.001),
        "MaxPool2d_LRP": Alpha1_Beta0_Rule(),
        "BatchNorm2d_LRP": IdentityRule(),
        "ReLU_convLRP": Alpha1_Beta0_Rule(),
        "ReLU_linearLRP": EpsilonRule(epsilon=0.001),
        "Sigmoid_LRP": EpsilonRule(epsilon=0.001),
        "Flatten_LRP": src.SkipFlattenRule(),
    }
    lrp_rules_variation = {
        "Conv2d_LRP": GammaRule(gamma=1),
        "Linear_LRP": EpsilonRule(epsilon=0.001),
        "MaxPool2d_LRP": GammaRule(gamma=1),
        "BatchNorm2d_LRP": IdentityRule(),
        "ReLU_convLRP": GammaRule(gamma=1),
        "ReLU_linearLRP": EpsilonRule(epsilon=0.001),
        "Sigmoid_LRP": EpsilonRule(epsilon=0.001),
        "Flatten_LRP": src.SkipFlattenRule(),
    }

    match mode:
        case "lrp":
            lrp_rules = lrp_rules_paper
        case "lrp_variation":
            lrp_rules = lrp_rules_variation
        case _:
            raise NotImplementedError(f'lrp mode "{mode}" unknown')

    if isinstance(module, src.Conv2d_LRP):
        module.rule = lrp_rules["Conv2d_LRP"]
    elif isinstance(module, src.Linear_LRP):
        module.rule = lrp_rules["Linear_LRP"]
    elif isinstance(module, src.MaxPool2d_LRP):
        module.rule = lrp_rules["MaxPool2d_LRP"]
    elif isinstance(module, src.BatchNorm2d_LRP):
        module.rule = lrp_rules["BatchNorm2d_LRP"]
    elif isinstance(module, src.ReLU_convLRP):
        module.rule = lrp_rules["ReLU_convLRP"]
    elif isinstance(module, src.ReLU_linearLRP):
        module.rule = lrp_rules["ReLU_linearLRP"]
    elif isinstance(module, src.Sigmoid_LRP):
        module.rule = lrp_rules["Sigmoid_LRP"]
    elif isinstance(module, src.Flatten_LRP):
        module.rule = lrp_rules["Flatten_LRP"]


def load_model_state(
    result_dir: str | Path, relation_module: src.RelationModuleFewShot
) -> src.RelationModuleFewShot:
    """Loads state dict from relation_module_best.pth in the provided result directory.

    Args:
        result_dir (str | Path): Path to result directory that contains relation_module_best.pth
        relation_module (RelationModuleFewShot): Instantiated model.

    Returns:
        RelationModuleFewShot: Trained relation module.
    """
    result_dir_path = Path(result_dir) if type(result_dir) == str else result_dir

    relation_module.load_state_dict(
        torch.load(result_dir_path / "relation_module_best.pth", map_location="cpu")
    )
    return relation_module
