from pathlib import Path
import pandas as pd
import torch
import torchvision  # type: ignore
import scipy.io as sio  # type: ignore
from PIL import Image

from .fewshot_dataset import FewShotDataset


class Cifar10Dataset(FewShotDataset):
    """Implements Cifar10 dataset as fewshot dataset

    Attributes:
        img_dir: directory where to find the images
    """

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        mode: str = "train",
        k_shot: int = 5,
        n_way: int = 5,
        dataset_dir: str = "dataset/cifar/",
    ):
        """Create new Cifar10 dataset object

        Args:
            transform: transformations to apply to images
            mode: one of {"train", "val", "test"}
            k_shot: how many images per class the supportset has
            n_way: how many classes the supportset has
            dataset_dir: where the dataset is located
        """

        dataset = torchvision.datasets.CIFAR10(str(Path(dataset_dir)), download=True)

        dataset_raw: pd.DataFrame = pd.DataFrame(
            {
                "images": [Image.fromarray(image) for image in dataset.data],
                "labels": dataset.targets,
            }
        )

        super().__init__(
            dataset_raw[["labels", "images"]],
            transform,
            mode=mode,
            k_shot=k_shot,
            n_way=n_way,
            load_images=False,
        )  # type: ignore
