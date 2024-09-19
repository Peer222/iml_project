from pathlib import Path
import pandas as pd
import torch
import torchvision  # type: ignore
import scipy.io as sio  # type: ignore
from .fewshot_dataset import FewShotDataset


class CarsDataset(FewShotDataset):
    """Implements Cars dataset as fewshot dataset

    Attributes:
        img_dir: directory where to find the images
    """

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        mode: str = "train",
        k_shot: int = 5,
        n_way: int = 5,
        img_dir: str = "./dataset/CARS/",
    ):
        """Create new Cars dataset object

        Args:
            transform: transformations to apply to images
            mode: one of {"train", "val", "test"}
            k_shot: how many images per class the supportset has
            n_way: how many classes the supportset has
            img_dir: Where the dataset is located
        """
        self.img_dir: str = img_dir
        datasets_raw = {}

        # dataset_raw: pd.DataFrame =

        # we split on classes anyways, use all of the different given splits and merge them
        for split in ["test", "train"]:
            muf = sio.loadmat(f"./dataset/CARS/cars_{split}_annos.mat")
            annotations = pd.DataFrame(muf["annotations"][0])
            annotations["fname"] = annotations["fname"].apply(lambda x: x[0])
            annotations["label"] = annotations["class"].apply(lambda x: x[0][0])
            annotations = annotations[["fname", "label"]]

            datasets_raw[split] = annotations
            datasets_raw[split]["images"] = annotations.apply(
                lambda row: self.row_to_path(row, f"cars_{split}"), axis=1
            )

        dataset_raw: pd.DataFrame = pd.concat(list(datasets_raw.values()))
        dataset_raw["labels"] = annotations["label"]  # type: ignore
        super().__init__(dataset_raw[["labels", "images"]], transform, mode=mode, k_shot=k_shot, n_way=n_way)  # type: ignore

    def row_to_path(self, row, mode_dir: str) -> str:
        """Convert dataset row to image tensor

        Args:
            row: pandas dataset row having label and filename
            mode_dir: subdir to use (named as modes)

        Returns:
            image
        """
        return str(Path(self.img_dir) / mode_dir / mode_dir / row["fname"])
