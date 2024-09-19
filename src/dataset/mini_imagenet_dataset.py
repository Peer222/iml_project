from pathlib import Path
import pandas as pd
import torchvision  # type: ignore
from .fewshot_dataset import FewShotDataset


class MiniImageNetDataset(FewShotDataset):
    """Implements MiniImageNet dataset as fewshot dataset

    Attributes:
        img_dir: directory where to find the images
    """

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        mode: str = "train",
        n_way: int = 5,
        k_shot: int = 5,
        img_dir: str = "./dataset/mini_imagenet/processed_images/",
        dataset_directory: str = "./dataset/mini_imagenet/csv_files",
    ):
        """Create MiniImageNetDataset

        Args:
            transform: torchvision transform to use. Use `torchvision.transforms.Compose([])` as identity function.
            mode: one of {"train", "test", "val"}
            n_way: how many classes should be used for fewshot learning
            k_shot: how many samples should every class have
            img_dir: directory where miniImageNet-images are located
            dataset_directory: directory where the datasets csv files are located
        """
        self.img_dir: str = img_dir
        datasets_raw = {}

        # we split on classes anyways, use all of the different given splits and merge them
        for split in ["test", "train", "val"]:
            datasets_raw[split] = pd.read_csv(
                Path(dataset_directory) / (split + ".csv")
            )
            datasets_raw[split]["images"] = datasets_raw[split].apply(
                lambda row: self.row_to_path(row, split), axis=1
            )

        dataset_raw: pd.DataFrame = pd.concat(list(datasets_raw.values()))
        dataset_raw["labels"] = dataset_raw["label"]
        super().__init__(dataset_raw[["labels", "images"]], transform, mode=mode, n_way=n_way, k_shot=k_shot)  # type: ignore

    def row_to_path(self, row, mode_dir: str) -> str:
        """Convert dataset row to pillow image

        Args:
            row: pandas dataset row having label and filename
            mode_dir: subdir to use (named as modes)

        Returns:
            image
        """
        return str(Path(self.img_dir) / mode_dir / row["label"] / row["filename"])
