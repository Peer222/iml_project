from pathlib import Path
import pandas as pd
import torchvision
from .fewshot_dataset import FewShotDataset


class CUBDataset(FewShotDataset):
    """Implements CUB dataset as fewshot dataset

    Attributes:
        img_dir: directory where to find the images
    """

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        mode: str = "train",
        k_shot: int = 5,
        n_way: int = 5,
        dataset_dir: str = "./dataset/CUB_200_2011/",
    ):
        """Create new CUB(Birds) dataset object

        Args:
            transform: transformations to apply to images
            mode: one of {"train", "val", "test"}
            k_shot: how many images per class the supportset has
            n_way: how many classes the supportset has
            dataset_dir: Where the dataset is located
        """
        self.dataset_dir: Path = Path(dataset_dir)

        df_images: pd.DataFrame = pd.read_csv(
            str(self.dataset_dir / "images.txt"), sep=" ", header=None
        )
        df_images.columns = ["image_id", "image_path"]
        df_images = df_images.set_index("image_id")

        df_image_labels = pd.read_csv(
            self.dataset_dir / "image_class_labels.txt", sep=" ", header=None
        )
        df_image_labels.columns = ["image_id", "label_id"]

        df_labels_id: pd.DataFrame = pd.read_csv(
            self.dataset_dir / "classes.txt", sep=" ", header=None
        )
        df_labels_id.columns = ["label_id", "label"]
        df_labels_id = df_labels_id.set_index("label_id")

        result = df_image_labels.join(df_images, on="image_id", how="inner").join(
            df_labels_id, on="label_id", how="inner"
        )

        result = result[["image_path", "label"]]
        result["images"] = result.apply(
            lambda row: str(self.dataset_dir / "images" / row["image_path"]), axis=1
        )

        result["labels"] = result["label"]  # type: ignore
        super().__init__(result[["labels", "images"]], transform, mode=mode, k_shot=k_shot, n_way=n_way)  # type: ignore
