import pandas as pd
import torch
import torchvision  # type: ignore
from typing import TypeAlias
from PIL import Image

SupportSet: TypeAlias = dict[str, torch.Tensor]


class FewShotDataset:
    """This class implements sampling of few shot tasks from datasets.
    It is not to be used alone but by inheriting from it.

    Attributes:
        k_way: how many classes for support set
        n_shot: how many examples per class for support set
        labels: sorted list of the labels ("sorted set")
        dataset: dataset as pandas dataframe
        transform: transformation to use on data
        supportset: last used supportset, if mode is val or test this is not resampled on batch sampling operation
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        transform: torchvision.transforms.Compose,
        mode: str = "train",
        n_way: int = 5,
        k_shot: int = 5,
        load_images: bool = True,
    ):
        """Create new FewShotDataset.

        Args:
            dataset: dataset as pandas dataframe. Needs to have columns "images" (PIL.Image.Image) and "labels" (str)
            transform: torchvision transform to use. Use `torchvision.transforms.Compose([])` as identity function.
            mode: one of {"train", "val", "test"}
            n_way: how many classes should be used for fewshot learning
            k_shot: how many samples should every class have

        Raises:
            NotImplementedError:  When giving an unsupported mode
        """
        self.load_images = load_images

        self.k_way = n_way
        self.n_shot = k_shot

        assert set(["labels", "images"]) <= set(dataset.columns)
        dataset_unfiltered = dataset
        labels_unfiltered = sorted(list(dataset_unfiltered["labels"].unique()))

        self.labels: pd.Series
        match mode:
            case "train":  # delete the classes needed for later fewshot learning
                self.labels = pd.Series(labels_unfiltered[: -2 * n_way])
                self.sample_supportset = True
            case "val":  # use classes not used in training and leave some for testing
                self.labels = pd.Series(labels_unfiltered[-2 * n_way : -n_way])
                self.sample_supportset = False
            case "test":  # use classes not used in training or validating
                self.labels = pd.Series(labels_unfiltered[-n_way:])
                self.sample_supportset = False
            case _:
                raise NotImplementedError(f"{mode=} not known")

        self.dataset: pd.DataFrame = dataset_unfiltered[dataset_unfiltered["labels"].isin(self.labels)]  # type: ignore
        self.transform: torchvision.transforms.Compose = transform
        self.supportset: SupportSet = self.get_supportset()

    def __len__(self) -> int:
        """How many images are in the dataset

        Returns:
            amount of images
        """
        return len(self.dataset)

    def get_supportset(self) -> SupportSet:
        """Create Supportset according to fewshot-learning-task

        Returns:
            Supportset
        """

        classes = self.labels.sample(self.k_way, replace=False)
        supportset = {
            c: torch.stack(
                tuple(
                    self.dataset[self.dataset["labels"] == c]["images"]
                    .sample(self.n_shot, replace=False)
                    .apply(
                        lambda path_or_image: self.transform(
                            Image.open(path_or_image)
                            if self.load_images
                            else path_or_image
                        )
                    )
                ),
                dim=0,
            )
            for c in classes
        }  # type: ignore
        return supportset

    def sample_batch(
        self, batch_size: int
    ) -> tuple[SupportSet, tuple[torch.Tensor, tuple[str]]]:
        """Sample a "batch" meaning give back one supportset and n images matching into one class of the supportset

        Args:
            batch_size: how many images to give back in query set

        Returns:
            supportset, (images, labels)
        """
        if self.sample_supportset:
            self.supportset = self.get_supportset()

        batch = self.dataset[
            self.dataset["labels"].isin(list(self.supportset.keys()))
        ].sample(
            batch_size
        )  # type: ignore

        return self.supportset, (
            torch.stack(
                tuple(
                    batch["images"].apply(
                        lambda path_or_image: self.transform(
                            Image.open(path_or_image)
                            if self.load_images
                            else path_or_image
                        )
                    )
                ),
                dim=0,
            ),
            tuple(batch["labels"]),
        )  # type: ignore
