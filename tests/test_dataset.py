from src.dataset import MiniImageNetDataset
from src.dataset import CarsDataset
from src.dataset import Cifar10Dataset
from src.dataset import CUBDataset
import torchvision
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path


class TestMiniImageNetDataset:
    def test_has_items(self):
        dataset = MiniImageNetDataset(
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            )
        )

        # check if we have the right amount of train data
        assert len(dataset) == 54000

    def test_sampling(self):
        dataset = MiniImageNetDataset(
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
            n_way=27,
            k_shot=33,
        )
        supportset, (batch_images, batch_labels) = dataset.sample_batch(50)
        assert batch_images.shape[0] == 50
        assert len(batch_labels) == 50
        assert len(supportset) == 27
        for i in range(5):
            assert list(supportset.values())[i].shape[0] == 33


class TestCARSDataset:
    def test_parse_mat_file(self):
        muf = sio.loadmat("./dataset/CARS/cars_test_annos.mat")
        annotations = pd.DataFrame(muf["annotations"][0])
        annotations["fname"] = annotations["fname"].apply(lambda x: x[0])
        annotations["class"] = annotations["class"].apply(lambda x: x[0][0])
        annotations = annotations[["fname", "class"]]

        assert (
            type(annotations["class"][0]) == np.uint8
            and type(annotations["fname"][0]) == np.str_
        )

    def test_sampling(self):
        dataset = CarsDataset(
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            k_shot=27,
            n_way=33,
        )
        supportset, (batch_images, batch_labels) = dataset.sample_batch(50)
        assert len(supportset) == 33
        assert batch_images.shape[0] == 50
        assert len(batch_labels) == 50
        for i in range(5):
            assert list(supportset.values())[i].shape[0] == 27


class TestCUBDataset:
    def test_sampling(self):
        dataset = CUBDataset(
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            k_shot=27,
            n_way=33,
        )
        supportset, (batch_images, batch_labels) = dataset.sample_batch(50)
        assert len(supportset) == 33
        assert batch_images.shape[0] == 50
        assert len(batch_labels) == 50
        for i in range(5):
            assert list(supportset.values())[i].shape[0] == 27


class TestCIFAR10Dataset:
    def test_dataset(self):
        torchvision.datasets.CIFAR10(str(Path("dataset/cifar/")), download=True)

    def test_cifar10_dataset(self):
        dataset = Cifar10Dataset(
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            k_shot=3,
            n_way=3,
        )

        supportset, (batch_images, batch_labels) = dataset.sample_batch(8)
        assert len(supportset) == 3
        assert batch_images.shape[0] == 8
        assert len(batch_labels) == 8
        for i in range(3):
            assert list(supportset.values())[i].shape[0] == 3
