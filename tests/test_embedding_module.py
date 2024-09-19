import pytest
import torch

from src.embedding_module import *


class TestDefaultEmbeddingModule:
    def test_shapes(self):
        for i in range(1, 100, 5):
            module = DefaultEmbeddingModule(in_channels=i, hidden_channels=64)
            x = torch.zeros(1, i, 84, 84)
            assert module(x).shape == torch.Size([1, 64, 21, 21])
            x = torch.zeros(25, i, 84, 84)
            assert module(x).shape == torch.Size([25, 64, 21, 21])

    def test_support_set_embeddings(self):
        module = ResNetEmbeddingModule()
        supportset = {
            # 5 images per class
            "apple": torch.zeros([5, 3, 224, 224]),
            "banana": torch.zeros([5, 3, 224, 224]),
            "cherry": torch.zeros([5, 3, 224, 224]),
        }
        embedding_supportset = module.get_supportset_embeddings(supportset)
        assert type(embedding_supportset) == dict
        assert len(embedding_supportset) == 3
        assert embedding_supportset["banana"].shape == torch.Size([5, 512, 7, 7])


class TestResNetEmbeddingModule:
    def test_shapes(self):
        module = ResNetEmbeddingModule()
        x = torch.zeros(1, 3, 224, 224)
        assert module(x).shape == torch.Size([1, 512, 7, 7])
        x = torch.zeros(25, 3, 224, 224)
        assert module(x).shape == torch.Size([25, 512, 7, 7])

    def test_support_set_embeddings(self):
        module = ResNetEmbeddingModule()
        supportset = {
            # 5 images per class
            "apple": torch.zeros([5, 3, 224, 224]),
            "banana": torch.zeros([5, 3, 224, 224]),
            "cherry": torch.zeros([5, 3, 224, 224]),
        }
        embedding_supportset = module.get_supportset_embeddings(supportset)
        assert type(embedding_supportset) == dict
        assert len(embedding_supportset) == 3
        assert embedding_supportset["cherry"].shape == torch.Size([5, 512, 7, 7])
