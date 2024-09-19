import pytest
import torch
import captum

from src.embedding_module import DefaultEmbeddingModule, ResNetEmbeddingModule
from src.relation_module import *
from src.util import set_lrp_rules


class TestLRP:
    def test_lrp_modules(self):
        linear_input = torch.ones([1, 512], requires_grad=True)
        layer = Linear_LRP(512, 1)
        set_lrp_rules(layer)
        linear = captum.attr.LRP(layer)
        assert linear.forward_func(linear_input).shape == torch.Size([1, 1])
        assert linear.attribute(linear_input).shape == linear_input.shape

        conv_input = torch.ones([1, 1, 3, 3], requires_grad=True)
        layer = Conv2d_LRP(1, 1, kernel_size=(5, 5), padding=1)
        set_lrp_rules(layer)
        conv = captum.attr.LRP(layer)
        assert conv.forward_func(conv_input).shape == torch.Size([1, 1, 1, 1])
        assert conv.attribute(conv_input).shape == conv_input.shape

        maxpool_input = torch.ones([1, 1, 2, 2], requires_grad=True)
        layer = MaxPool2d_LRP((2, 2))
        set_lrp_rules(layer)
        maxpool = captum.attr.LRP(layer)
        assert maxpool.forward_func(maxpool_input).shape == torch.Size([1, 1, 1, 1])
        assert maxpool.attribute(maxpool_input).shape == maxpool_input.shape

        batchnorm_input = torch.ones([1, 1, 3, 3], requires_grad=True)
        module = nn.Sequential(BatchNorm2d_LRP(1), Conv2d_LRP(1, 1, kernel_size=(3, 3)))
        for submodule in module.modules():
            set_lrp_rules(submodule)
        batchnorm = captum.attr.LRP(module)
        assert batchnorm.attribute(batchnorm_input).shape == batchnorm_input.shape

        reluconv_input = torch.ones([1, 1, 3, 3], requires_grad=True)
        module = nn.Sequential(Conv2d_LRP(1, 1, kernel_size=(3, 3)), ReLU_convLRP())
        for submodule in module.modules():
            set_lrp_rules(submodule)
        reluconv = captum.attr.LRP(module)
        assert reluconv.attribute(reluconv_input).shape == reluconv_input.shape

        relulin_input = torch.ones([1, 512], requires_grad=True)
        module = nn.Sequential(Linear_LRP(512, 1), ReLU_linearLRP())
        for submodule in module.modules():
            set_lrp_rules(submodule)
        relulin = captum.attr.LRP(module)
        assert relulin.attribute(relulin_input).shape == relulin_input.shape

        sigmoid_input = torch.ones([1, 512], requires_grad=True)
        module = nn.Sequential(Linear_LRP(512, 1), Sigmoid_LRP())
        for submodule in module.modules():
            set_lrp_rules(submodule)
        sigmoid = captum.attr.LRP(module)
        assert sigmoid.attribute(sigmoid_input).shape == sigmoid_input.shape

        flatten_input = torch.ones([1, 1, 16, 16], requires_grad=True)
        module = nn.Sequential(Flatten_LRP(), Linear_LRP(256, 1))
        for submodule in module.modules():
            set_lrp_rules(submodule)
        flatten = captum.attr.LRP(module)
        assert flatten.attribute(flatten_input).shape == flatten_input.shape


class TestRN:
    def test_shapes(self):
        embedding_module1 = DefaultEmbeddingModule(in_channels=3, hidden_channels=64)
        module1 = RelationModule(
            0,
            64,
            hidden_channels=64,
            conv_out_channels=576,
        )
        for i in range(1, 100, 5):
            x = torch.zeros(i, 3, 84, 84)
            x = embedding_module1(x)
            assert module1(x).shape == torch.Size([i, 1])

        embedding_module2 = ResNetEmbeddingModule()
        module2 = RelationModule(
            1,
            512,
            hidden_channels=512,
            conv_out_channels=512,
        )
        for i in range(1, 100, 5):
            x = torch.zeros(i, 3, 224, 224)
            x = embedding_module2(x)
            assert module2(x).shape == torch.Size([i, 1])

    def test_lrp(self):
        input = torch.ones([3, 512, 7, 7], requires_grad=True)
        module = RelationModule(1, 512, 512, 512)
        for submodule in module.modules():
            set_lrp_rules(submodule)
        module = captum.attr.LRP(module)
        output = module.attribute(input)


class TestRNFSC:
    def test_shapes(self):
        supportset1 = {
            # 5 images per class
            "apple": torch.zeros([5, 3, 84, 84]),
            "banana": torch.zeros([5, 3, 84, 84]),
            "cherry": torch.zeros([5, 3, 84, 84]),
        }

        embedding_module1 = DefaultEmbeddingModule(in_channels=3, hidden_channels=64)
        supportset_embeddings1 = embedding_module1.get_supportset_embeddings(
            supportset1
        )
        module1 = RelationModuleFewShot(
            0,
            64 * 2,
            hidden_channels=64,
            conv_out_channels=576,
        )
        module1.update_supportset(supportset_embeddings1)
        for i in range(1, 100, 5):
            x = torch.zeros(i, 64, 21, 21)
            assert module1(x).shape == torch.Size([i, len(supportset1)])

        supportset2 = {
            # 5 images per class
            "apple": torch.zeros([5, 3, 224, 224]),
            "banana": torch.zeros([5, 3, 224, 224]),
            "cherry": torch.zeros([5, 3, 224, 224]),
        }

        embedding_module2 = ResNetEmbeddingModule()
        supportset_embeddings2 = embedding_module2.get_supportset_embeddings(
            supportset2
        )
        module2 = RelationModuleFewShot(
            1,
            512 * 2,
            hidden_channels=512,
            conv_out_channels=512,
        )
        module2.update_supportset(supportset_embeddings2)
        for i in range(1, 100, 5):
            x = torch.zeros(i, 512, 7, 7)
            assert module2(x).shape == torch.Size([i, len(supportset2)])

    def test_backprop(self):
        supportset = {
            # 5 images per class
            "apple": torch.zeros([5, 512, 7, 7]),
            "banana": torch.zeros([5, 512, 7, 7]),
            "cherry": torch.zeros([5, 512, 7, 7]),
        }
        module = RelationModuleFewShot(
            1, 512 * 2, hidden_channels=512, conv_out_channels=512
        )
        for submodule in module.modules():
            set_lrp_rules(submodule)
        module.update_supportset(supportset)
        x = torch.randn([8, 512, 7, 7])
        target_labels = torch.tensor([0, 1, 2, 0, 1, 2, 1, 2], dtype=torch.int64)

        lrp_module = captum.attr.LRP(module)
        x.requires_grad = True
        lrp_weighting = lrp_module.attribute(x, target_labels)
        x.requires_grad = False

        ce_loss = nn.CrossEntropyLoss()
        loss = ce_loss(module(x), target_labels) + ce_loss(
            module(x * lrp_weighting.detach()), target_labels
        )
        loss.backward()

    def test_getlabels(self):
        supportset = {
            # 5 images per class
            "apple": torch.zeros([5, 512, 7, 7]),
            "banana": torch.zeros([5, 512, 7, 7]),
            "cherry": torch.zeros([5, 512, 7, 7]),
        }
        labels = ("apple", "cherry", "banana", "apple", "banana")
        module = RelationModuleFewShot(
            1, 512 * 2, hidden_channels=512, conv_out_channels=512
        )
        module.update_supportset(supportset)

        assert torch.equal(
            module.get_label_indices(labels),
            torch.tensor([0, 2, 1, 0, 1], dtype=torch.int),
        )
