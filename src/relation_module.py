import torch
from torch import Tensor, nn

import captum  # type: ignore
from captum.attr._utils.lrp_rules import EpsilonRule, Alpha1_Beta0_Rule, IdentityRule, PropagationRule  # type: ignore

import src.util

# https://captum.ai/tutorials/TorchVision_Interpret
# https://arxiv.org/pdf/1910.09840.pdf


class SkipFlattenRule(PropagationRule):
    """Propagation rule for nn.Flatten() is not implemented so far.
    Does nothing
    """

    def _manipulate_weights(self, module, inputs, outputs):
        """This does nothing. Implements identity for flatten layers for usage with captum"""
        pass


# captum versions of nn.Modules for LRP propagation
# because of a captum bug we set this in the training loop
class Conv2d_LRP(nn.Conv2d):
    """Alias for Conv2d to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class Linear_LRP(nn.Linear):
    """Alias for Linear to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class MaxPool2d_LRP(nn.MaxPool2d):
    """Alias for MaxPool2d to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class BatchNorm2d_LRP(nn.BatchNorm2d):
    """Alias for BatchNorm2d to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class ReLU_convLRP(nn.ReLU):
    """Alias for ReLU after convolutions to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class ReLU_linearLRP(nn.ReLU):
    """Alias for ReLU after linear layers to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class Sigmoid_LRP(nn.Sigmoid):
    """Alias for Sigmoid to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class Flatten_LRP(nn.Flatten):
    """Alias for Flatten to which the proper LRP rule will later be added

    Attributes:
        rule: LRP propagation rule
    """

    pass


class RelationModule(nn.Module):
    """Implements RelationNet using pytorch

    Attributes:
        padding: Which padding to use for convolutions
        in_channels: How many channels the input has
        hidden_channels: How many channels the hidden layers should use
        conv_out_channels: How many channels come out of the convolution layers
        relation_module_conv: The convolution layers as nn.Sequential
        flatten: flatten layer
        relation_module_final: final linear layers
    """

    def __init__(
        self,
        padding: int,
        in_channels: int,
        hidden_channels: int = 64,
        conv_out_channels: int = 576,
    ) -> None:
        """Default Relation network architecture.

        https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

        Args:
            padding (int): Padding for conv layers (1 for resnet embeddings else 0)
            in_channels (int): Number of channels of the embeddings.
            hidden_channels (int): Number of hidden channels of the conv layers in the relation module.
            conv_out_channels (int): Number of features after last convolution block in the relation module. Defaults to 576 for miniImageNet (from paper).
        """
        super().__init__()

        self.padding = padding
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv_out_channels = conv_out_channels

        self.relation_module_conv = nn.Sequential(
            # 1 block
            Conv2d_LRP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=padding,
            ),
            BatchNorm2d_LRP(num_features=self.hidden_channels),
            ReLU_convLRP(),
            MaxPool2d_LRP(kernel_size=(2, 2)),
            # 2 block
            Conv2d_LRP(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=padding,
            ),
            BatchNorm2d_LRP(num_features=self.hidden_channels),
            ReLU_convLRP(),
            MaxPool2d_LRP(kernel_size=(2, 2)),
        )

        # flatten output to pass spatial conv output to fc layers
        self.flatten = Flatten_LRP()

        self.relation_module_final = nn.Sequential(
            Linear_LRP(in_features=self.conv_out_channels, out_features=8),
            ReLU_linearLRP(),
            Linear_LRP(in_features=8, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of default relation network that produces a single relation score for each instance in the batch.

        Args:
            x (Tensor): Input image.

        Returns:
            Tensor: Relation score.
        """
        x = self.relation_module_conv(x)
        x = self.flatten(x)

        return self.relation_module_final(x)


class RelationModuleFewShot(RelationModule):
    """Implements RelationNet for fewshot-learning using pytorch

    Attributes:
        supportset: Supportset to use for few shot learning
        labels: string labels for supportset classes
        num_classes: how many classes are used in fewshot-learning
    """

    def __init__(
        self,
        padding: int,
        in_channels: int,
        hidden_channels: int = 64,
        conv_out_channels: int = 576,
    ) -> None:
        """Relation network

        Args:
            padding (int): Padding for the convolutional layers inside the module. Should be 1 with ResNet embeddings else 0
            in_channels (int): Number of input channels. Should be twice the number of output channels of the embedding module.
            hidden_channels (int): Number of hidden channels of the conv layers of the relation module.
            conv_out_channels (int, optional): Number of features after last convolution block in the relation module. Defaults to 576 for miniImageNet (from paper).
        """
        super().__init__(
            padding=padding,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            conv_out_channels=conv_out_channels,
        )

        self.supportset: dict[str, Tensor] = {}
        self.labels: list[str] = []
        self.num_classes: int = 0

        self.lrp_module = captum.attr.LRP(self)

    def update_supportset(self, supportset_embeddings: dict[str, Tensor]) -> None:
        """Computes and saves the summed embeddings of all images for each class for internal state of module.

        Args:
            supportset_embeddings (dict[str, Tensor]): Support set embeddings in format {"label": k embeddings, ...}
        """
        self.supportset = {}
        self.num_classes = len(supportset_embeddings)
        self.labels = list(supportset_embeddings.keys())

        for label, embeddings in supportset_embeddings.items():
            self.supportset[label] = embeddings.mean(dim=0).unsqueeze(dim=0)

    def get_label_indices(self, labels: tuple[str]) -> Tensor:
        """Gets label indices in few shot setting for every provided label in batch.

        Args:
            labels (tuple[str]): batch of target labels.

        Raises:
            ValueError: If label is not present in internally stored labels.

        Returns:
            Tensor: target label indices.
        """
        try:
            label_indices = [self.labels.index(label) for label in labels]
        except ValueError:
            raise ValueError(
                "A label could not be found in internally stored labels for n-way classification. Maybe you have to call update_supportset first."
            )
        return torch.tensor(label_indices, dtype=torch.int64)

    def forward(self, x_embedding: Tensor) -> Tensor:
        """Computes relation scores for each input embedding with the provided class embeddings.

        Args:
            x (Tensor): Input embeddings. Shape = (Batch, C, H, W)

        Returns:
            Tensor: Relation scores. Shape = (Batch, NumClasses)
        """
        # forward pass for each class
        full_embeddings = [
            torch.concat(
                # concatenates the class embedding to each image inside the batch (1 * [embedding] -> batch_size * [embedding]))
                [
                    support_embedding.expand(len(x_embedding), -1, -1, -1),
                    x_embedding,
                ],
                dim=1,
            )
            for support_embedding in self.supportset.values()
        ]
        full_embedding = torch.stack(full_embeddings, dim=0)

        flatten_for_conv_shape = torch.tensor(full_embedding.shape)[1:]
        flatten_for_conv_shape[0] = -1
        t = self.relation_module_conv(
            full_embedding.reshape(flatten_for_conv_shape.tolist())
        )
        t = self.flatten(t)
        t = self.relation_module_final(t)

        # after reshape dim0 represents the different supportset classes and dim1 represents the different x_embeddings
        # swap this via transpose
        return t.reshape(full_embedding.shape[0], full_embedding.shape[1]).transpose(
            0, 1
        )

    def forward_pass(
        self,
        x_embedding: Tensor,
        label_indices: Tensor,
        mode: str = "train",
        use_base_grad: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Performs a complete forward pass.
        1. Normal forward pass
        2. lrp-weighted forward pass

        Args:
            x_embedding (Tensor): Query input embeddings
            label_indices (Tensor): Indices of ground truth labels of query
            mode (str, optional): Mode of . Defaults to "train".

        Returns:
            tuple[Tensor, Tensor]: predictions, predictions with lrp weighting.
        """
        self.eval()
        x_embedding.requires_grad = True

        for module in self.modules():
            src.util.set_lrp_rules(module)

        lrp_weighting = self.lrp_module.attribute(
            x_embedding, label_indices
        )  # TODO test if input is unchanged
        lrp_embedding = x_embedding * (
            1 + src.util.normalize_lrp_weighting(lrp_weighting)
        )

        self.zero_grad()
        if mode == "train":
            self.train()
        x_embedding.requires_grad = False

        # first forward pass
        pred = self.forward(x_embedding)

        if not use_base_grad:
            self.zero_grad()

        # second forward pass
        pred_lrp = self.forward(lrp_embedding.detach())

        return pred, pred_lrp
