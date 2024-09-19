from abc import abstractmethod

import torch
from torch import nn, Tensor
from torchvision.models.resnet import ResNet, ResNet18_Weights  # type: ignore


class EmbeddingModule(nn.Module):
    """Abstract Class which provides an interface to modules providing an embedding used by the learned metric.

    Attributes:
        in_channels: how many channels the input has
        output_channels: how many channels the embedding has
        hidden_channels: how many channels are used internally in the embedding
        network: the network which generates the embedding
    """

    def __init__(
        self, in_channels: int = 3, hidden_channels: int = 64, output_channels: int = 64
    ) -> None:
        """Super class for embedding module.

        Args:
            in_channels (int, optional): Number of channels of the input image. Defaults to 3.
        """
        super().__init__()

        self.in_channels = in_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        self.network: nn.Module = nn.Identity()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Abstract method for computing the embedding of the input.

        Args:
            x (Tensor): Input image(s).

        Returns:
            Tensor: Embedding(s).
        """
        raise NotImplementedError

    def get_supportset_embeddings(
        self, supportset: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Computes embeddings of the support set.

        Args:
            support_set (dict[str, Tensor]): Support set in format {"label": k images, ...}

        Returns:
            dict[str, Tensor]: Support set with embeddings in format {"label": k embeddings, ...}
        """
        embedding_supportset = {}
        for label, images in supportset.items():
            embedding_supportset[label] = self.forward(images)
        return embedding_supportset


class DefaultEmbeddingModule(EmbeddingModule):
    """Implements default embedding network from
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf.

    Attributes:
        network: network which generates the embedding
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64) -> None:
        """Implements default embedding network from
        https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf.

        Expects in_channels x 84 x 84 inputs

        Args:
            in_channels (int, optional): Number of channels of the input image. Defaults to 3.
        """
        super().__init__(in_channels, hidden_channels=hidden_channels)

        self.network = nn.Sequential(
            # 1 block
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 2 block
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 3 block
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.hidden_channels),
            nn.ReLU(),
            # 4 block
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.hidden_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Abstract method for computing the embedding of the input.

        Args:
            x (Tensor): Input image(s).

        Returns:
            Tensor: Embedding(s).
        """
        return self.network(x)


class ResNetEmbeddingModule(EmbeddingModule):
    """Create Image embedding via ResNetModel
    Parameters are mostly taken out of the original paper.

    Attributes:
        network: The network to create the embedding.
    """

    def __init__(self) -> None:
        """Initialize Resnet18
        Pretrained from pytorch.

        Args:
            in_channels: input channels of the image
        """
        super().__init__(3, hidden_channels=512, output_channels=512)

        self.network: ResNet = torch.hub.load(
            "pytorch/vision:v0.16.2", "resnet18", weights=ResNet18_Weights.DEFAULT
        )

    def forward(self, x: torch.Tensor):
        """Forward input through resnet module.
        Returns image embedding generated through using resnet without the last layer

        Args:
            x: input

        Returns:
            image embedding
        """
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        return x
