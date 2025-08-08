from torch import nn
from torch.nn import Sequential

from src.model.mfm import MFM


class LCNN(nn.Module):
    """
    LightCNN architecture implemented from the following paper with the addition of a dropout layer at the end:
    https://arxiv.org/abs/1904.05576
    """

    def __init__(self, input_channels, output_size):
        """
        Args:
            input_channels (int): number of input channels.
            output_size (int): number of output logits.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(160),
            MFM(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(80),
            nn.Linear(80, output_size),
        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            image (Tensor): input image.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
