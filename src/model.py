import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        self.dropout = dropout 
        self.num_classes = num_classes 

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            # input 224x224 -> 112x112
            self.create_conv_block(
                in_channels = 3, # RGB channels
                out_channels = 32,
                dropout = dropout),
            # from 112x112 -> 56x56
            self.create_conv_block(
                in_channels = 32, 
                out_channels = 64, 
                dropout = dropout),
            # from 56x56 -> 28x28
            self.create_conv_block(
                in_channels = 64,
                out_channels = 128, 
                dropout = dropout),
            # 28x28 -> 14x14
            self.create_conv_block(
                in_channels = 128,
                out_channels = 256,
                dropout = dropout),
            # 14x14 -> 7x7
            # current  shape is 7x7x512
            self.create_conv_block(
                in_channels=256,
                out_channels= 512,
                dropout = dropout 
            ),
            # Convert the matrices to vectors
            nn.Flatten(),
            ### 2 Layer Perceptron
            # First layer
            nn.Linear(in_features = 7*7*512,
                      out_features = 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Second layer
            nn.Linear(in_features = 1024,
                      out_features = num_classes ),
            )            

    def create_conv_block(self,in_channels, out_channels,dropout):

        return  nn.Sequential(
            nn.Conv2d(# conv+ pooling + relu 
                # keep the same size after convolution (kernel size = 3 and padding = 1 )
                in_channels = in_channels, # RGB channels  
                out_channels = out_channels,
                kernel_size = 3, 
                stride = 1 , # 1 pixel step while moving the kernel
                padding = 1 , # keep same size after conv) 
                    ),
            # Reduces the image size in half
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.ReLU(),
            # Batch normalization
            # appl first BN, because dropout changes the distribution
            # of the data while training
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
