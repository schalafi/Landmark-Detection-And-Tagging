import torch
import torch.nn as nn




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out



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
            # input 224x224 -> 112x112 output
            self.create_conv_block(
                in_channels = 3, # RGB channels
                out_channels = 32,
                dropout = dropout),
            # from 112x112 -> 56x56
            self.create_conv_block(
                in_channels = 32, 
                out_channels = 64, 
                dropout = dropout),
            # Residual Block after second convolutional block
            ResidualBlock(64, 64),
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
            # Residual Block after fourth convolutional block
            ResidualBlock(256, 256),
            # 14x14 -> 7x7
            # current  shape is 7x7x512
            self.create_conv_block(
                in_channels=256,
                out_channels= 512,
                dropout = dropout 
            ),
            
            #nn.AdaptiveAvgPool2d((1, 1)),  # global Average Pooling

            # Convert the matrices to vectors
            nn.Flatten(),
            
            ### 2 Layer Perceptron
            # First layer
            nn.Linear(in_features = 7*7*512,
                      out_features = 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
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
