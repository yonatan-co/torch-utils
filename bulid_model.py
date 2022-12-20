import torch
from torch.nn import Sequential, ReLU, Conv2d, MaxPool2d, Linear, Module, Flatten

class Model(Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.block1 = Sequential(
        Conv2d(in_channels=input_shape,
               out_channels=hidden_units,
               kernel_size=3,
               padding=0),
        ReLU(),
        Conv2d(in_channels=hidden_units,
               out_channels=hidden_units,
               kernel_size=3,
               padding=0),
        ReLU(),
        MaxPool2d(kernel_size=2,
                  stride=2)
    ),
    # shape : (hidden_units, img_size/2, img_size / 2)
    self.block2 = Sequential(
      Conv2d(in_channels=hidden_units,
             out_channels=hidden_units,
             kernel_size=3,
             padding=0),
      ReLU(),
      Conv2d(in_channels=hidden_units,
             out_channels=hidden_units,
             kernel_size=3,
             padding=0),
      ReLU(),
      MaxPool2d(kernel_size=2)
    )
    # shape : (hidden_units, img_size / 4, img_size / 4)
    self.classifier = Sequential(
        Flatten(),
        Linear(in_features=hidden_units * 16 * 16,
               out_features=3)        
    )
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.classifier(x)
    return x

model = Model(input_shape=3,
              hidden_units=10,
              output_shape=3) 
