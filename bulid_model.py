import torch
from torch.nn import Sequential, ReLU, Conv2d, MaxPool2d, Linear, Module, Flatten

class TinyVGG(Module):
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


import torchvision
from torch.nn import Sequential, Dropout, Linear 

def create_efficentnet(output_shape: int,
                       model_type: str="b0",
                       trainable: bool=False,
                       name: str="effnetb0"):
  """
  a function to create an efficentnet model with proper classifier.
  note: the model is suitable only for images with shape of : (batch_size, 3, 224, 224)
  Args:
    output_shape: the output shape of the new classifer initilized for the model.
    model_type: the type of efficentnet to create (e.g 'b0', 'b2').
    trainable: to make the model's layers trainable or not?
    name: the name of the created model
  
  Returns:
    an efficent net model with suitable for your problem. 
  """
  if model_type == "b0":
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model= torchvision.models.efficientnet_b0(weights=weights)
    hidden = 1280
  elif model_type == "b1":
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    model= torchvision.models.efficientnet_b1(weights=weights)
    hidden = 1280
  elif model_type == "b2":
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model= torchvision.models.efficientnet_b2(weights=weights)
    hidden = 1408
  elif model_type == "b3":
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model= torchvision.models.efficientnet_b3(weights=weights)
    hidden = 1536
  elif model_type == "b4":
    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model= torchvision.models.efficientnet_b4(weights=weights)
    hidden = 1792
  elif model_type == "b5":
    weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
    model= torchvision.models.efficientnet_b5(weights=weights)
    hidden = 2048
  elif model_type == "b6":
    weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
    model= torchvision.models.efficientnet_b6(weights=weights)
    hidden = 2304
  elif model_type == "b7":
    weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    model= torchvision.models.efficientnet_b7(weights=weights)  
    hidden = 2560

  else:
    print(f"efficentnet of type {model_type} was not found")


  if not trainable:
    for param in model.parameters():
      param.requires_grad = False

  
  model.classifier = Sequential(
      Dropout(0.2),
      Linear(in_features=hidden, 
             out_features=output_shape)
  )
  model.name = name
  print(f"created new {model.name}")
  return model
