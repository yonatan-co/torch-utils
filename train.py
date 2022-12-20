
import os 
import torch
import data_setup, engine, build_model, utils

from torchvision import transforms

EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10 
LEARNING_RATE = 0.001

train_dir = "/content/data/pizza_steak_sushi/train"
test_dir = "/content/data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"


data_transform = transforms.Compose([
    transforms.Resize(64, 64),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)


model = build_model.Model(input_shape=3,
                          hidden_units=HIDDEN_UNITS,
                          output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)


results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epochs=EPOCHS,
                       device=device)

utils.save_model(mdoel=model,
                 target_dir="models",
                 model_name="TinyVGG")
