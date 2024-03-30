import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import default_data_collator
from tqdm import tqdm
from loguru import logger

from model import SiMBA

# Set up logging
logger.add("logs/training.log", rotation="10 MB", retention=3)

# Set up TensorBoard writer
writer = SummaryWriter("runs/simba_experiment")

# Load the dataset
dataset = load_dataset("cifar10")


# Define the data preprocessing function
def preprocess_data(examples):
    images = examples["img"]
    _labels = examples["label"]

    # Normalize pixel values to [0, 1]
    images = images.float() / 255.0

    return {"pixel_values": images, "labels": _labels}


# Preprocess the dataset
dataset = dataset.map(preprocess_data, batched=True, remove_columns=["img"], num_proc=4)

# Create data loaders with pinned memory
train_dataloader = DataLoader(
    dataset["train"],
    batch_size=32,
    shuffle=True,
    collate_fn=default_data_collator,
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    dataset["test"],
    batch_size=32,
    collate_fn=default_data_collator,
    num_workers=4,
    pin_memory=True,
)

# Define the model
model = SiMBA(image_size=32, channels=128, num_blocks=6, heads=8)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Define the device and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the number of epochs and gradient accumulation steps
num_epochs = 50
grad_accum_steps = 4

# Define early stopping parameters
early_stopping_patience = 5
best_val_loss = float("inf")
early_stopping_counter = 0

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch + 1} - Training")
    ):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / grad_accum_steps
        loss.backward()
        train_loss += loss.item()

        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Validation"):
            inputs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    logger.info(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model checkpoint
        torch.save(model.state_dict(), "best_model.pth")
        logger.info(f"Best model checkpoint saved. Val Loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered. Best Val Loss: {best_val_loss:.4f}")
            break

# Load the best model checkpoint
model.load_state_dict(torch.load("best_model.pth"))

# Evaluate the best model on the test set
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Test"):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_total += targets.size(0)
        test_correct += torch.sum(predicted == targets).item()

test_loss /= len(val_dataloader)
test_accuracy = test_correct / test_total

logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Close the TensorBoard writer
writer.close()

logger.info("Training completed.")
