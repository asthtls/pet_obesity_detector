import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_load import get_dataloader
from model import get_model
from config import Config
from torch.cuda.amp import GradScaler, autocast

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_model_path, early_stop_patience):
    model.to(device)
    best_val_loss = float('inf')
    patience_count = 0  # Early stopping patience count 초기화
    
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Loop
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # AMP를 사용하여 혼합 정밀도로 forward-pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass와 최적화
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved with Validation Loss: {val_loss:.4f}")
            patience_count = 0  # Best model found, reset patience count
        else:
            patience_count += 1

        # # Early Stopping Check
        # if patience_count >= early_stop_patience:
        #     print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
        #     break

def main():
    # Load configuration from config.py
    config = Config()

    # Device Configuration
    torch.cuda.set_per_process_memory_fraction(0.8)
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Data Loaders
    train_loader = get_dataloader(config.train_csv, batch_size=config.batch_size, shuffle=True)
    val_loader = get_dataloader(config.val_csv, batch_size=config.batch_size, shuffle=False)

    # Model Initialization
    model = get_model(model_type=config.cnn_model_name)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model
    train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        config.num_epochs, 
        device, 
        config.save_model_path, 
        config.early_stop_patience
    )

if __name__ == "__main__":
    main()
