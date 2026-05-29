import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ConvNet  # Importing the model architecture we just made!

def main():
    # 1. Setup Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5

    # 3. Data Transformations & Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model, Loss Function, and Optimizer
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. The Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass: Compute predicted outputs by passing images to the model
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients 
            optimizer.step()      # Update weights
            
            running_loss += loss.item()
            
            if (i+1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} complete. Average Loss: {running_loss/len(train_loader):.4f}")

    print("Training finished!")
    
    # 6. Save the trained weights so we can use them later for deployment
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model weights saved to mnist_cnn.pth")

if __name__ == '__main__':
    main()