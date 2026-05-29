import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ConvNet

def evaluate():
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Transformations & Loading Test Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    # 3. Load Model and Weights
    model = ConvNet().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # 4. Evaluation Loop
    correct = 0
    total = 0
    
    print("Evaluating model on test dataset...")
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get output scores
            outputs = model(images)
            
            # Get the index of the highest score (this is the predicted digit)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the 10,000 test images: {accuracy:.2f}%")

if __name__ == '__main__':
    evaluate()