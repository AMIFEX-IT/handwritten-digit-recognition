import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from model import ConvNet

# 1. Setup device and load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()  # Set to evaluation mode

# 2. Define the preprocessing transform for the drawn image
# The interface will give us an image, but we need to match the exact format used in training
transform = transforms.Compose([
    transforms.Resize((28, 28)),          # Resize to 28x28 pixels
    transforms.ToTensor(),                # Convert to PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize mean and std to 0.5
])

def predict_digit(sketchpad):
    if sketchpad is None:
        return "Please draw a digit!"
    
    # Gradio sketchpad returns a dictionary with 'composite' or 'layers' image data.
    # We grab the composite image where the drawing is present.
    img = sketchpad["composite"]
    
    # The canvas uses transparency (RGBA). We convert it to Grayscale ('L')
    # to extract just the drawn strokes.
    img = Image.fromarray(img).convert('L')
    
    # Apply our image transformations and add a batch dimension (1, 1, 28, 28)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Run the image through the model
    with torch.no_grad():
        outputs = model(img_tensor)
        # Apply softmax to convert raw scores (logits) into probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Return a dictionary of labels and their corresponding probabilities for Gradio to display
    return {str(i): float(probabilities[i]) for i in range(10)}

# 3. Build the Gradio Web Interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy", label="Draw a single digit (0-9)"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="Handwritten Digit Recognition 🧠",
    description="Draw a number inside the box and watch the trained Convolutional Neural Network (CNN) predict it live!",
    theme="soft"
)

# 4. Launch the local web server
if __name__ == "__main__":
    interface.launch()