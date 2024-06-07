import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# Define the model architecture
model = ssdlite320_mobilenet_v3_large(pretrained=False)

# Load the saved state dictionary
model.load_state_dict(torch.load('ssd_mobilenet_v3.pth'))

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully.")
# Transform to be applied to the image
transform = transforms.Compose([
    transforms.ToTensor()
])


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image)
    return predictions


# Example usage
image_path = 'path_to_your_image.jpg'
predictions = predict(image_path)
print(predictions)
