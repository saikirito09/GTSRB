import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Building the model
class TrafficSignClassifier(nn.Module):
    def __init__(self):
        super(TrafficSignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 43)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv2(nn.functional.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv4(nn.functional.relu(self.conv3(x)))))
        x = self.dropout(x)
        
        # Flatten the tensor using reshape
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the saved model state dictionary
model_state_dict = torch.load('traffic_classifier.pth')

# Create an instance of the TrafficSignClassifier model
model = TrafficSignClassifier()

# Load the model state dictionary into the model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Streamlit app
st.title("Traffic Sign Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((30, 30))
    image = np.array(image)

    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Permute dimensions to match the expected order (channels, height, width)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Pass the tensor through the model
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class label
    _, predicted_label = torch.max(output, 1)
    predicted_label = predicted_label.item()

    # Display the image and predicted label
    st.image(image, caption="Input Image", use_column_width=True)
    st.write(f"Predicted label: {predicted_label}")
else:
    st.write("Please upload an image to get the predicted label.")
