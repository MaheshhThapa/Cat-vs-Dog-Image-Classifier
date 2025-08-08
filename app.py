import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import webbrowser
import threading

from network import Network  # import your model definition here

# Load the trained model weights
model = Network()
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=torch.device("cpu")))
model.eval()

# Hardcoded class index to label mapping
idx_to_class = {0: "cat", 1: "dog"}

# Image preprocessing - same as during training
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img):
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        # Ensure outputs shape is [batch_size, num_classes]
        if outputs.dim() == 1 or outputs.size(1) == 1:
            outputs = outputs.view(1, -1)
        _, predicted = torch.max(outputs, 1)
    return idx_to_class[predicted.item()]

def classify_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    prediction = predict(image)
    return f"Prediction: {prediction}"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Cat vs Dog Classifier"
)

# Auto-open browser when launching
def open_browser():
    webbrowser.open("http://127.0.0.1:7860")

threading.Timer(1, open_browser).start()
demo.launch()
