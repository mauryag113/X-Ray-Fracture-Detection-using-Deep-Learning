# app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import FractureClassifier

# 🧠 Class labels
class_names = ['fractured', 'not fractured']

# 📌 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FractureClassifier().to(device)
model.load_state_dict(torch.load("final_fracture_model.pth", map_location=device))
model.eval()

# 🔁 Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🖼️ Streamlit App
st.set_page_config(page_title="Fracture Detection", layout="centered")
st.title("🦴 Fracture Detection from X-ray")
st.markdown("Upload an X-ray image, and the model will predict if it's **fractured** or **not fractured**.")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🔍 Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # 🔁 Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 🔮 Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

    # 📢 Display result
    st.subheader("📌 Prediction:")
    st.success(f"**{label.upper()}** ({confidence:.2f}% confidence)")
