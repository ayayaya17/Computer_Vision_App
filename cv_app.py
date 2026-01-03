import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd

# Step 1: Streamlit config
st.set_page_config(page_title="ResNet18 Image Classifier (CPU)", layout="centered")
st.title("ResNet18 Image Classification (CPU-only)")
st.write("Upload an image → get Top-5 predictions (ImageNet).")

# Step 3: Force CPU
device = torch.device("cpu")

# Step 4: Load pretrained ResNet18 + eval mode
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# Step 5: Recommended preprocessing transforms
preprocess = weights.transforms()

# Step 6: Upload UI
uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 7: Convert to tensor + inference (no gradients)
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # Step 8: Softmax + Top-5
    probs = torch.softmax(outputs, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probs, 5)

    categories = weights.meta["categories"]
    top5_labels = [categories[i] for i in top5_idx.tolist()]
    top5_values = top5_prob.tolist()

    # Show results table
    df = pd.DataFrame({"Class": top5_labels, "Probability": top5_values})
    st.subheader("Top-5 Predictions")
    st.dataframe(df, use_container_width=True)

    # Step 9: Bar chart
    st.subheader("Prediction Probabilities (Top-5)")
    chart_df = df.set_index("Class")
    st.bar_chart(chart_df)

else:
    st.info("Upload an image to start.")
