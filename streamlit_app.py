import io
import cv2
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import time
import random
from ultralytics import YOLO
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np



st.set_page_config(page_title="🦷 Oral Disease Classifier", layout="wide")


if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

placeholder = st.empty()


# Intro page
if not st.session_state.intro_done:
    with placeholder.container():
        intro_image_path = "Doctor.jpeg"
        if not os.path.exists(intro_image_path):
            st.error(f"Intro image {intro_image_path} not found!")
            st.stop()
        intro_image = Image.open(intro_image_path)
        intro_image = intro_image.resize(
            (int(intro_image.width * 1.4), int(intro_image.height * 1.4))
        )

        st.markdown(
            """
        <style>
        .intro-text {font-size:65px;color:#007bff;text-align:left;margin-top:40px;position:relative;z-index:2;opacity:1;transition: opacity 1s;}
        .square {width:70px;height:70px;background-color:#007bff;position:absolute;border-radius:1px;opacity:0.3;animation:moveSquare 5s linear infinite;}
        @keyframes moveSquare {0% {transform: translateY(0) translateX(0) rotate(0deg);} 50% {transform: translateY(50px) translateX(30px) rotate(180deg);} 100% {transform: translateY(0) translateX(0) rotate(360deg);}}
        .square-container {position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;overflow:hidden;}
        </style>
        """,
            unsafe_allow_html=True,
        )

        square_html = '<div class="square-container">'
        for i in range(25):
            left = random.randint(0, 90)
            top = random.randint(0, 90)
            delay = random.uniform(0, 7)
            size = random.randint(15, 35)
            square_html += f'<div class="square" style="left:{left}%; top:{top}%; width:{size}px; height:{size}px; animation-delay:{delay}s;"></div>'
        square_html += "</div>"
        st.markdown(square_html, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 4])
        with col1:
            st.markdown(
                '<div class="intro-text" id="intro-text">we care<br>about your dental health</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.image(intro_image, use_container_width=True)

    for i in range(20, -1, -1):
        opacity = i / 20
        st.markdown(
            f"""
        <script>
        document.getElementById("intro-text").style.opacity = "{opacity}";
        </script>
        """,
            unsafe_allow_html=True,
        )
        time.sleep(0.2)

    st.session_state.intro_done = True
    placeholder.empty()

# Main App CSS
st.markdown(
    """
<style>
body {background: #e0f7fa; font-family: 'Arial', sans-serif; overflow-x:hidden;}
footer {visibility:hidden;}
.prob-box {border-radius:10px;padding:10px;margin-bottom:5px;font-weight:bold;color:white;text-align:center;font-size:14px;}
.glow {background: linear-gradient(90deg,#e74c3c,#c0392b);box-shadow:0 0 20px #e74c3c;}
.neutral {background: linear-gradient(90deg,#3498db,#2980b9);}
</style>
""",
    unsafe_allow_html=True,
)


# Upload image
uploaded_file = st.file_uploader("Upload a mouth photo", type=["png", "jpg", "jpeg"])
if uploaded_file is None:
    st.info("Please upload an image of your jaw opened.")
    st.stop()

image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")


# Localization model (image size 1280)
with st.spinner("Running localization model..."):
    localization_model = YOLO("detection-best.pt")
    results = localization_model(
        image, imgsz=1280, conf=0.65, iou=0.3, agnostic_nms=True
    )

boxes = results[0].boxes
detected_diseases = []


# Classification model setup (6-class)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Calculus", "Caries", "Discoloration", "Gingivitis", "Hypodontia", "Ulcer"]

model_path = "best_model.pth"
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found!")
    st.stop()

classification_model = models.convnext_small(weights=None)
classification_model.classifier[2] = nn.Linear(
    classification_model.classifier[2].in_features, len(classes)
)
classification_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
classification_model.to(DEVICE)
classification_model.eval()


# ---------------- GradCAM++ Setup ----------------
def find_last_conv(model):
    last_conv = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
            break
    return last_conv


target_layer = find_last_conv(classification_model)
if target_layer is None:
    raise RuntimeError("No convolution layer found for GradCAM++")

cam = GradCAMPlusPlus(model=classification_model, target_layers=[target_layer])

# Visualization transform (unnormalize to 0–1)
vis_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def predict(img: Image.Image):
    img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classification_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def generate_gradcam(pil_img, target_class=None):
    img_tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)

    # unnormalized image for visualization
    vis_img = vis_transform(pil_img).numpy().transpose(1, 2, 0)
    vis_img = np.clip(vis_img, 0, 1).astype(np.float32)

    with torch.no_grad():
        logits = classification_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    target_idx = pred_idx if target_class is None else target_class

    targets = [ClassifierOutputTarget(target_idx)]
    grayscale_cam = cam(img_tensor, targets=targets)[0]  # HxW heatmap

    heatmap = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
    return heatmap, pred_idx, probs[pred_idx]


# Layout: Image left, results right
col_img, col_res = st.columns([1, 1])

with col_img:
    if len(boxes) > 0:
        detection_labels = [str(i + 1) for i in range(len(boxes))]
        boxed_image = results[0].plot(labels=detection_labels)
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        st.image(
            boxed_image,
            caption="Localization Result (1280px)",
            use_container_width=True,
        )
    else:
        st.image(image, caption="Input Image", use_container_width=True)

    # Grad-CAM++ section
    st.markdown("## Grad-CAM++ Explanation")
    with st.spinner("Generating Grad-CAM++..."):
        heatmap, pred_idx, pred_prob = generate_gradcam(image)
        predicted_class = classes[pred_idx]

        # Resize heatmap to match original image width
        heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

    st.image(
        heatmap_resized, caption="Grad-CAM++ Heatmap Overlay", use_container_width=True
    )


with col_res:
    st.subheader("Detected Diseases")

    # Localization results
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = (
                localization_model.names[cls_id]
                if cls_id in localization_model.names
                else f"Class {cls_id}"
            )
            detected_diseases.append(cls_name)

            st.write(f"**{cls_name} ({conf*100:.1f}%)**")
            st.progress(float(min(max(conf, 0.0), 1.0)))

    # Classification results (6-class) for any remaining disease
    probs = predict(image)
    for i, cls in enumerate(classes):
        if cls not in detected_diseases:
            conf = float(probs[i])
            st.write(f"**{cls} ({conf*100:.1f}%)**")
            st.progress(conf)


# medical advices
medical_advice = {
    "Calculus": [
        "Brush teeth twice daily with fluoride toothpaste.",
        "Use dental floss to remove plaque.",
        "Visit dentist every 6 months for cleaning.",
    ],
    "Caries": [
        "Limit sugary snacks and drinks.",
        "Maintain regular dental checkups.",
        "Consider fluoride treatments if recommended.",
    ],
    "Gingivitis": [
        "Practice proper gum care with gentle brushing.",
        "Use antiseptic mouthwash daily.",
        "Schedule professional dental cleaning.",
    ],
    "Ulcer": [
        "Avoid spicy and acidic foods.",
        "Use oral gels or rinses to relieve pain.",
        "Stay hydrated and maintain good nutrition.",
    ],
    "Discoloration": [
        "Avoid staining foods and drinks.",
        "Brush twice daily and consider whitening treatments.",
        "Visit dentist for professional advice.",
    ],
    "Hypodontia": [
        "Consult a dentist for replacement options.",
        "Maintain oral hygiene to prevent further issues.",
        "Consider orthodontic evaluation if necessary.",
    ],
}

st.write("")
st.markdown("### Medical Advice for Detected Diseases:")

shown_advice = set()
for disease in detected_diseases:
    for advice in medical_advice.get(disease, []):
        if advice not in shown_advice:
            st.write(f"- {advice}")
            shown_advice.add(advice)
