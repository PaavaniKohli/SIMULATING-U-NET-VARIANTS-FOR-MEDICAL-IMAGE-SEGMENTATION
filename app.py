import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image

st.set_page_config(layout="wide")
st.title("Medical Image Segmentation using U-Net Variants")

@st.cache_resource
def load_models():
    return {
        "With Augmentation": {
            "U-Net": tf.keras.models.load_model("models/with_aug/UNet_model_256WA.h5", compile=False),
            "U-Net++": tf.keras.models.load_model("models/with_aug/unet_plus_plus.h5", compile=False),
            "Attention U-Net": tf.keras.models.load_model("models/with_aug/Attn_UNet_model_256WA.h5", compile=False),
            "Dense U-Net": tf.keras.models.load_model("models/with_aug/Dense-Unet.h5", compile=False),
            "VGG U-Net": tf.keras.models.load_model("models/with_aug/VGG-UNet_model.h5", compile=False),
            "Double U-Net": tf.keras.models.load_model("models/with_aug/Double_UNet_model_256WA.h5", compile=False),
            "ResNet U-Net": tf.keras.models.load_model("models/with_aug/Res_Unet.h5", compile=False),
        },
        "Without Augmentation": {
            "U-Net": tf.keras.models.load_model("models/without_aug/UNet_model_256.h5", compile=False),
            "U-Net++": tf.keras.models.load_model("models/without_aug/unet_plus_pluswa.h5", compile=False),
            "Attention U-Net": tf.keras.models.load_model("models/without_aug/Attn_UNet_model_256.h5", compile=False),
            "Dense U-Net": tf.keras.models.load_model("models/without_aug/Dense-Unetwa.h5", compile=False),
            "VGG U-Net": tf.keras.models.load_model("models/without_aug/VGG-UNet_modelwa.h5", compile=False),
            "Double U-Net": tf.keras.models.load_model("models/without_aug/Double_UNet_model_256.h5", compile=False),
            "ResNet U-Net": tf.keras.models.load_model("models/without_aug/Res_Unetwa.h5", compile=False),
        }
    }

def dice_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

models = load_models()

aug_choice = st.radio("Choose Data Version", ["With Augmentation", "Without Augmentation"])
model_names = list(models[aug_choice].keys())
selected_model = st.selectbox("Choose Model Variant", model_names)
model = models[aug_choice][selected_model]

use_test_image = st.checkbox("Use a random image from test dataset instead of uploading")

chosen_filename = None

if use_test_image:
    test_files = os.listdir("test_images")
    chosen_filename = random.choice(test_files)
    image_path = os.path.join("test_images", chosen_filename)
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Random Test Image: {chosen_filename}", width=300)
else:
    uploaded_file = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
        chosen_filename = uploaded_file.name  # use filename to check for ground truth

if 'image' in locals():
    input_size = (256, 256)
    image_resized = image.resize(input_size)
    img_array = np.array(image_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_input)[0, :, :, 0]
    pred_mask = (prediction > 0.5).astype(np.uint8) * 255

    col1, col2 = st.columns(2)
    col1.image(pred_mask, caption="Predicted Binary Mask", clamp=True, width=300)

    if chosen_filename:
        mask_path = os.path.join("test_masks", chosen_filename)
        if os.path.exists(mask_path):
            gt_mask = Image.open(mask_path).resize(input_size)
            gt_np = (np.array(gt_mask) > 0).astype(np.uint8)
            col2.image(gt_mask, caption="Ground Truth Mask", clamp=True, width=300)

            pred_np = (prediction > 0.5).astype(np.uint8)
            dice = dice_score(gt_np, pred_np)
            st.metric(label="Dice Score", value=f"{dice:.4f}")
        else:
            col2.warning("Ground truth mask not found for this image.")
