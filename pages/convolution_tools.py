import streamlit as st
import numpy as np
from PIL import Image

# Manual convolution (NumPy only)
def manual_convolution(image_array, kernel):
    h, w, c = image_array.shape
    kh, kw = kernel.shape

    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image_array, ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode="edge")

    output = np.zeros_like(image_array, dtype=np.float32)

    for x in range(h):
        for y in range(w):
            region = padded[x:x+kh, y:y+kw]
            for ch in range(3):
                output[x, y, ch] = np.sum(region[:,:,ch] * kernel)

    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def run():
    st.title("✨ Blur & Sharpen Tools (No OpenCV) 💗")

    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if not uploaded:
        st.info("Upload an image to continue 💗")
        return

    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Original Image", use_container_width=True)

    option = st.selectbox("Choose Filter", [
        "Blur 3x3",
        "Blur 5x5",
        "Sharpen",
        "Edge Detection",
        "Emboss"
    ])

    if option == "Blur 3x3":
        kernel = np.ones((3,3))/9

    elif option == "Blur 5x5":
        kernel = np.ones((5,5))/25

    elif option == "Sharpen":
        kernel = np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ])

    elif option == "Edge Detection":
        kernel = np.array([
            [-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]
        ])

    elif option == "Emboss":
        kernel = np.array([
            [-2,-1,0],
            [-1,1,1],
            [0,1,2]
        ])

    if st.button("Apply Filter 💖"):
        result = manual_convolution(img_np, kernel)
        result_img = Image.fromarray(result)

        st.image(result_img, caption="Filtered Image", use_container_width=True)

        # Download
        st.download_button(
            "Download Result",
            data=result_img.tobytes(),
            file_name="result.png",
            mime="image/png",
        )





