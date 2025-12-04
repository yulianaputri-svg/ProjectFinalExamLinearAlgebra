import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ================================================================
# FUNCTION: APPLY MATRIX TO IMAGE (INVERSE MAPPING)
# ================================================================
def apply_matrix_image(img_pil, M, out_size=None):
    img = np.array(img_pil)
    h, w = img.shape[:2]

    if out_size is None:
        H, W = h, w
    else:
        H, W = out_size

    out = np.zeros((H, W, 3), dtype=np.uint8)
    Minv = np.linalg.pinv(M)

    for i in range(H):
        for j in range(W):
            src = Minv @ np.array([i, j, 1])
            x, y = int(src[0]), int(src[1])
            if 0 <= x < h and 0 <= y < w:
                out[i, j] = img[x, y]

    return Image.fromarray(out)


# ================================================================
# FUNCTION: APPLY MATRIX TO GRID POINTS
# ================================================================
def transform_points(points, M):
    pts = np.array([[p[0], p[1], 1] for p in points]).T
    res = (M @ pts)[:2].T
    return [(float(a), float(b)) for a, b in res]


# ================================================================
# MAIN PAGE — MATRIX TRANSFORMATION
# ================================================================
def run():
    st.title("📐 Matrix Transformations 💗")

    mode = st.radio("Mode:", ["✨ Image Transformation", "📊 Points / Grid Visualization"])

    # ================================================================
    # IMAGE TRANSFORMATION MODE
    # ================================================================
    if mode == "✨ Image Transformation":
        uploaded = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])
        if not uploaded:
            st.info("Upload an image to start ✨")
            return

        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="💖 Original Image", use_container_width=True)

        trans = st.selectbox(
            "Choose Transformation",
            ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Combined (custom)"]
        )

        M = np.eye(3)

        # TRANSLATION
        if trans == "Translation":
            tx = st.slider("Translate X", -300, 300, 50)
            ty = st.slider("Translate Y", -300, 300, 50)
            M = np.array([[1,0,tx],[0,1,ty],[0,0,1]])

        # SCALING
        elif trans == "Scaling":
            sx = st.slider("Scale X", 0.2, 4.0, 1.5)
            sy = st.slider("Scale Y", 0.2, 4.0, 1.5)
            M = np.array([[sx,0,0],[0,sy,0],[0,0,1]])

        # ROTATION
        elif trans == "Rotation":
            deg = st.slider("Angle (deg)", -180, 180, 30)
            rad = np.radians(deg)
            M = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0,0,1]
            ])

        # SHEARING
        elif trans == "Shearing":
            shx = st.slider("Shear X", -2.0, 2.0, 0.3)
            shy = st.slider("Shear Y", -2.0, 2.0, 0.0)
            M = np.array([[1,shx,0],[shy,1,0],[0,0,1]])

        # REFLECTION
        elif trans == "Reflection":
            axis = st.radio("Axis", ["Horizontal", "Vertical"])
            w, h = img_pil.size
            if axis == "Horizontal":
                M = np.array([[-1,0,w],[0,1,0],[0,0,1]])
            else:
                M = np.array([[1,0,0],[0,-1,h],[0,0,1]])

        # COMBINED
        elif trans == "Combined (custom)":
            tx = st.slider("Translate X", -300, 300, 0)
            ty = st.slider("Translate Y", -300, 300, 0)
            deg = st.slider("Rotate", -180, 180, 0)
            sx = st.slider("Scale", 0.5, 3.0, 1.0)
            sh = st.slider("Shear", -1.0, 1.0, 0.0)

            rad = np.radians(deg)

            # Compose matrices
            T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
            R = np.array([[np.cos(rad), -np.sin(rad),0],[np.sin(rad), np.cos(rad),0],[0,0,1]])
            S = np.array([[sx,0,0],[0,sx,0],[0,0,1]])
            Sh = np.array([[1,sh,0],[0,1,0],[0,0,1]])

            M = T @ R @ S @ Sh

        expand = st.checkbox("Keep original canvas size?", True)
        out_size = (img_pil.height, img_pil.width) if expand else None

        interp = st.slider("Animation (0=origin, 1=full)", 0.0, 1.0, 1.0, 0.01)
        M_interp = (1 - interp) * np.eye(3) + interp * M

        if st.button("💗 Apply Transformation"):
            result = apply_matrix_image(img_pil, M_interp, out_size)
            st.image(result, caption="✨ Transformed Image", use_container_width=True)

            orig = np.array(img_pil).astype(float)
            resA = np.array(result).astype(float)

            if resA.shape != orig.shape:
                resA = np.array(Image.fromarray(resA.astype('uint8')).resize((orig.shape[1], orig.shape[0])))

            alpha = st.slider("Blend (Original → Result)", 0.0, 1.0, 0.5)
            blend = np.clip((1-alpha)*orig + alpha*resA, 0,255).astype('uint8')
            st.image(blend, caption="✨ Blend Preview", use_container_width=True)

            # Pixel inspector
            st.subheader("🔍 Pixel Inspector")
            h, w = orig.shape[:2]
            x = st.slider("X", 0, h-1, h//2)
            y = st.slider("Y", 0, w-1, w//2)
            st.write("Original Pixel:", tuple(orig[x,y].astype(int)))
            st.write("Result Pixel:", tuple(resA[x,y].astype(int)))

            # Histogram
            fig, ax = plt.subplots(figsize=(6,3))
            for i, col in enumerate(["r","g","b"]):
                ax.hist(orig[:,:,i].ravel(), bins=60, alpha=0.4, label=f"Orig-{col}")
                ax.hist(resA[:,:,i].ravel(), bins=60, alpha=0.4, label=f"New-{col}", histtype="step")
            ax.legend()
            st.pyplot(fig)

            # Download
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button("📥 Download", buf.getvalue(), file_name="transformed.png", mime="image/png")


    # ================================================================
    # GRID MODE (FIXED M DEFAULT)
    # ================================================================
    else:
        st.subheader("📊 Grid Visualization")

        n = st.slider("Grid Size", 5, 30, 10)
        pts = [(x, y) for x in range(n) for y in range(n)]

        trans = st.selectbox(
            "Choose Transformation",
            ["Translation","Scaling","Rotation","Shearing","Reflection","Combined"]
        )

        # 💗 FIX: Default matrix so no error occurs
        M = np.eye(3)

        # TRANSLATION
        if trans == "Translation":
            tx = st.slider("Translate X", -n, n, 2)
            ty = st.slider("Translate Y", -n, n, 2)
            M = np.array([[1,0,tx],[0,1,ty],[0,0,1]])

        # SCALING
        elif trans == "Scaling":
            sx = st.slider("Scale X", 0.2,3.0,1.2)
            sy = st.slider("Scale Y", 0.2,3.0,1.2)
            M = np.array([[sx,0,0],[0,sy,0],[0,0,1]])

        # ROTATION
        elif trans == "Rotation":
            deg = st.slider("Angle", -180,180,30)
            rad = np.radians(deg)
            M = np.array([[np.cos(rad), -np.sin(rad),0],
                          [np.sin(rad), np.cos(rad),0],
                          [0,0,1]])

        # SHEARING
        elif trans == "Shearing":
            shx = st.slider("Shear X", -2.0,2.0,0.3)
            shy = st.slider("Shear Y", -2.0,2.0,0.0)
            M = np.array([[1,shx,0],[shy,1,0],[0,0,1]])

        # REFLECTION
        elif trans == "Reflection":
            axis = st.radio("Axis", ["Horizontal","Vertical"])
            if axis=="Horizontal":
                M = np.array([[-1,0,n],[0,1,0],[0,0,1]])
            else:
                M = np.array([[1,0,0],[0,-1,n],[0,0,1]])

        # COMBINED
        elif trans == "Combined":
            tx = st.slider("Translate X", -n, n, 0)
            ty = st.slider("Translate Y", -n, n, 0)
            deg = st.slider("Rotate", -180,180,0)
            sx = st.slider("Scale", 0.5,2.0,1.0)
            sh = st.slider("Shear", -1.0,1.0,0.0)

            rad = np.radians(deg)

            T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
            R = np.array([[np.cos(rad), -np.sin(rad),0],[np.sin(rad), np.cos(rad),0],[0,0,1]])
            S = np.array([[sx,0,0],[0,sx,0],[0,0,1]])
            Sh = np.array([[1,sh,0],[0,1,0],[0,0,1]])

            M = T @ R @ S @ Sh

        # INTERPOLATION
        interp = st.slider("Interpolation", 0.0,1.0,1.0,0.01)
        M_interp = (1-interp)*np.eye(3) + interp*M

        new_pts = transform_points(pts, M_interp)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c='pink', label="Original")
        ax.scatter([p[0] for p in new_pts], [p[1] for p in new_pts], c='red', label="Transformed")

        for a,b in zip(pts,new_pts):
            ax.plot([a[0],b[0]], [a[1],b[1]], color='gray', linewidth=0.5)

        ax.legend()
        ax.grid(True)
        st.pyplot(fig)




