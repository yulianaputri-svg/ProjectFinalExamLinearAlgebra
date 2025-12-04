# pages/convolution_tools.py
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import io
import math

# Try optional rembg (better background removal). If not available, fallback.
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

st.set_option('deprecation.showfileUploaderEncoding', False)


# --------------------------
# HELPERS
# --------------------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype('uint8'), 'RGB')

def ensure_small(img: Image.Image, max_dim=1024):
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w*scale), int(h*scale))
        return img.resize(new_size, Image.LANCZOS), True
    return img, False

def download_image_bytes(pil_img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def show_histogram(np_img):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, col in enumerate(['r', 'g', 'b']):
        ax.hist(np_img[:, :, i].ravel(), bins=64, alpha=0.5, label=col)
    ax.legend()
    st.pyplot(fig)


# --------------------------
# CONVOLUTION (manual, per-channel)
# --------------------------
def manual_convolution(img_np: np.ndarray, kernel: np.ndarray):
    """
    img_np: HxWx3 uint8
    kernel: kxk float
    returns uint8 image
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")
    h, w, c = img_np.shape
    k = kernel.shape[0]
    pad = k // 2
    # pad with edge values (faster than reflect for PIL compatibility)
    padded = np.pad(img_np, ((pad, pad),(pad, pad),(0,0)), mode='edge').astype(np.float32)
    out = np.zeros_like(img_np, dtype=np.float32)

    # Convolution per channel
    for ch in range(c):
        for i in range(h):
            # small micro-optim: slice row blocks
            for j in range(w):
                region = padded[i:i+k, j:j+k, ch]
                out[i, j, ch] = (region * kernel).sum()
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# Fast-ish separable gaussian generator
def gaussian_kernel(k):
    if k % 2 == 0: k += 1
    sigma = 0.3*((k-1)*0.5 - 1) + 0.8
    ax = np.arange(-(k//2), k//2+1, dtype=np.float32)
    gauss = np.exp(-(ax**2)/(2*sigma*sigma))
    gauss = gauss / gauss.sum()
    kern = np.outer(gauss, gauss)
    return kern

# Motion blur kernel (directional)
def motion_blur_kernel(k, angle_deg=0):
    k = int(k) if int(k)%2==1 else int(k)+1
    kern = np.zeros((k,k), dtype=np.float32)
    center = k//2
    # horizontal line then rotate
    kern[center, :] = 1.0 / k
    if angle_deg != 0:
        # rotate using PIL to avoid cv2
        pil_k = Image.fromarray((kern*255).astype('uint8'))
        pil_k = pil_k.rotate(angle_deg, resample=Image.BILINEAR)
        kern = (np.array(pil_k).astype(np.float32))/255.0
        s = kern.sum()
        if s != 0:
            kern = kern / s
    return kern

# Sobel kernels
SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
SOBEL_Y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)

# --------------------------
# IMAGE TRANSFORM HELPERS
# --------------------------
def adjust_brightness_contrast(np_img, brightness=0, contrast=1.0):
    """
    brightness: -100 .. 100 (added)
    contrast: 0.0 .. 3.0 (multiplied)
    """
    img = np_img.astype(np.float32)
    img = (img - 127.5) * contrast + 127.5 + brightness
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_sepia(np_img):
    # sepia transform
    m = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]])
    out = np_img.astype(np.float32) @ m.T
    return np.clip(out, 0, 255).astype(np.uint8)

def pink_tone(np_img):
    r = np_img[:,:,0].astype(np.float32)
    g = np_img[:,:,1].astype(np.float32)
    b = np_img[:,:,2].astype(np.float32)
    r = np.clip(r * 1.15 + 20, 0, 255)
    g = np.clip(g * 1.05 + 5, 0, 255)
    b = np.clip(b * 0.9 - 10, 0, 255)
    return np.stack([r,g,b], axis=2).astype(np.uint8)

def apply_invert(np_img):
    return 255 - np_img

def to_gray(np_img):
    return (0.299*np_img[:,:,0] + 0.587*np_img[:,:,1] + 0.114*np_img[:,:,2]).astype(np.uint8)


# --------------------------
# CARTOON / SKETCH
# --------------------------
def pencil_sketch(np_img, blur_ksize=21):
    gray = to_gray(np_img)
    # invert
    inv = 255 - gray
    # blur via PIL Gaussian
    pil = Image.fromarray(inv)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=(blur_ksize//2)))
    blurred = np.array(pil)
    # dodge blend: result = grayscale / (255 - blurred) * 256
    denom = (255 - blurred).astype(np.float32)
    denom[denom==0] = 1
    sketch = np.clip((gray.astype(np.float32) * 256.0) / denom, 0, 255).astype(np.uint8)
    return np.stack([sketch, sketch, sketch], axis=2)

def cartoonize(np_img, k_smooth=5, edge_thresh=80):
    # smoothing by repeated box-blur (approx bilateral)
    pil = Image.fromarray(np_img)
    for _ in range(max(1, k_smooth//2)):
        pil = pil.filter(ImageFilter.BoxBlur(radius=2))
    smooth = np.array(pil)

    # edges via sobel magnitude
    gx = manual_convolution(np_img, SOBEL_X)
    gy = manual_convolution(np_img, SOBEL_Y)
    # get magnitude from single channel representation
    mag = np.hypot(gx[:,:,0].astype(np.float32), gy[:,:,0].astype(np.float32))
    mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    edge_mask = mag > edge_thresh
    edge_mask = edge_mask.astype(np.uint8) * 255
    # apply mask: where edge, darken; else use smooth
    out = smooth.copy()
    out[edge_mask==255] = (out[edge_mask==255] * 0.2).astype(np.uint8)
    return out


# --------------------------
# BACKGROUND REMOVAL (OPTIONS)
# --------------------------
def remove_bg_rembg(pil_img):
    # returns RGBA PIL image
    try:
        out = rembg_remove(pil_img)
        return out  # already RGBA or RGB with trimmed background
    except Exception as e:
        raise RuntimeError("rembg removal failed: " + str(e))

def remove_bg_hsv(np_img, h_low=0, h_high=180, s_low=0, s_high=60, v_low=200, v_high=255):
    # PIL's HSV uses H:0-255, S:0-255, V:0-255. We'll convert via simple formula using colorscale.
    pil = Image.fromarray(np_img).convert("HSV")
    hsv = np.array(pil)  # H 0-255
    H = (hsv[:,:,0].astype(int) * 180 // 255)  # convert back to 0-180 approx
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    mask = ((H >= h_low) & (H <= h_high) & (S >= s_low) & (S <= s_high) & (V >= v_low) & (V <= v_high))
    # background mask True where background
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([np_img, alpha])
    return Image.fromarray(rgba)

def remove_bg_color_pick(np_img, pick_rgb, tol=30):
    # pick_rgb: tuple (r,g,b)
    diff = np.linalg.norm(np_img.astype(int) - np.array(pick_rgb).astype(int), axis=2)
    mask = diff < tol
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([np_img, alpha])
    return Image.fromarray(rgba)


# --------------------------
# CUSTOM KERNEL PARSER
# --------------------------
def parse_kernel(text: str):
    """
    parse kernel from textarea: rows separated by newline, numbers by spaces
    returns numpy 2D array
    """
    try:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() != ""]
        rows = []
        for ln in lines:
            parts = [float(x) for x in ln.replace(',', ' ').split()]
            rows.append(parts)
        # check square or rectangular kernel
        k = np.array(rows, dtype=np.float32)
        return k
    except Exception as e:
        raise ValueError("Invalid kernel format: " + str(e))


# --------------------------
# MAIN RUN
# --------------------------
def run():
    st.title("🖼️ Image Processing Suite — Full Features 💗")

    uploaded = st.file_uploader("📸 Upload image", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("Upload an image to access features")
        return

    pil_img = Image.open(uploaded)
    pil_img = pil_img.convert("RGB")
    # warn & offer resize for very large images
    pil_small, was_resized = ensure_small(pil_img, max_dim=1024)
    if was_resized:
        st.warning("Image was auto-resized to max dimension 1024 for speed. You can re-upload a smaller image if needed.")
    img_np = pil_to_np(pil_small)

    st.image(pil_small, caption="💖 Original", use_container_width=True)

    # Layout: tabs for full features
    tabs = st.tabs(["Basic Tools", "Filters", "Convolution", "Background Removal", "Extras & Download"])

    # ---------------- Basic Tools ----------------
    with tabs[0]:
        st.header("🔧 Basic Tools")
        col1, col2, col3 = st.columns(3)
        with col1:
            angle = st.slider("Rotate (deg)", -180, 180, 0)
            if st.button("Rotate"):
                pil_img2 = pil_small.rotate(-angle, expand=True)
                st.image(pil_img2, caption=f"Rotated {angle}°", use_container_width=True)
                st.download_button("Download Rotated", data=download_image_bytes(pil_img2), file_name="rotated.png")
        with col2:
            if st.button("Flip Horizontal"):
                pil_img2 = pil_small.transpose(Image.FLIP_LEFT_RIGHT)
                st.image(pil_img2, caption="Flipped Horizontal", use_container_width=True)
                st.download_button("Download", data=download_image_bytes(pil_img2), file_name="flipped_h.png")
            if st.button("Flip Vertical"):
                pil_img2 = pil_small.transpose(Image.FLIP_TOP_BOTTOM)
                st.image(pil_img2, caption="Flipped Vertical", use_container_width=True)
                st.download_button("Download", data=download_image_bytes(pil_img2), file_name="flipped_v.png")
        with col3:
            w, h = pil_small.size
            new_w = st.number_input("Resize width", value=w, min_value=1)
            new_h = st.number_input("Resize height", value=h, min_value=1)
            if st.button("Resize"):
                pil_img2 = pil_small.resize((int(new_w), int(new_h)), resample=Image.LANCZOS)
                st.image(pil_img2, caption="Resized", use_container_width=True)
                st.download_button("Download Resize", data=download_image_bytes(pil_img2), file_name="resized.png")

        # Brightness & Contrast
        st.subheader("Brightness & Contrast")
        bright = st.slider("Brightness (-100..100)", -100, 100, 0)
        contrast = st.slider("Contrast (0.0..3.0)", 0.0, 3.0, 1.0, 0.01)
        if st.button("Apply Brightness/Contrast"):
            out = adjust_brightness_contrast(img_np, brightness=bright, contrast=contrast)
            out_pil = np_to_pil(out)
            st.image(out_pil, caption="Adjusted", use_container_width=True)
            st.download_button("Download Adjusted", data=download_image_bytes(out_pil), file_name="bc_adjusted.png")

    # ---------------- Filters ----------------
    with tabs[1]:
        st.header("🎨 Filters")
        colf1, colf2 = st.columns(2)
        with colf1:
            basic = st.selectbox("Basic Filters", ["None", "Grayscale", "Invert", "Sepia", "Pink-tone", "Sketch", "Cartoon"])
            if basic != "None":
                if basic == "Grayscale":
                    out = np.stack([to_gray(img_np)]*3, axis=2)
                elif basic == "Invert":
                    out = apply_invert(img_np)
                elif basic == "Sepia":
                    out = apply_sepia(img_np)
                elif basic == "Pink-tone":
                    out = pink_tone(img_np)
                elif basic == "Sketch":
                    out = pencil_sketch(img_np, blur_ksize=st.slider("Sketch blur", 5, 51, 21, 2))
                elif basic == "Cartoon":
                    out = cartoonize(img_np, k_smooth=st.slider("Smooth level", 1, 9, 3), edge_thresh=st.slider("Edge thr", 10, 200, 80))
                out_pil = np_to_pil(out)
                st.image(out_pil, caption=f"{basic}", use_container_width=True)
                st.download_button("Download Filtered", data=download_image_bytes(out_pil), file_name=f"{basic}.png")
        with colf2:
            # channel toggle
            st.subheader("Channel Toggle")
            r = st.checkbox("Red", True)
            g = st.checkbox("Green", True)
            b = st.checkbox("Blue", True)
            ch_out = img_np.copy()
            if not r: ch_out[:,:,0] = 0
            if not g: ch_out[:,:,1] = 0
            if not b: ch_out[:,:,2] = 0
            st.image(np_to_pil(ch_out), caption="Channel Toggle", use_container_width=True)
            st.download_button("Download Channel", data=download_image_bytes(np_to_pil(ch_out)), file_name="channel.png")

    # ---------------- Convolution ----------------
    with tabs[2]:
        st.header("🧮 Convolution & Kernel Editor")
        conv_choice = st.selectbox("Convolution Preset", [
            "None", "Box Blur 3x3", "Box Blur 5x5", "Gaussian", "Motion Blur", "Sharpen", "Emboss", "Sobel Edge", "Custom Kernel"
        ])
        preview = None
        if conv_choice == "Box Blur 3x3":
            kernel = np.ones((3,3), dtype=np.float32) / 9.0
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Box Blur 5x5":
            kernel = np.ones((5,5), dtype=np.float32) / 25.0
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Gaussian":
            k = st.slider("Gaussian kernel size (odd)", 3, 25, 7, 2)
            kernel = gaussian_kernel(k)
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Motion Blur":
            k = st.slider("Length", 3, 31, 9, 2)
            ang = st.slider("Angle (deg)", 0, 360, 0)
            kernel = motion_blur_kernel(k, angle_deg=ang)
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Sharpen":
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Emboss":
            kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)
            preview = manual_convolution(img_np, kernel)
        elif conv_choice == "Sobel Edge":
            gx = manual_convolution(img_np, SOBEL_X)
            gy = manual_convolution(img_np, SOBEL_Y)
            mag = np.hypot(gx[:,:,0].astype(float), gy[:,:,0].astype(float))
            mag = (mag / (mag.max()+1e-8) * 255).astype(np.uint8)
            preview = np.stack([mag, mag, mag], axis=2)
        elif conv_choice == "Custom Kernel":
            st.write("Enter kernel rows, numbers separated by spaces or commas. Example:\n`1 0 -1`\\n`1 0 -1`\\n`1 0 -1`")
            txt = st.text_area("Kernel", value="0 -1 0\n-1 5 -1\n0 -1 0", height=120)
            try:
                kernel = parse_kernel(txt)
                if kernel.size > 0:
                    preview = manual_convolution(img_np, kernel)
            except Exception as e:
                st.error("Kernel parse error: " + str(e))

        if preview is not None:
            st.image(np_to_pil(preview), caption="Convolution Result", use_container_width=True)
            st.download_button("Download Convolution", data=download_image_bytes(np_to_pil(preview)), file_name="conv_result.png")

    # ---------------- Background Removal ----------------
    with tabs[3]:
        st.header("🧼 Background Removal (multiple methods)")
        method = st.selectbox("Method", [
            "HSV Threshold (fast)", "Pick Color (approx)", "rembg (optional, best if available)"
        ])
        if method == "rembg (optional, best if available)" and not REMBG_AVAILABLE:
            st.warning("rembg not installed in this environment — fallback to HSV or color pick. If you want rembg, install it in requirements.")
        if method == "HSV Threshold (fast)" or (method.startswith("rembg") and not REMBG_AVAILABLE):
            st.write("Adjust HSV thresholds (Hue 0-180 approx)")
            h_low = st.slider("H low", 0, 180, 0)
            h_high = st.slider("H high", 0, 180, 180)
            s_low = st.slider("S low", 0, 255, 0)
            s_high = st.slider("S high", 0, 255, 60)
            v_low = st.slider("V low", 0, 255, 200)
            v_high = st.slider("V high", 0, 255, 255)
            if st.button("Remove Background (HSV)"):
                out_pil = remove_bg_hsv(img_np, h_low, h_high, s_low, s_high, v_low, v_high)
                st.image(out_pil, caption="Background Removed (HSV)", use_container_width=True)
                st.download_button("Download PNG (alpha)", data=download_image_bytes(out_pil, fmt="PNG"), file_name="bg_removed.png")
        elif method == "Pick Color (approx)":
            pick = st.color_picker("Pick background color", "#ffffff")
            tol = st.slider("Tolerance", 0, 200, 40)
            if st.button("Remove Background (color pick)"):
                rgb = tuple(int(pick[i:i+2], 16) for i in (1,3,5))
                out_pil = remove_bg_color_pick(img_np, rgb, tol=tol)
                st.image(out_pil, caption="Background Removed (Pick Color)", use_container_width=True)
                st.download_button("Download PNG (alpha)", data=download_image_bytes(out_pil, fmt="PNG"), file_name="bg_removed_color.png")
        else:
            # rembg method
            if REMBG_AVAILABLE:
                if st.button("Remove Background with rembg"):
                    try:
                        out = remove_bg_rembg(pil_small if 'pil_small' in locals() else pil_img)
                        st.image(out, caption="Background Removed (rembg)", use_container_width=True)
                        # ensure output is RGBA
                        buf = io.BytesIO()
                        out.save(buf, format="PNG")
                        st.download_button("Download PNG (rembg)", data=buf.getvalue(), file_name="bg_removed_rembg.png")
                    except Exception as e:
                        st.error("rembg failed: " + str(e))
            else:
                st.info("rembg not available in this environment. Add to requirements to use it.")

    # ---------------- Extras & Download ----------------
    with tabs[4]:
        st.header("⚡ Extras & Utilities")
        st.subheader("Histogram & Pixel Inspector")
        if st.button("Show Histogram"):
            show_histogram(img_np)

        h, w = img_np.shape[:2]
        sx = st.slider("Inspect X (row)", 0, h-1, h//2)
        sy = st.slider("Inspect Y (col)", 0, w-1, w//2)
        st.write("Pixel (R,G,B):", tuple(img_np[sx, sy].astype(int)))

        # Before/After blend with any current preview/result - allow user to upload a modified image to compare
        st.subheader("Before / After Blend")
        st.write("If you want to blend original vs result, upload the result image (or re-run tool and download then re-upload).")
        up2 = st.file_uploader("Upload result image to blend (optional)", type=["png","jpg","jpeg"], key="blend")
        if up2:
            pil_res = Image.open(up2).convert("RGBA")
            pil_orig_rgba = pil_small.convert("RGBA")
            alpha = st.slider("Blend alpha (0=orig,1=result)", 0.0, 1.0, 0.5)
            # resize to same
            if pil_res.size != pil_orig_rgba.size:
                pil_res = pil_res.resize(pil_orig_rgba.size, Image.LANCZOS)
            arr_orig = np.array(pil_orig_rgba).astype(float)
            arr_res = np.array(pil_res).astype(float)
            blend = np.clip((1-alpha)*arr_orig + alpha*arr_res, 0, 255).astype(np.uint8)
            st.image(Image.fromarray(blend), caption="Blend Result", use_container_width=True)
            st.download_button("Download Blend", data=download_image_bytes(Image.fromarray(blend)), file_name="blend.png")

        st.markdown("---")
        st.write("✅ All tools above run without OpenCV so this app is deployable to Streamlit Cloud.")
        st.info("If you want rembg background removal on deployment, add `rembg` in requirements.txt and deploy again.")








