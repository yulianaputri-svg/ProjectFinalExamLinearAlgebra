import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# manual convolution
def manual_convolution(img, kernel):
    h, w, c = img.shape
    k = kernel.shape[0]
    pad = k // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    out = np.zeros_like(img)

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = img_pad[i:i+k, j:j+k, ch]
                out[i,j,ch] = np.clip(np.sum(region * kernel), 0, 255)

    return out

def gaussian_kernel(k):
    ax = cv2.getGaussianKernel(k,0)
    return ax @ ax.T

def motion_blur_kernel(k, angle):
    kern = np.zeros((k,k))
    kern[k//2,:] = 1.0 / k
    M = cv2.getRotationMatrix2D((k/2-0.5, k/2-0.5), angle, 1)
    return cv2.warpAffine(kern, M, (k,k))

def run():
    st.title("🖼️ Image Processing Tools 💗")

    uploaded = st.file_uploader("📸 Upload Image", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("Upload an image first 💗")
        return

    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)
    st.image(img, caption="💖 Original", use_container_width=True)

    feature = st.selectbox("✨ Select Feature", [
        "Blur (Manual)", "Gaussian Blur", "Motion Blur", "Sharpen (Manual)",
        "Emboss", "Sketch / Outline", "Invert / Negative", "Sepia",
        "Pink-tone Filter", "Channel Toggle", "Sobel Edge", "Canny Edge",
        "Erosion", "Dilation", "Opening", "Closing",
        "Background Removal (HSV threshold)",
        "Background Removal (Pick Color)",
        "Background Removal (GrabCut)"
    ])

    result = img.copy()

    # -------- Processing --------

    if feature == "Blur (Manual)":
        k = st.slider("Kernel Size", 3, 31, 5, 2)
        kernel = np.ones((k,k))/ (k*k)
        result = manual_convolution(img, kernel)

    elif feature == "Gaussian Blur":
        k = st.slider("Kernel Size", 3, 31, 7, 2)
        kernel = gaussian_kernel(k)
        result = manual_convolution(img, kernel)

    elif feature == "Motion Blur":
        k = st.slider("Kernel Size", 3, 31, 9, 2)
        ang = st.slider("Angle", 0, 360, 0)
        kernel = motion_blur_kernel(k, ang)
        result = manual_convolution(img, kernel)

    elif feature == "Sharpen (Manual)":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        result = manual_convolution(img, kernel)

    elif feature == "Emboss":
        kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        result = manual_convolution(img, kernel)
        result = np.clip(result + 128, 0, 255)

    elif feature == "Sketch / Outline":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv,(21,21),0)
        sketch = cv2.divide(gray,255-blur,scale=256)
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    elif feature == "Invert / Negative":
        result = 255 - img

    elif feature == "Sepia":
        k = np.array([[0.393,0.769,0.189],
                      [0.349,0.686,0.168],
                      [0.272,0.534,0.131]])
        res = img @ k.T
        result = np.clip(res,0,255).astype('uint8')

    elif feature == "Pink-tone Filter":
        r = img[:,:,0]*1.2 + 20
        g = img[:,:,1]*1.05 + 5
        b = img[:,:,2]*0.9 - 10
        result = np.stack([
            np.clip(r,0,255),
            np.clip(g,0,255),
            np.clip(b,0,255)
        ],axis=2).astype('uint8')

    elif feature == "Channel Toggle":
        r = st.checkbox("Red", True)
        g = st.checkbox("Green", True)
        b = st.checkbox("Blue", True)
        result = img.copy()
        if not r: result[:,:,0] = 0
        if not g: result[:,:,1] = 0
        if not b: result[:,:,2] = 0

    elif feature == "Sobel Edge":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1,0)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0,1)
        sob = np.hypot(sx,sy)
        sob = np.uint8(255 * sob / sob.max())
        result = cv2.cvtColor(sob, cv2.COLOR_GRAY2RGB)

    elif feature == "Canny Edge":
        t1 = st.slider("Lower", 10,200,50)
        t2 = st.slider("Upper", 20,300,150)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray,t1,t2)
        result = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

    elif feature in ["Erosion","Dilation","Opening","Closing"]:
        k = st.slider("Kernel", 1,20,3)
        it = st.slider("Iterations",1,10,1)
        kern = np.ones((k,k),np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if feature=="Erosion": op = cv2.erode(gray,kern,iterations=it)
        elif feature=="Dilation": op = cv2.dilate(gray,kern,iterations=it)
        elif feature=="Opening": op = cv2.morphologyEx(gray, cv2.MORPH_OPEN,kern,iterations=it)
        else: op = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,kern,iterations=it)

        result = cv2.cvtColor(op, cv2.COLOR_GRAY2RGB)

    elif feature == "Background Removal (HSV threshold)":
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        h_low = st.slider("H low",0,179,0)
        h_high = st.slider("H high",0,179,179)
        s_low = st.slider("S low",0,255,0)
        s_high = st.slider("S high",0,255,60)
        v_low = st.slider("V low",0,255,200)
        v_high = st.slider("V high",0,255,255)

        mask = cv2.inRange(hsv, (h_low,s_low,v_low), (h_high,s_high,v_high))
        result = cv2.bitwise_and(img,img,mask=255-mask)

    elif feature == "Background Removal (Pick Color)":
        pick = st.color_picker("Pick bg color","#ffffff")
        rgb = np.array([int(pick[i:i+2],16) for i in (1,3,5)])
        hsv_val = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0,0]

        tol = st.slider("Tolerance",0,100,30)
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        dist = np.sqrt(np.sum((hsv-hsv_val)**2,axis=2))
        mask = (dist < tol)*255
        result = cv2.bitwise_and(img,img,mask=255-mask.astype('uint8'))

    elif feature == "Background Removal (GrabCut)":
        h,w = img.shape[:2]
        x = st.slider("x",0,w-1,w//8)
        y = st.slider("y",0,h-1,h//8)
        rw = st.slider("width",10,w,w//2)
        rh = st.slider("height",10,h,h//2)
        rect = (x,y,rw,rh)

        mask = np.zeros((h,w),np.uint8)
        bg = np.zeros((1,65),np.float64)
        fg = np.zeros((1,65),np.float64)

        try:
            cv2.grabCut(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),
                        mask,rect,bg,fg,5,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==0)|(mask==2),0,1).astype('uint8')
            result = img * mask2[:,:,None]
        except:
            st.error("GrabCut error, adjust rectangle.")


    # -------- Display result --------
    st.image(result, caption="✨ Result", use_container_width=True)

    alpha = st.slider("Blend (Original → Result)", 0.0,1.0,0.5)
    orig = img.astype(float)
    resA = result.astype(float)
    if resA.shape != orig.shape:
        resA = cv2.resize(resA, (orig.shape[1],orig.shape[0]))
    blend = np.clip((1-alpha)*orig + alpha*resA, 0,255).astype('uint8')

    st.image(blend, caption="✨ Blend Comparison", use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    st.download_button("📥 Download Result", buf.getvalue(),
                       file_name="result.png", mime="image/png")



