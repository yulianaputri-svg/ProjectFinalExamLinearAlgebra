import streamlit as st

# --- Pink Theme CSS ---
pink_style = """
<style>
.main { background-color: #fff0f6 !important; }
[data-testid="stSidebar"] { background-color: #ffdce6 !important; }
h1, h2, h3 { color: #d63384 !important; font-weight: 800 !important; }
.stButton>button {
    background-color: #ff4dab !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 8px 18px !important;
    border: none !important;
}
.stAlert { background-color: #ffe3ee !important; border-left: 6px solid #ff4dab !important; }
</style>
"""
st.markdown(pink_style, unsafe_allow_html=True)

# --- Imports ---
import pages.matrix_tools as matrix_tools
import pages.convolution_tools as convolution_tools
import pages.team as team

# --- Sidebar ---
st.sidebar.title("🎀 Pink Navigation Menu")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📐 Matrix Transformations", "🖼️ Image Processing Tools", "👥 Team Members"]
)

# --- Routing ---
if page == "🏠 Home":
    st.title("💗 Matrix-Based Image Processing — Final Project")
    st.header("📘 Overview")
    st.write("""
    ✨ Image Transformation using **Matrix Operations**  
    ✨ Convolution Filters (manual & advanced)  
    ✨ Background Removal (HSV, Color Picker, GrabCut)  
    ✨ Edge Detection (Sobel, Canny)  
    ✨ Morphology Tools  
    ✨ Histogram + Pixel Inspector  
    ✨ WOW Features & Pink Theme  
    """)
    st.info("💡 Use the sidebar menu to switch pages.")

elif page == "📐 Matrix Transformations":
    matrix_tools.run()

elif page == "🖼️ Image Processing Tools":
    convolution_tools.run()

elif page == "👥 Team Members":
    team.run()
