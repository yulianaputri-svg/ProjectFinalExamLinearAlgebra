import streamlit as st

# 🌸 Pink CSS Style
pink_style = """
<style>
/* Page background */
.main {
    background-color: #ffe9f3 !important;
}

/* Title color */
h1 {
    color: #d63384 !important;
    font-weight: 900;
}

/* Team card */
.team-card {
    background-color: #ffe3ee;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 10px rgba(255, 128, 172, 0.3);
    border-left: 5px solid #ff4dab;
}

/* Name */
.team-name {
    font-size: 22px;
    font-weight: 800;
    color: #d63384;
}

/* Role */
.team-role {
    font-size: 16px;
    color: #b4005a;
    font-weight: 600;
}
</style>
"""

st.markdown(pink_style, unsafe_allow_html=True)

def run():
    st.title("💗 Meet Our Team 🎀")

    members = [
        ("Zahara Amalia", "Make a Report and Design", "assets/zahara.png"),
        ("Yuliana Bella", "Developer & Programmer", "assets/bella.png"),
        ("Kayla Zavanna", "Make a Report and Research Literature Review", "assets/kayla.png"),
        ("Andhika Ramadhan", "Testing & Developer", "assets/andhika.png")
    ]

    for name, role, photo in members:
        st.markdown("<div class='team-card'>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(photo, width=160)

        with col2:
            st.markdown(f"<div class='team-name'>✨ {name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='team-role'>🎀 {role}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

