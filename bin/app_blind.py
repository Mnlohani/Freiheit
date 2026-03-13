import streamlit as st

st.set_page_config(page_title="Two Button Test")

st.markdown("""
<style>
    .stApp { background: #000000; }
    #MainMenu, footer, header { visibility: hidden; }

    /* FIRST button on page = full screen invisible */
    section.main div.stButton:nth-of-type(1) > button {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: transparent !important;
        border: none !important;
        z-index: 1000 !important;
        opacity: 0 !important;
        cursor: pointer !important;
    }

    /* SECOND button on page = small visible at bottom */
    section.main div.stButton:nth-of-type(2) > button {
        position: fixed !important;
        bottom: 3vh !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: auto !important;
        height: auto !important;
        background: #222222 !important;
        color: #AAAAAA !important;
        border: 1px solid #444444 !important;
        border-radius: 2rem !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        z-index: 1001 !important;
        opacity: 1 !important;
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

# Black screen text
st.markdown("""
<div style="
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: #000000; color: #FFD700;
    font-size: 2rem;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    z-index: 999;">
    👁️
    <br>Tap anywhere to confirm
</div>
""", unsafe_allow_html=True)

# Button 1 — full screen invisible
if st.button("confirm", key="confirm_lang"):
    st.markdown("""
    <div style="
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: #000000; color: #FFD700;
        font-size: 2rem;
        display: flex; align-items: center;
        justify-content: center;
        z-index: 2000;">
        ✅ Language Confirmed!
    </div>
    """, unsafe_allow_html=True)

# Button 2 — small visible at bottom
if st.button("🌐 Change Language", key="change_lang"):
    st.markdown("""
    <div style="
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: #000000; color: #FFD700;
        font-size: 2rem;
        display: flex; align-items: center;
        justify-content: center;
        z-index: 2000;">
        🎤 Say your language
    </div>
    """, unsafe_allow_html=True)