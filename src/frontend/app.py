import streamlit as st
from streamlit_ace import st_ace
import requests
import os
import json
import pandas as pd

if 'init_config' not in st.session_state:
    st.set_page_config(page_title="Q-SOLV Math AI",
                       layout="wide", initial_sidebar_state="expanded")
    st.session_state.init_config = True


@st.cache_data(ttl=5)
def get_backend_status(url):
    try:
        return requests.get(url, timeout=1).json()
    except:
        return None


st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .stTextArea textarea { background-color: #262730 !important; color: #fafafa !important; }
        [data-testid="stSidebar"] { background-color: #161b22 !important; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🚀 Q-SOLV Menu")

page = st.sidebar.radio(
    "Go to:", ["IDE Playground", "Evaluation Results", "Pipeline & Architecture"])

BACKEND_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------
# PAGE 1: IDE PLAYGROUND
# ---------------------------------------------------------
if page == "IDE Playground":
    st.title("💠 Q-SOLV: Solve Math with Qwen-coder")
    col_desc, col_links = st.columns([2, 1])

    with col_desc:
        st.markdown("""
        **Q-SOLV** is a mathematical reasoning system that synergizes 
        **Qwen2.5-Coder-7B-Instruct** with the **Program-of-Thought (PoT)**. 
        Instead of direct prediction, the AI architects executable Python logic 
        to ensure deterministic accuracy for complex word problems.
    """)
    with col_links:
        st.markdown("""
        [![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github)](https://github.com/lhldanh/Q-Solv)
        [![Model](https://img.shields.io/badge/Model-QSolv--Qwen2.5--7B--LoRA-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k)
    """)

    st.divider()
    col_input, col_editor = st.columns(2)

    with col_input:
        st.subheader("📝 Problem Input")
        user_prompt = st.text_area("Question:", height=150, placeholder="Input your math question here...",
                                   key="input_prompt", label_visibility="collapsed")

        if st.button("🚀 Solve", use_container_width=True):
            if user_prompt:
                with st.spinner("Qwen is coding..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/generate",
                            json={"question": user_prompt},
                            timeout=60
                        )
                        res.raise_for_status()
                        data = res.json()
                        st.session_state.generated_code = data["generated_code"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Connection Error: {e}")
            else:
                st.warning("Please enter a question!")

    with col_editor:
        st.subheader("🐍 Python Code Editor")
        initial_code = st.session_state.get(
            "generated_code", "# Code will be generated here...")

        import hashlib
        editor_key = f"ace_editor_{hashlib.md5(initial_code.encode()).hexdigest()[:8]}"

        code = st_ace(value=initial_code, language="python",
                      theme="monokai", height=350, key=editor_key, auto_update=True)

        if st.button("⚙️ Execute", use_container_width=True):
            with st.spinner("Executing..."):
                try:
                    exec_res = requests.post(
                        f"{BACKEND_URL}/execute", json={"code": code})
                    result = exec_res.json()["execution_result"]
                    if result["error"]:
                        st.error(f"**Error:**\n{result['error']}")
                    else:
                        st.success(f"**Output:**\n{result['logs']}")
                except:
                    st.error("Execution failed.")

# ---------------------------------------------------------
# PAGE 2: EVALUATION RESULTS
# ---------------------------------------------------------

elif page == "Evaluation Results":
    st.title("📊 Model Evaluation & Comparison")
    IMG_DIR = "images"
    REPORT_PATH = os.path.join(IMG_DIR, "compare_report.json")

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
        m1, m2, m3 = st.columns(3)
        m1.metric("Base Accuracy", f"{report['base_metrics']['accuracy']:.1%}")
        m2.metric("Adapter Accuracy",
                  f"{report['adapter_metrics']['accuracy']:.1%}")
        m3.metric("Improvement", f"+{report['improvement']:.1%}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        pie_path = os.path.join(IMG_DIR, "comparison_pie.svg")
        if os.path.exists(pie_path):
            st.image(pie_path, caption="Error Structure Comparison")
    with c2:
        bar_path = os.path.join(IMG_DIR, "comparison_bar.svg")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="Direct Performance Comparison")

# ---------------------------------------------------------
# PAGE 3: PIPELINE & ARCHITECTURE
# ---------------------------------------------------------

elif page == "Pipeline & Architecture":
    st.title("🔗 System Pipeline & Architecture")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    SVG_PATH = os.path.join(root_dir, "images", "Pipeline _Architecture.svg")

    if os.path.exists(SVG_PATH):
        st.image(SVG_PATH, use_container_width=True)
        with open(SVG_PATH, "rb") as f:
            st.download_button("📥 Download SVG", f,
                               "Pipeline_Architecture.svg", "image/svg+xml")
    else:
        st.info(f"File not found: {SVG_PATH}")

# --- STATUS FOOTER  ---
st.sidebar.divider()
status = get_backend_status(f"{BACKEND_URL}/")
if status:
    st.sidebar.caption(f"🟢 Backend: Online ({status['device'].upper()})")
else:
    st.sidebar.caption("🔴 Backend: Offline")
