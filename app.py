import streamlit as st
import os
import time
from PIL import Image

# --- Imports ---
from scripts.vision_client import analyze_image
from scripts.retriever import retrieve_poems, get_embedding
from scripts.generator import generate_poem

# Safe Import for Modules
try:
    from scripts.visualizer import LatentSpaceVisualizer
    from scripts.audio import AudioEngine
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# --- Config ---
st.set_page_config(layout="wide", page_title="Poetic camera")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 5px; height: 3em; }
    
    /* Make headers stand out */
    h2 { border-bottom: 2px solid #238636; padding-bottom: 10px; }
    
    /* Hide the default Streamlit menu for immersion */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Caching ---
@st.cache_data(show_spinner=False)
def run_vision_cached(image_file):
    try:
        with open("temp_input.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        return analyze_image("temp_input.jpg")
    except Exception as e:
        return f"Error: {e}"

# --- Session State ---
keys = ['narrative', 'retrieved_items', 'generated_poem', 'audio_path', 'last_upload_id', 'query_vector']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# 1. INPUT ZONE (Sidebar)

with st.sidebar:
    st.header("Get me some clicks of your reality")
    input_method = st.radio("Source", ["Upload", "Camera"], label_visibility="collapsed")
    
    image_source = None
    if input_method == "Upload":
        image_source = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    else:
        image_source = st.camera_input("Capture Scene")
    
    st.markdown("---")
    if st.button("Reset System"):
        st.cache_data.clear()
        for k in keys: st.session_state[k] = None
        st.rerun()


# MAIN LOGIC

st.title("Poetic Camera")
st.caption("Glance your reality through the eyes of Emily Dickinson.")

if image_source:
    
    # --- STATE MANAGEMENT ---
    file_id = f"{image_source.name}_{image_source.size}"
    if st.session_state.last_upload_id != file_id:
        st.session_state.narrative = None
        st.session_state.retrieved_items = None
        st.session_state.generated_poem = None
        st.session_state.query_vector = None 
        st.session_state.last_upload_id = file_id

    # Layout: We add 'gap="medium"' for breathing room
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    # --- CARD 1: STIMULUS ---
    with col1:
        # VISUAL SEPARATION: Everything goes inside a bordered container
        with st.container(border=True):
            st.subheader("Visual Ingestion")
            st.image(image_source, use_container_width=True)
            st.caption("Raw Input")

    # --- CARD 2: ANALYSIS ---
    with col2:
        with st.container(border=True):
            st.subheader("Internal Monologue")
            
            # 1. Vision Analysis
            if not st.session_state.narrative:
                with st.status("Analyzing visual data...", expanded=True) as s:
                    st.session_state.narrative = run_vision_cached(image_source)
                    s.update(label="Vision Processed", state="complete", expanded=False)
            
            if st.session_state.narrative:
                st.info(f"**Seen:** {st.session_state.narrative}")

            # 2. Memory Retrieval
            if st.session_state.narrative and not st.session_state.retrieved_items:
                with st.spinner("Accessing Pinecone Memory..."):
                    st.session_state.retrieved_items = retrieve_poems(st.session_state.narrative)
                    st.session_state.query_vector = get_embedding(st.session_state.narrative)


            if st.session_state.retrieved_items and MODULES_AVAILABLE:
                st.write("---")
                st.caption("Semantic Position (Latent Space)")
                
                # Initialize with default background file
                # Make sure "data/dickinson_metadata_dense.json" exists!
                viz = LatentSpaceVisualizer() 
                
                fig = viz.visualize_query_context(
                    st.session_state.query_vector, 
                    st.session_state.retrieved_items
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- CARD 3: SYNTHESIS ---
    with col3:
        with st.container(border=True):
            st.subheader("Generative Inference")
            
            if st.session_state.retrieved_items:
                
                # We nest another container for the controls to make them look like a "Control Panel"
                st.markdown("#### Configuration")
                
                temperature = st.slider(
                    "Creativity (Temperature)", 
                    min_value=0.1, max_value=1.0, value=0.7
                )
                
                with st.expander("Review Source Inspirations"):
                    for i, m in enumerate(st.session_state.retrieved_items):
                        meta = m.get('metadata', {})
                        title = meta.get('title', f"Poem #{i+1}")
                        
                        # FIX: Strip whitespace so formatting doesn't break
                        raw_text = meta.get('text', "No text found.")
                        clean_text = raw_text.strip() 

                        st.markdown(f"**{title}**")
                        # Use * for italics, and ensure no spaces between * and text
                        st.caption(f"*{clean_text}*") 
                        st.divider()

                st.write("---") # Visual separator before the big button

                if st.button("Generate Poem", type="primary", use_container_width=True):
                    with st.status("Composing...", expanded=True) as status:
                        status.write("Llama 3 is writing...")
                        st.session_state.generated_poem = generate_poem(
                            st.session_state.narrative,
                            st.session_state.retrieved_items,
                            temperature=temperature
                        )
                        
                        if MODULES_AVAILABLE:
                            status.write("Synthesizing voice...")
                            audio = AudioEngine()
                            st.session_state.audio_path = audio.synthesize(st.session_state.generated_poem)
                        
                        status.update(label="Complete!", state="complete", expanded=False)

                # Output Area
                
                if st.session_state.generated_poem:
                    # 1. Replace standard hyphens with Em-dashes to prevent bullet points
                    clean_poem = st.session_state.generated_poem.replace("- ", "— ")
    
                    st.markdown(
                        f"<div style='text-align: center; font-style: italic;'>{clean_poem}</div>", 
                        unsafe_allow_html=True
                    )
    
                    # Audio player remains below
                    if st.session_state.audio_path:
                        st.audio(st.session_state.audio_path)
                    
                    

else:
    # Empty State
    st.info("Waiting for input... Upload a photo to begin.")