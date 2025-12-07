import streamlit as st
import io
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
st.set_page_config(layout="wide", page_title="Poetic Camera")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 5px; height: 3em; font-family: monospace; }
    h1, h2, h3 { font-family: 'Courier New', Courier, monospace; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHING FUNCTIONS ---

@st.cache_data(show_spinner=False)
def run_vision_cached(image_file):
    try:
        with open("temp_input.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        return analyze_image("temp_input.jpg")
    except Exception as e:
        return f"Error: {e}"

@st.cache_data(show_spinner=False)
def load_universe_vectors():
    """Fetches the 'Universe' background only ONCE per session."""
    print("[SYSTEM] Cache Miss: Fetching Background Universe vectors...")
    try:
        results = retrieve_poems("Life Death Eternity Nature Soul Love Time", top_k=50)
        return [item['values'] for item in results if 'values' in item]
    except Exception:
        return []

# --- Session State ---
keys = ['narrative', 'retrieved_items', 'generated_poem', 'audio_bytes', 'last_upload_id', 'query_vector']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ==========================================
# 1. INPUT ZONE (Sidebar)
# ==========================================
with st.sidebar:
    st.header("Input Configuration")
    
    # FIX: Added key="input_mode" to prevent duplicate ID error
    input_method = st.radio(
        "Source", 
        ["Upload", "Camera"], 
        label_visibility="collapsed",
        key="input_mode" 
    )
    
    image_source = None
    if input_method == "Upload":
        image_source = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    else:
        image_source = st.camera_input("Capture Scene")
    
    st.markdown("---")
    if st.button("System Reset"):
        st.cache_data.clear()
        for k in keys: st.session_state[k] = None
        st.rerun()

# ==========================================
# MAIN LOGIC
# ==========================================
st.title("Poetic Camera")
st.caption("System Status: Online | Mode: Multimodal RAG")

if image_source:
    
    # --- STATE MANAGEMENT ---
    file_id = f"{image_source.name}_{image_source.size}"
    if st.session_state.last_upload_id != file_id:
        st.session_state.narrative = None
        st.session_state.retrieved_items = None
        st.session_state.generated_poem = None
        st.session_state.query_vector = None 
        # Note: Changed 'audio_path' to 'audio_bytes' in session state keys above
        st.session_state.audio_bytes = None 
        st.session_state.last_upload_id = file_id

    # Layout
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    # --- CARD 1: VISUAL INGESTION ---
    with col1:
        with st.container(border=True):
            st.subheader("I. Ingestion")
            st.image(image_source, use_container_width=True)
            st.caption("Status: Image Captured")

    # --- CARD 2: INTERNAL MONOLOGUE ---
    with col2:
        with st.container(border=True):
            st.subheader("II. Processing")
            
            # 1. Vision Analysis
            if not st.session_state.narrative:
                with st.status("[SYSTEM] Initializing Vision Pipeline...", expanded=True) as s:
                    st.write("Task: Image Analysis (Gemini Flash 1.5)")
                    st.session_state.narrative = run_vision_cached(image_source)
                    s.update(label="[SYSTEM] Vision Analysis: Complete", state="complete", expanded=False)
            
            if st.session_state.narrative:
                st.info(f"**Narrative:** {st.session_state.narrative}")

            # 2. Memory Retrieval
            if st.session_state.narrative and not st.session_state.retrieved_items:
                with st.spinner("Task: Vector Search (Pinecone)..."):
                    st.session_state.retrieved_items = retrieve_poems(st.session_state.narrative)
                    st.session_state.query_vector = get_embedding(st.session_state.narrative)

            # 3. Visualization
            if st.session_state.retrieved_items and MODULES_AVAILABLE:
                st.write("---")
                st.caption("Latent Space Visualization")
                
                # FIX: Load universe and pass to visualizer
                universe = load_universe_vectors()
                viz = LatentSpaceVisualizer(background_vectors=universe)
                
                fig = viz.visualize_query_context(
                    st.session_state.query_vector, 
                    st.session_state.retrieved_items
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # --- CARD 3: GENERATIVE INFERENCE ---
    with col3:
        with st.container(border=True):
            st.subheader("III. Output")
            
            if st.session_state.retrieved_items:
                
                st.markdown("#### Parameters")
                
                temperature = st.slider(
                    "Model Temperature", 
                    min_value=0.1, max_value=1.0, value=0.7
                )
                
                with st.expander("Context Data (Retrieval Results)"):
                    for i, m in enumerate(st.session_state.retrieved_items):
                        meta = m.get('metadata', {})
                        title = meta.get('title', f"Poem #{i+1}")
                        clean_text = meta.get('text', "No text.").strip() 
                        st.markdown(f"**{title}**")
                        st.caption(f"_{clean_text}_") 
                        st.divider()

                st.write("---") 

                if st.button("Execute Generation Sequence", type="primary", use_container_width=True):
                    with st.status("[SYSTEM] Running Generation Sequence...", expanded=True) as status:
                        
                        status.write("Task: Text Inference (Llama-3-70b)")
                        st.session_state.generated_poem = generate_poem(
                            st.session_state.narrative,
                            st.session_state.retrieved_items,
                            temperature=temperature
                        )
                        
                        if MODULES_AVAILABLE:
                            status.write("Task: Audio Synthesis (Edge TTS)")
                            audio = AudioEngine()
                            # FIX: Store bytes, not path
                            st.session_state.audio_bytes = audio.synthesize(st.session_state.generated_poem)
                        
                        status.update(label="[SYSTEM] Sequence Finished", state="complete", expanded=False)


                # Output Area
                if st.session_state.generated_poem:
                    clean_poem = st.session_state.generated_poem.replace("- ", "— ")
                    st.markdown(
                        f"<div style='text-align: center; font-style: italic; padding: 10px; font-family: serif;'>{clean_poem}</div>", 
                        unsafe_allow_html=True
                    )
    
                    # DEBUG & PLAYBACK LOGIC
                    if st.session_state.audio_bytes:
                        byte_size = len(st.session_state.audio_bytes)
                        
                        # 1. Sanity Check: Is the file too small? (< 1KB)
                        # If Google blocks us, it sends a tiny text file error instead of audio.
                        if byte_size < 1000:
                            st.warning(f"[SYSTEM] Audio generation blocked (Size: {byte_size} bytes). Provider rejected request.")
                        else:
                            # 2. Success: Play the audio
                            # usage of 'audio/mpeg' is more strictly supported by browsers than 'audio/mp3'
                            st.caption(f"Audio Stream: {byte_size / 1024:.1f} KB")
                            st.audio(st.session_state.audio_bytes, format="audio/mpeg")
                    
                    elif st.session_state.audio_bytes is None and st.session_state.generated_poem:
                        # Audio tried to run but returned Nothing
                        st.warning("Audio Engine returned no data.")

else:
    st.info("System Idle: Waiting for Visual Input.")
