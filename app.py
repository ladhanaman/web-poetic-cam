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
# 1. SIDEBAR (Controls Only)
# ==========================================
with st.sidebar:
    st.header("Input Configuration")
    
    # 1. Select Mode
    input_method = st.radio(
        "Source", 
        ["Upload", "Camera"], 
        label_visibility="collapsed",
        key="input_mode" 
    )
    
    # 2. Upload Logic (Stays in Sidebar because it doesn't need width)
    sidebar_upload = None
    if input_method == "Upload":
        sidebar_upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
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

# --- CAMERA HANDLING (The Fix) ---
# We initialize image_source to None
image_source = None

if input_method == "Upload":
    # If upload, grab from sidebar
    image_source = sidebar_upload

elif input_method == "Camera":
    # --- CRITICAL FIX: CAMERA IN MAIN AREA ---
    # We place the camera widget here, spanning the full width of the main column.
    # This forces the browser to request a high-res stream (720p or 1080p).
    
    # We use an expander so the camera closes neatly after you take the photo
    with st.expander("Open Viewfinder", expanded=(st.session_state.last_upload_id is None)):
        camera_shot = st.camera_input("Capture Scene")
        if camera_shot:
            image_source = camera_shot

# --- PROCESSING PIPELINE ---
if image_source:
    
    # Check for new file
    file_id = f"{image_source.name}_{image_source.size}"
    if st.session_state.last_upload_id != file_id:
        st.session_state.narrative = None
        st.session_state.retrieved_items = None
        st.session_state.generated_poem = None
        st.session_state.query_vector = None 
        st.session_state.audio_bytes = None 
        st.session_state.last_upload_id = file_id

    # Layout
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    # --- CARD 1: VISUAL INGESTION ---
    with col1:
        with st.container(border=True):
            st.subheader("I. Ingestion")
            
            # Display Image without stretching it (stops blur if image is small)
            st.image(image_source, use_container_width=True)
            
            # DEBUG: Show actual resolution to verify the fix
            img = Image.open(image_source)
            st.caption(f"Res: {img.size[0]} x {img.size[1]} px")

    # --- CARD 2: INTERNAL MONOLOGUE ---
    # --- CARD 2: INTERNAL MONOLOGUE ---
    with col2:
        with st.container(border=True):
            st.subheader("II. Processing")
            
            # 1. Vision Analysis
            if not st.session_state.narrative:
                with st.status("[SYSTEM] Initializing Vision Pipeline...", expanded=True) as s:
                    st.write("Task: Image Analysis (Llama 3.2 Vision)")
                    # This captures the result (or the error string)
                    result = run_vision_cached(image_source) 
                    st.session_state.narrative = result
                    s.update(label="[SYSTEM] Vision Analysis: Complete", state="complete", expanded=False)
            
            # --- ERROR CHECKING ---
            # Check if narrative is None or empty
            if not st.session_state.narrative:
                st.error("Vision Analysis returned no data. Check logs.")
            
            # Check if narrative contains an Error message
            elif st.session_state.narrative.startswith("ERROR:") or "Error:" in st.session_state.narrative:
                st.error(f"Pipeline Failed: {st.session_state.narrative}")
                st.stop() # Stop execution here so it doesn't try to use bad data
            
            else:
                # Only run this if we have a valid narrative
                st.info(f"**Narrative:** {st.session_state.narrative}")

                # 2. Memory Retrieval
                if not st.session_state.retrieved_items:
                    with st.spinner("Task: Vector Search (Pinecone)..."):
                        
                    st.session_state.retrieved_items = retrieve_poems(st.session_state.narrative)
                    st.session_state.query_vector = get_embedding(st.session_state.narrative)

            # 3. Visualization
            if st.session_state.retrieved_items and MODULES_AVAILABLE:
                st.write("---")
                st.caption("Latent Space Visualization")
                
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
                temperature = st.slider("Model Temperature", 0.1, 1.0, 0.7)
                
                with st.expander("Context Data"):
                    for i, m in enumerate(st.session_state.retrieved_items):
                        meta = m.get('metadata', {})
                        raw_title = meta.get('title', f"{i+1}")
                        clean_text = meta.get('text', "No text.").strip()

                        clean_title = raw_title

                        #if "poem poem" in clean_title.lower():
                            #clean_title = clean_title.lower().replace("poem poem", "Poem").title()

                        #clean_title = clean_title.replace("_", " ").title()


                        st.markdown(f"**{clean_title}**")
                        st.caption(f"{clean_text}") 
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
                            status.write("Task: Audio Synthesis (Google TTS)")
                            audio = AudioEngine()
                            st.session_state.audio_bytes = audio.synthesize(st.session_state.generated_poem)
                        
                        status.update(label="[SYSTEM] Sequence Finished", state="complete", expanded=False)

                # Output Area
                if st.session_state.generated_poem:
                    clean_poem = st.session_state.generated_poem.replace("- ", "— ")
                    st.markdown(
                        f"<div style='text-align: center; font-style: italic; padding: 10px; font-family: serif;'>{clean_poem}</div>", 
                        unsafe_allow_html=True
                    )
    
                    if st.session_state.audio_bytes:
                        byte_size = len(st.session_state.audio_bytes)
                        if byte_size < 1000:
                            st.warning(f"[SYSTEM] Audio blocked (Size: {byte_size} bytes).")
                        else:
                            st.caption(f"Audio Stream: {byte_size / 1024:.1f} KB")
                            st.audio(st.session_state.audio_bytes, format="audio/mpeg")

else:
    st.info("System Idle: Select 'Camera' or 'Upload' to begin.")
