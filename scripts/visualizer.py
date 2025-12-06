import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

# We import the retriever here to fetch data internally
from scripts.retriever import retrieve_poems

# --- PERFORMANCE FIX ---
# We use Streamlit caching HERE so the visualizer doesn't re-fetch 
# the universe 3 times. It happens once and stays in memory.
@st.cache_data(show_spinner=False)
def fetch_universe_vectors():
    print("[SYSTEM] Visualizer: Loading Universe (Cached)...")
    try:
        results = retrieve_poems("Life Death Eternity Nature Soul Love Time", top_k=50)
        return [item['values'] for item in results if 'values' in item]
    except Exception as e:
        print(f"[ERROR] Universe Load Failed: {e}")
        return []

class LatentSpaceVisualizer:
    def __init__(self):
        """
        Initializes the math engine.
        Since app.py doesn't pass data, we load it here using the cached helper.
        """
        self.pca = PCA(n_components=3)
        self.scaler = StandardScaler() # Keeps the red dot close to the blue dots
        self.background_vectors = fetch_universe_vectors() # <--- Internal Load

    def visualize_query_context(
        self, 
        query_vector: List[float], 
        retrieved_items: List[Dict[str, Any]]
    ) -> Any:
        
        vectors = []
        labels = []
        types = [] 
        sizes = []

        # 1. Load Universe (Background)
        for vec in self.background_vectors:
            vectors.append(vec)
            labels.append("Latent Background")
            types.append("Universe")
            sizes.append(3) 

        # 2. Load Memories (Matches)
        for item in retrieved_items:
            if 'values' in item:
                vectors.append(item['values'])
                meta = item.get('metadata', {})
                labels.append(meta.get('title', 'Match'))
                types.append("Memory")
                sizes.append(10)

        # 3. Load Sensation (Query)
        vectors.append(query_vector)
        labels.append("Your Vision")
        types.append("Sensation")
        sizes.append(15)

        # --- THE MATH ---
        X = np.array(vectors)
        
        # Safety Check: Need at least 3 points for 3D PCA
        if len(X) < 3: 
            return None

        # STEP 1: NORMALIZE (Squash outliers so dots are closer)
        X_scaled = self.scaler.fit_transform(X)

        # STEP 2: REDUCE (PCA)
        X_embedded = self.pca.fit_transform(X_scaled)

        # --- PLOTTING ---
        df = pd.DataFrame(X_embedded, columns=['x', 'y', 'z'])
        df['Label'] = labels
        df['Type'] = types
        df['Size'] = sizes

        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='Type', size='Size', hover_name='Label',
            title="Semantic Position (Normalized)",
            color_discrete_map={
                "Universe": "#2c2f33",    # Dark Grey
                "Memory": "#0068C9",      # Blue
                "Sensation": "#FF4B4B"    # Red
            },
            opacity=0.8
        )
        
        # Clean UI: Hide axes
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False), 
                zaxis=dict(visible=False)
            )
        )
        return fig