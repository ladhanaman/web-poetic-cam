import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

class LatentSpaceVisualizer:
    def __init__(self, background_vectors: List[List[float]] = None):
        """
        Updated to accept 'background_vectors' passed from app.py.
        """
        self.pca = PCA(n_components=3)
        self.scaler = StandardScaler() 
        # We use the data passed from the app, or an empty list as fallback
        self.background_vectors = background_vectors if background_vectors else []

    def visualize_query_context(
        self, 
        query_vector: List[float], 
        retrieved_items: List[Dict[str, Any]]
    ) -> Any:
        
        vectors = []
        labels = []
        types = [] 
        sizes = []

        # 1. Load Universe (from the passed data)
        for vec in self.background_vectors:
            vectors.append(vec)
            labels.append("Latent Background")
            types.append("Universe")
            sizes.append(3) 

        # 2. Load Memories (from Pinecone results)
        for item in retrieved_items:
            if 'values' in item:
                vectors.append(item['values'])
                meta = item.get('metadata', {})
                labels.append(meta.get('title', 'Match'))
                types.append("Memory")
                sizes.append(10)

        # 3. Load Sensation (User Input)
        vectors.append(query_vector)
        labels.append("Your Vision")
        types.append("Sensation")
        sizes.append(15)

        # --- MATH ENGINE ---
        X = np.array(vectors)
        
        # Safety Check
        if len(X) < 3: return None

        # Normalize & Reduce
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_embedded = self.pca.fit_transform(X_scaled)
        except Exception as e:
            print(f"PCA Error: {e}")
            return None

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
                "Universe": "#2c2f33",
                "Memory": "#0068C9",
                "Sensation": "#FF4B4B"
            },
            opacity=0.8
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
        )
        return fig
