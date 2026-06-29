import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# --- ORGANIC WARM THEME CUSTOM CSS INJECTION ---
st.set_page_config(page_title="Semantic Similarity Dashboard", layout="wide")
st.markdown("""
    <style>
    :root {
        --bg-color: #FDFBF7;
        --card-bg: #F5EFEB;
        --text-color: #3E3631;
        --accent-color: #C97A63;
        --secondary-accent: #7D8F74;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: var(--card-bg);
    }
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
        background-color: transparent;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-color);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-color);
        border-bottom-color: var(--accent-color);
    }
    .critical-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid var(--accent-color);
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .critical-title {
        font-weight: bold;
        color: var(--accent-color);
        text-transform: uppercase;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    # Free, open-source, lightweight (~90MB) NLP model running locally
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- SIDEBAR MANAGEMENT & NAVIGATION ---
st.sidebar.markdown("<h2 style='color: #C97A63;'>🌿 Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Modify the text parameters below to compute real-time structural similarities.")

default_texts = (
    "The cat restfully napped on the warm rug.\n"
    "A feline slept peacefully on the cozy carpet.\n"
    "Artificial intelligence models are transforming modern industries.\n"
    "Advanced machine learning systems redesign corporate ecosystems."
)

text_input = st.sidebar.text_area(
    "Enter sentences (one per line, minimum 3):",
    value=default_texts,
    height=200
)

sentences = [line.strip() for line in text_input.split("\n") if line.strip()]

# --- MAIN DASHBOARD INTERFACE ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Semantic Word & Sentence Similarity Dashboard</h1>", unsafe_allow_html=True)

if len(sentences) < 2:
    st.error("Please enter at least two distinct sentences to construct similarity matrices.")
else:
    # Generate dense embeddings directly without manual preprocessing
    embeddings = model.encode(sentences)
    
    # Calculate Pairwise Cosine Similarity Matrix
    # Dot product of normalized vectors equals cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0) # Correct float precision bounds
    
    # Dashboard Layout Tabs
    tab1, tab2 = st.tabs(["📊 Dynamic Visual Analytics", "🧠 Paul's Critical Thinking Metrics"])
    
    with tab1:
        st.markdown("### Model Similarity Configurations")
        
        # Graph 1: Pairwise Heatmap
        fig_heat = px.imshow(
            similarity_matrix,
            labels=dict(x="Sentence Index", y="Sentence Index", color="Cosine Score"),
            x=[f"S{i}" for i in range(len(sentences))],
            y=[f"S{i}" for i in range(len(sentences))],
            color_continuous_scale=[[0, '#F5EFEB'], [0.5, '#7D8F74'], [1, '#C97A63']],
            text_auto=".4f"
        )
        fig_heat.update_layout(title="Pairwise Cosine Similarity Matrix", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Column break for structural separation
        col1, col2 = st.columns(2)
        
        with col1:
            # Graph 2: Bar Chart relative to Target Base Sentence
            base_idx = st.selectbox("Select Target Base Sentence for Bar Chart Comparison:", range(len(sentences)), format_func=lambda x: f"S{x}: {sentences[x][:40]}...")
            scores = similarity_matrix[base_idx]
            
            fig_bar = px.bar(
                x=[f"S{i}" for i in range(len(sentences))],
                y=scores,
                labels={'x': 'Compared Sentence', 'y': 'Similarity Vector Score'},
                color=scores,
                color_continuous_scale=[['0', '#7D8F74'], ['1', '#C97A63']]
            )
            fig_bar.update_layout(title=f"Similarity Scores Relative to Sentence S{base_idx}", yaxis_range=[0, 1.05], plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            # Graph 3: 2D Embedding Space Plot using Principal Component Analysis (PCA)
            # Adjust components dynamically based on text volume
            n_comp = min(2, len(sentences))
            pca = PCA(n_components=n_comp)
            components = pca.fit_transform(embeddings)
            
            # Fill remaining coordinates with zeros if dimensions are restricted
            if n_comp < 2:
                components = np.hstack((components, np.zeros((components.shape[0], 1))))
                
            fig_scatter = px.scatter(
                x=components[:, 0],
                y=components[:, 1],
                text=[f"S{i}" for i in range(len(sentences))],
                hover_name=sentences
            )
            fig_scatter.update_traces(marker=dict(size=14, color='#C97A63', line=dict(width=1, color='#3E3631')), textposition='top center')
            fig_scatter.update_layout(title="2D Spatial Embedding Projections (PCA Vector Space)", xaxis_title="Principal Axis 1", yaxis_title="Principal Axis 2", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Quick Reference Guide Box
        st.markdown("#### Sentence Reference Mapping")
        for i, sentence in enumerate(sentences):
            st.write(f"**S{i}**: {sentence}")

    with tab2:
        st.markdown("### Critical Thinking Assessment Frame")
        st.markdown("Evaluation formulated based on Paul's Universal Critical Thinking Standards.")
        
        # Calculate dynamic logic statistics for presentation
        flat_matrix = similarity_matrix.copy()
        np.fill_diagonal(flat_matrix, -1)
        max_idx = np.unravel_index(flat_matrix.argmax(), flat_matrix.shape)
        
        standards = {
            "Clarity": f"The system takes raw, unedited text strings through a sidebar wrapper interface and calculates numeric semantic vectors. A high score near 1.0000 establishes near-identical conceptual context, while values near 0.0000 reveal disparate textual configurations.",
            "Accuracy": "Evaluations rely strictly on the pre-trained 'all-MiniLM-L6-v2' Transformer architecture mapping 384-dimensional mathematical vector definitions. No external modifications, arbitrary configurations, or unverified semantic assumptions are applied.",
            "Precision": f"Calculations bypass relative linguistic categorizations ('high' or 'low'). The exact mathematical similarity maximum detected across non-identical pairs tracks at precisely {flat_matrix[max_idx]:.4f} linking S{max_idx[0]} and S{max_idx[1]}.",
            "Relevance": "The generated Bar Charts, Heatmaps, and 2D Spatial projections are sourced directly from the calculated dot products of the normalized embedding tensors, ensuring that all visual markers directly map back to structural models.",
            "Logic": f"The highest computed semantic alignment registers between S{max_idx[0]} and S{max_idx[1]}. This aligns perfectly with rational semantic structures, as both records share central contextual structures without needing manual preprocessing adjustments.",
            "Significance": f"The model isolates structural conceptual boundaries effectively. While sentences may look visually different or use distinct vocabulary, the vector positioning clearly flags the most significant structural relationship with a score of {flat_matrix[max_idx]:.4f}.",
            "Fairness": "Limitation Acknowledgment: This pre-trained model evaluates expressions within structural context maps. It lacks dynamic domain-specific knowledge extensions, specialized jargon capabilities, and contextual awareness for tracking subtle forms of sarcasm."
        }
        
        for key, text in standards.items():
            st.markdown(f"""
            <div class="critical-card">
                <div class="critical-title">{key}</div>
                <div>{text}</div>
            </div>
            """, unsafe_allow_html=True)
