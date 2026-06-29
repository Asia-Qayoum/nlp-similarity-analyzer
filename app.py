import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import time

# ==========================================
# SECTION 1: CONFIGURATION & GLASSMORPHISM CSS
# ==========================================
st.set_page_config(page_title="NLP-Similarity-Analyzer", layout="wide", page_icon="🌿")

def inject_glass_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f1c16 0%, #1c3628 50%, #15291e 100%);
            background-attachment: fixed;
            color: #21593c;
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(10, 15, 12, 0.6) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1, h2, h3, h4, p, span, label { color: #F8FAF9 !important; font-weight: 500; }
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        }
        .stTextArea textarea, [data-baseweb="select"] > div {
            background: rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            color: #FFF !important;
            border-radius: 12px !important;
        }
        [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, rgba(212, 165, 116, 0.7), rgba(244, 144, 106, 0.7)) !important;
            color: #FFF !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
        }
        .nav-link {
            display: block; padding: 15px 20px; margin-bottom: 12px;
            color: #E8F1EC !important; text-decoration: none;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
        }
        .nav-link:hover { background: rgba(255, 255, 255, 0.15); color: #D4A574 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# SECTION 2: NLP CORE & SESSION STATE
# ==========================================
@st.cache_resource(show_spinner="Waking up the neural network...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def init_session_state():
    if "history" not in st.session_state: st.session_state.history = []
    if "current_results" not in st.session_state: st.session_state.current_results = None

def compute_similarity(texts):
    model = load_model()
    start_time = time.perf_counter()
    embeddings = model.encode(texts)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    np.fill_diagonal(sim_matrix, -1) 
    elapsed = (time.perf_counter() - start_time) * 1000
    return embeddings, sim_matrix, elapsed

# ==========================================
# SECTION 3: DARK GLASS VISUALIZATIONS
# ==========================================
CHART_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#FFFFFF', size=13), 'margin': dict(t=40, b=0, l=0, r=0)
}

def create_heatmap(matrix, labels):
    fig = px.imshow(
        matrix, x=labels, y=labels, 
        color_continuous_scale=[[0.0, 'rgba(255,255,255,0.05)'], [1.0, '#D4A574']],
        text_auto=".3f", aspect="auto"
    )
    fig.update_layout(**CHART_THEME, title="Pairwise Semantic Heatmap")
    return fig

def create_bar_chart(matrix, labels, target_idx=0):
    scores = matrix[target_idx].copy()
    scores[target_idx] = 1.0 
    fig = px.bar(
        x=labels, y=scores, color=scores, 
        color_continuous_scale=[[0.0, '#7ECFC0'], [1.0, '#F4906A']]
    )
    fig.update_layout(**CHART_THEME, title=f"Similarity relative to Text {target_idx+1}", yaxis_title="Cosine Score")
    return fig

def create_pca_scatter(embeddings, labels):
    n_comp = min(2, len(embeddings))
    pca = PCA(n_components=n_comp)
    comps = pca.fit_transform(embeddings)
    if n_comp == 1: comps = np.hstack((comps, np.zeros((comps.shape[0], 1))))
    fig = px.scatter(x=comps[:, 0], y=comps[:, 1], text=labels, size=[15]*len(labels))
    fig.update_traces(marker=dict(color='#D4A574', line=dict(width=2, color='white')), textposition='top center')
    fig.update_layout(**CHART_THEME, title="2D Semantic Projections (PCA)")
    return fig

def create_radar(scores_dict):
    df = pd.DataFrame(dict(r=list(scores_dict.values()), theta=list(scores_dict.keys())))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself', fillcolor='rgba(212, 165, 116, 0.4)', line_color='#F4906A', line_width=3)
    fig.update_layout(**CHART_THEME, polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)'), angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')), title="Critical Thinking Radar")
    return fig

# --- NEW: Horizontal Bar Chart for Heuristics ---
def create_heuristic_bar(heuristics):
    df = pd.DataFrame(list(heuristics.items()), columns=['Metric', 'Score'])
    fig = px.bar(
        df, x='Score', y='Metric', orientation='h',
        color='Score', color_continuous_scale=[[0, '#7ECFC0'], [1, '#D4A574']]
    )
    fig.update_layout(**CHART_THEME, title="Critical Thinking Index", yaxis={'categoryorder':'total ascending'})
    fig.update_xaxes(range=[0, 1])
    return fig

# ==========================================
# SECTION 4: SINGLE PAGE LAYOUT
# ==========================================
def main():
    inject_glass_css()
    init_session_state()
    
    # --- SIDEBAR ---
    st.sidebar.markdown("<h2 style='color:#D4A574; text-align:center;'>🌿 Quick Navigate</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <a href="#analysis" class="nav-link">🔍 Analysis Engine</a>
        <a href="#visualisations" class="nav-link">📊 Visualisations</a>
        <a href="#critical-thinking" class="nav-link">🧠 Critical Thinking</a>
    """, unsafe_allow_html=True)
    
    # --- HEADER & INPUT ---
    # Put everything inside one glass card to fix the gap
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>Organic Semantic AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A8C686 !important;'>Discover the hidden mathematical connections between your words.</p>", unsafe_allow_html=True)
    
    n_inputs = st.selectbox("Inputs to compare:", [2, 3, 4, 5, 6], index=1)
    with st.form("semantic_form"):
        cols = st.columns(2)
        text_inputs = []
        for i in range(n_inputs):
            with cols[i % 2]:
                text_inputs.append(st.text_area(f"Text Block {i+1}", key=f"t_{i}", height=120))
        submitted = st.form_submit_button("Synthesize & Analyze 🚀", use_container_width=True)
        if submitted:
            valid_texts = [t.strip() for t in text_inputs if t.strip()]
            if len(valid_texts) < 2: st.error("Please enter at least two text blocks.")
            else:
                emb, matrix, ms = compute_similarity(valid_texts)
                st.session_state.current_results = {"texts": valid_texts, "matrix": matrix, "embeddings": emb, "max_score": np.max(matrix)}
    st.markdown("</div>", unsafe_allow_html=True)

    # --- VISUALISATIONS ---
    if st.session_state.current_results:
        res = st.session_state.current_results
        labels = [f"T{i+1}" for i in range(len(res['texts']))]
        
        st.markdown("<div id='visualisations' class='glass-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_heatmap(res['matrix'], labels), use_container_width=True)
        with c2: st.plotly_chart(create_bar_chart(res['matrix'], labels), use_container_width=True)
        st.plotly_chart(create_pca_scatter(res['embeddings'], labels), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- CRITICAL THINKING ---
        st.markdown("<div id='critical-thinking' class='glass-card'>", unsafe_allow_html=True)
        score = float(res['max_score'])
        heuristics = {
            "Clarity": 0.9, "Accuracy": 0.95, "Precision": score, 
            "Relevance": 0.85, "Logic": min(score + 0.1, 1.0), 
            "Significance": 0.8, "Fairness": 0.7
        }
        
        # Displaying Radar and New Bar Chart side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_radar(heuristics), use_container_width=True)
        with col2:
            st.plotly_chart(create_heuristic_bar(heuristics), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
