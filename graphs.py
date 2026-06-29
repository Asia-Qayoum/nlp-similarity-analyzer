import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Shared Palette & Styling
PALETTE = ["#D4A574", "#A8C686", "#7ECFC0", "#F4906A", "#E8F1EC"]
LAYOUT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FFFFFF', size=13), margin=dict(l=20, r=20, t=40, b=20)
)

def critical_thinking_bar(heuristics: dict) -> go.Figure:
    """Displays just the bars for the critical thinking index."""
    import pandas as pd
    df = pd.DataFrame(list(heuristics.items()), columns=['Dimension', 'Score'])
    fig = px.bar(df, x='Score', y='Dimension', orientation='h', 
                 color='Score', color_continuous_scale=[[0, '#7ECFC0'], [1, '#D4A574']])
    fig.update_layout(**LAYOUT_BASE, title="Critical Thinking Results", yaxis={'categoryorder':'total ascending'})
    fig.update_xaxes(range=[0, 1])
    return fig

def radar_chart(scores_dict: dict[str, float]) -> go.Figure:
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], 
                                  fill='toself', fillcolor='rgba(212, 165, 116, 0.4)', 
                                  line_color='#F4906A'))
    fig.update_layout(**LAYOUT_BASE, polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Thinking Index")
    return fig

def heatmap_similarity(matrix, labels) -> go.Figure:
    fig = px.imshow(matrix, x=labels, y=labels, color_continuous_scale=[[0, 'rgba(0,0,0,0)'], [1, '#D4A574']], text_auto=".2f")
    fig.update_layout(**LAYOUT_BASE, title="Similarity Matrix")
    return fig

def pca_scatter(embeddings, labels) -> go.Figure:
    pca = PCA(n_components=2)
    comps = pca.fit_transform(embeddings)
    fig = px.scatter(x=comps[:, 0], y=comps[:, 1], text=labels)
    fig.update_traces(marker=dict(color='#D4A574'), textposition='top center')
    fig.update_layout(**LAYOUT_BASE, title="2D Projections")
    return fig
