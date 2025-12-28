import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.supabase_client import SupabaseManager

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Civil Engineering & AI Strategic Dashboard",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CONSTANTS & STYLING ---
# Color Palette (Boğaziçi Blue Inspired + User Request)
COLORS = {
    'primary': '#1E3A8A',          # Dark Blue
    'secondary': '#2563EB',        # Bright Blue
    'construction_mgmt': '#FF6B6B',# Red/Coral
    'transportation': '#4ECDC4',   # Turquoise
    'structural': '#45B7D1',       # Blue
    'materials': '#FFA07A',        # Orange
    'environmental': '#98D8C8',    # Green
    'geotechnical': '#D4A5A5',     # Pale Red
    'default': '#95A5A6',          # Grey
    'academic': '#6C5CE7',
    'industry': '#00B894',
    'tech_news': '#FDCB6E'
}

AI_COLORS = {
    'Machine Learning': '#6C5CE7',
    'Computer Vision': '#00B894',
    'NLP': '#FDCB6E',
    'Robotics': '#E17055',
    'Deep Learning': '#74B9FF',
    'Generative AI': '#A29BFE'
}

# Custom CSS
st.markdown("""
<style>
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #1E3A8A;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px;
        color: #4B5563;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #EFF6FF;
        color: #1E3A8A;
        border: 2px solid #1E3A8A;
    }
    h1, h2, h3 {
        color: #111827;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: bold;
    }
    .delta-pos { color: #10B981; }
    .delta-neg { color: #EF4444; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=600)
def load_data():
    db = SupabaseManager()
    data = db.fetch_dataset_articles()
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Preprocessing
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        df['year'] = df['publication_date'].dt.year
        df['month_year'] = df['publication_date'].dt.to_period('M').astype(str)
        
    # Unpack JSON lists if string
    for col in ['ce_areas', 'ai_technologies']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return df

# --- HELPER FUNCTIONS ---
def get_color(key, palette=COLORS):
    return palette.get(key, palette['default'])

def calculate_maturity_index(df_grouped):
    """
    Calculates AI Maturity Index for grouped data.
    Row count (30%) + Diversity (30%) + Avg Relevance (20%) + Academic Ratio (20%)
    """
    # Normalize each component 0-10
    
    # 1. Volume (Log scale to not punish small areas too much, or linear?) 
    # Let's use linear cap at max.
    max_count = df_grouped['count'].max()
    score_vol = (df_grouped['count'] / max_count) * 10
    
    # 2. Diversity (Unique AI techs count / Total possible AI techs)
    # We need to pre-calc diversity outside this usually, but let's assume passed in
    if 'diversity' in df_grouped.columns:
         max_div = df_grouped['diversity'].max()
         score_div = (df_grouped['diversity'] / max_div) * 10
    else:
        score_div = 0
        
    # 3. Quality (Avg Relevancy 0-100 -> 0-10)
    score_qual = (df_grouped['relevancy_score'] / 100) * 10
    
    # 4. Academic Ratio (0-1 -> 0-10)
    if 'academic_ratio' in df_grouped.columns:
        score_acad = df_grouped['academic_ratio'] * 10
    else:
        score_acad = 0
        
    maturity = (score_vol * 0.3) + (score_div * 0.3) + (score_qual * 0.2) + (score_acad * 0.2)
    return maturity

# --- MAIN APP ---
def main():
    st.markdown("<h1>🏗️ Strategic Dashboard: Civil Engineering & AI</h1>", unsafe_allow_html=True)
    
    with st.spinner("Fetching dataset from Supabase..."):
        df = load_data()
        
    if df.empty:
        st.error("No data found! Please check backend connection.")
        return

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("🎯 Filters")
        
        # Year Range
        min_year = int(df['year'].min()) if pd.notnull(df['year'].min()) else 2010
        max_year = int(df['year'].max()) if pd.notnull(df['year'].max()) else datetime.now().year
        
        year_range = st.slider("Publication Year", min_year, max_year, (min_year, max_year))
        
        # CE Areas
        all_areas = sorted(list(df['primary_ce_area'].dropna().unique()))
        selected_areas = st.multiselect("CE Areas", all_areas, default=all_areas)
        
        # AI Tech
        all_techs = sorted(list(df['primary_ai_tech'].dropna().unique()))
        selected_techs = st.multiselect("AI Technologies", all_techs, default=[]) # Default all if empty
        
        # Source Category
        all_sources = sorted(list(df['source_category'].dropna().unique()))
        selected_sources = st.multiselect("Source Category", all_sources, default=all_sources)
        
        # Apply Filters
        mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
        if selected_areas: mask &= df['primary_ce_area'].isin(selected_areas)
        if selected_techs: mask &= df['primary_ai_tech'].isin(selected_techs)
        if selected_sources: mask &= df['source_category'].isin(selected_sources)
        
        df_filtered = df[mask]
        
        st.divider()
        st.write(f"**Selected Articles:** {len(df_filtered)}")
        st.write(f"**Total Database:** {len(df)}")
        
        if st.button("Reset Filters"):
            st.rerun()

        st.markdown("---")
        if st.button("🔄 Reload Data (Clear Cache)"):
            st.cache_data.clear()
            st.rerun()

    # --- TABS ---
    tab_overview, tab_ce, tab_ai, tab_trends, tab_quality = st.tabs([
        "📊 1. Overview", 
        "🏗️ 2. CE Analysis", 
        "🤖 3. AI Analysis", 
        "📈 4. Trends", 
        "💎 5. Quality"
    ])

    # === TAB 1: EXECUTIVE SUMMARY ===
    with tab_overview:
        # 1. KPIs
        top_ce = df_filtered['primary_ce_area'].mode()[0] if not df_filtered.empty else "-"
        top_ai = df_filtered['primary_ai_tech'].mode()[0] if not df_filtered.empty else "-"
        # Calculate Academic %
        source_counts = df_filtered['source_category'].value_counts()
        total_count = len(df_filtered)
        acad_count = source_counts.get('academic', 0)
        acad_pct = (acad_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_count}</div><div class="kpi-label">Total Articles</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.4rem; padding-top:10px">{top_ce}</div><div class="kpi-label">#1 CE Area</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.4rem; padding-top:10px">{top_ai}</div><div class="kpi-label">#1 AI Tech</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="kpi-card"><div class="kpi-value">{acad_pct:.0f}%</div><div class="kpi-label">Academic Sources</div></div>', unsafe_allow_html=True)

        # 2. Charts (Row 1)
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Distribution by CE Area")
            ce_dist = df_filtered['primary_ce_area'].value_counts().reset_index()
            ce_dist.columns = ['Area', 'Count']
            # Limit to top 5 + Other for donut
            if len(ce_dist) > 6:
                top5 = ce_dist.head(5)
                other = pd.DataFrame([{'Area': 'Others', 'Count': ce_dist.iloc[5:]['Count'].sum()}])
                ce_dist_chart = pd.concat([top5, other])
            else:
                ce_dist_chart = ce_dist
                
            fig_donut = px.pie(ce_dist_chart, values='Count', names='Area', hole=0.5, 
                               color_discrete_map=COLORS)
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c2:
            st.subheader("Top CE Areas by Volume")
            # Bar chart with custom colors
            fig_bar = px.bar(ce_dist.head(8), x='Area', y='Count', 
                             color='Area', color_discrete_map=COLORS,
                             text='Count')
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of Articles", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. Trend Line
        st.subheader("Global Publication Trend")
        trend_data = df_filtered.groupby('year').size().reset_index(name='Count')
        fig_trend = px.line(trend_data, x='year', y='Count', markers=True)
        fig_trend.update_traces(line_color=COLORS['primary'], line_width=4)
        fig_trend.update_layout(xaxis=dict(tickmode='linear'), plot_bgcolor='white')
        st.plotly_chart(fig_trend, use_container_width=True)

    # === TAB 2: CE AREA ANALYSIS ===
    with tab_ce:
        # Pre-calculation for Advanced Metrics
        ce_groups = df_filtered.groupby('primary_ce_area')
        
        # Helper to get unique AI techs per group
        def count_unique_techs(x):
            techs = set()
            for tech_list in x:
                if isinstance(tech_list, list):
                    for t in tech_list: techs.add(t.get('technology'))
            return len(techs)

        ce_summary = ce_groups.agg({
            'id': 'count',
            'relevancy_score': 'mean',
            'source_category': lambda x: (x == 'academic').sum() / len(x) if len(x) > 0 else 0,
            'ai_technologies': count_unique_techs
        }).rename(columns={'id': 'count', 'source_category': 'academic_ratio', 'ai_technologies': 'diversity'})
        
        ce_summary['maturity_index'] = calculate_maturity_index(ce_summary)
        ce_summary = ce_summary.sort_values('maturity_index', ascending=False)
        
        col_ce1, col_ce2 = st.columns([2, 1])
        
        with col_ce1:
            st.subheader("🔥 CE Area x AI Technology Heatmap")
            st.caption("Intensity shows number of articles applying specific AI Tech in a CE Area.")
            
            # Prepare matrix
            # Explode AI technologies? Or just use primary.
            # Strategy says "Interactive Heatmap". Using primary is cleaner for high level, 
            # but user might want full data. Let's use primary for clarity as per "Overview" logic, 
            # or unpack if we want detailed "usage". Let's unpack for accuracy.
            
            heatmap_data = []
            for _, row in df_filtered.iterrows():
                ce = row['primary_ce_area']
                if not ce: continue
                # Use primary tech or all? Let's use primary to avoid overcounting in this simple view, 
                # or all for "usage". Let's use primary for now to match the "Primary" fields available nicely.
                # Actually, user prompt mentioned "Construction Management + Computer Vision".
                # Let's stick to Primary vs Primary for the main heatmap to keep totals matching article counts roughly.
                tech = row['primary_ai_tech']
                if tech:
                    heatmap_data.append({'CE Area': ce, 'AI Tech': tech})
            
            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data)
                # Count
                hm_pivot = pd.crosstab(hm_df['CE Area'], hm_df['AI Tech'])
                # Filter Top X to specific requirements (Top 8 CE, Top 12 AI)
                top_ce_idx = hm_pivot.sum(axis=1).nlargest(8).index
                top_ai_idx = hm_pivot.sum(axis=0).nlargest(12).index
                
                hm_pivot = hm_pivot.loc[top_ce_idx, top_ai_idx]
                
                fig_hm = px.imshow(hm_pivot, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
                fig_hm.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Insufficient data for heatmap.")

        with col_ce2:
            st.subheader("🏆 AI Maturity Index")
            st.caption("Composite score: Volume, Diversity, Quality, Academic%")
            
            fig_mat = px.bar(ce_summary.head(10).sort_values('maturity_index', ascending=True), 
                             x='maturity_index', y=ce_summary.head(10).index, 
                             orientation='h', 
                             color='maturity_index', color_continuous_scale='Viridis',
                             text_auto='.1f')
            fig_mat.update_layout(showlegend=False, xaxis_title="Score (0-10)", yaxis_title="")
            st.plotly_chart(fig_mat, use_container_width=True)

        st.subheader("Radar Chart: Area Performance Comparison")
        # Comparing Top 5 Areas
        top_5_radar = ce_summary.head(5).reset_index()
        # Normalize columns for radar
        categories = ['Volume', 'Quality', 'Diversity', 'Academic Focus']
        # We need to melt this
        # Scale 0-1
        scaler = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        top_5_radar['nr_count'] = scaler(top_5_radar['count'])
        top_5_radar['nr_qual'] = scaler(top_5_radar['relevancy_score'])
        top_5_radar['nr_div'] = scaler(top_5_radar['diversity'])
        top_5_radar['nr_acad'] = scaler(top_5_radar['academic_ratio'])
        
        fig_radar = go.Figure()
        
        for i, row in top_5_radar.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['nr_count'], row['nr_qual'], row['nr_div'], row['nr_acad']],
                theta=categories,
                fill='toself',
                name=row['primary_ce_area']
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # === TAB 3: AI TECH ANALYSIS ===
    with tab_ai:
        col_ai1, col_ai2 = st.columns([1, 1])
        
        with col_ai1:
            st.subheader("Top AI Technologies")
            ai_counts = df_filtered['primary_ai_tech'].value_counts().head(15).reset_index()
            ai_counts.columns = ['Tech', 'Count']
            
            fig_ai_bar = px.bar(ai_counts, x='Count', y='Tech', orientation='h',
                                color='Count', color_continuous_scale='Purples')
            fig_ai_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ai_bar, use_container_width=True)
            
        with col_ai2:
            st.subheader("AI Tech Word Cloud")
            # Fake logic for wordcloud display since we can't install wordcloud lib easily if not present
            # We will use a Bubble Chart approach as a "Cloud" alternative or try importing
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                text = " ".join(ai_counts['Tech'].tolist() * 5) # Fake frequency by repeating names? No, use full text if possible.
                # Provide keywords from summary?
                text = ""
                for t, c in zip(ai_counts['Tech'], ai_counts['Count']):
                     text += (t.replace(" ", "_") + " ") * int(c/2) # Simple weighting
                
                wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            except ImportError:
                st.warning("WordCloud library not found. Showing Treemap instead.")
                fig_tree = px.treemap(ai_counts, path=['Tech'], values='Count')
                st.plotly_chart(fig_tree)

        st.subheader("Data Flow: CE Areas → AI Technologies (Sankey)")
        # Sankey Logic
        # Source: CE Area, Target: AI Tech
        # We need to map strings to indices
        
        sankey_data = df_filtered.groupby(['primary_ce_area', 'primary_ai_tech']).size().reset_index(name='value')
        # Filter small flows
        sankey_data = sankey_data[sankey_data['value'] > 2] # Min threshold
        
        if not sankey_data.empty:
            all_nodes = list(pd.concat([sankey_data['primary_ce_area'], sankey_data['primary_ai_tech']]).unique())
            node_map = {name: i for i, name in enumerate(all_nodes)}
            
            source_indices = sankey_data['primary_ce_area'].map(node_map).tolist()
            target_indices = sankey_data['primary_ai_tech'].map(node_map).tolist()
            values = sankey_data['value'].tolist()
            
            # Colors
            node_colors = [get_color(n) if n in COLORS else '#888' for n in all_nodes] # Simplified coloring
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="blue" # Default, or complex mapping
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values
                )
            )])
            fig_sankey.update_layout(title_text="Flow from Areas to Technologies", font_size=12)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # --- INTERACTIVE NETWORK GRAPH ---
            st.divider()
            st.subheader("🕸️ Interactive Network Graph")
            st.caption("Visualizing strong connections between CE Areas (Blue) and AI Technologies (Orange).")
            
            # 1. Build Graph
            G = nx.Graph()
            edge_list = []
            node_type_map = {}
            
            # Iterate filtered DF to build edges
            for _, row in df_filtered.iterrows():
                ce_list = []
                ai_list = []
                # Extract
                if isinstance(row.get('ce_areas'), list):
                     ce_list = [x.get('area') for x in row['ce_areas'] if x.get('area')]
                if isinstance(row.get('ai_technologies'), list):
                     ai_list = [x.get('technology') for x in row['ai_technologies'] if x.get('technology')]
                
                # Assign types
                for c in ce_list: node_type_map[c] = 'CE'
                for a in ai_list: node_type_map[a] = 'AI'
                
                # Create Edges
                for c in ce_list:
                    for a in ai_list:
                        edge_list.append(tuple(sorted((c, a))))
                        
            # Count edges
            from collections import Counter
            edge_counts = Counter(edge_list)
            
            # Filter Edges (Min weight 2 for dashboard)
            for (u, v), w in edge_counts.items():
                if w >= 2:
                    G.add_edge(u, v, weight=w)
            
            if G.number_of_nodes() > 0:
                # Layout
                # Use spring layout with k parameter to spread nodes
                pos = nx.spring_layout(G, k=0.5, seed=42)
                
                # Create Plotly Traces
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')

                node_x = []
                node_y = []
                node_text = []
                node_marker_colors = []
                node_sizes = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Size by degree or frequency
                    deg = G.degree[node]
                    node_sizes.append(10 + (deg * 1.5))
                    
                    # Color by type
                    ntype = node_type_map.get(node, 'Unknown')
                    color = '#1E3A8A' if ntype == 'CE' else '#E17055' # Blue vs Orange
                    node_marker_colors.append(color)
                    
                    node_text.append(f"{node} ({ntype}) - Connections: {deg}")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[node if G.degree[node] > 5 else "" for node in G.nodes()], # Label major nodes
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=node_text,
                    marker=dict(
                        showscale=False,
                        color=node_marker_colors,
                        size=node_sizes,
                        line_width=1,
                        line_color='white'))

                fig_net = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                             )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("Not enough connections for Network Graph with current filters.")

        else:
            st.warning("Not enough data for Sankey Diagram (Try adjusting filters).")

    # === TAB 4: TRENDS ===
    with tab_trends:
        st.subheader("Evolution of Top 5 CE Areas (2015-2025)")
        
        # Filter data for trends
        trend_df = df_filtered[df_filtered['year'] >= 2015].groupby(['year', 'primary_ce_area']).size().reset_index(name='Count')
        
        # Keep only top 5 overall areas
        top_5_areas = df_filtered['primary_ce_area'].value_counts().head(5).index.tolist()
        trend_df_top5 = trend_df[trend_df['primary_ce_area'].isin(top_5_areas)]
        
        fig_multi = px.line(trend_df_top5, x='year', y='Count', color='primary_ce_area', 
                            markers=True, color_discrete_map=COLORS)
        st.plotly_chart(fig_multi, use_container_width=True)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Growth Rate (2020 vs 2024)")
            # Compare 2020 count vs 2024 count per area
            y20 = df_filtered[df_filtered['year']==2020].groupby('primary_ce_area').size()
            y24 = df_filtered[df_filtered['year']==2024].groupby('primary_ce_area').size()
            
            growth_df = pd.DataFrame({'2020': y20, '2024': y24}).fillna(0)
            growth_df['growth_pct'] = ((growth_df['2024'] - growth_df['2020']) / (growth_df['2020'] + 1)) * 100
            
            # Top growth
            growth_df = growth_df.sort_values('growth_pct', ascending=False).head(8)
            
            fig_growth = px.bar(growth_df, x=growth_df.index, y='growth_pct',
                                color='growth_pct', color_continuous_scale='RdYlGn',
                                title="Percentage Growth (2020 to 2024)")
            st.plotly_chart(fig_growth, use_container_width=True)

    # === TAB 5: QUALITY & SOURCES ===
    with tab_quality:
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.subheader("Source Category Distribution")
            fig_pie_src = px.pie(df_filtered, names='source_category', color='source_category', 
                                 hole=0.4, color_discrete_map=COLORS)
            st.plotly_chart(fig_pie_src, use_container_width=True)
            
        with col_q2:
            st.subheader("Relevance Score Distribution")
            fig_box = px.box(df_filtered, x='source_category', y='relevancy_score', color='source_category',
                             color_discrete_map=COLORS)
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.subheader("Quality Validation: Relevance vs Confidence")
        # Scatter with trendline?
        if 'confidence_score' in df_filtered.columns:
            # Need to ensure numeric
            # Ensure float
            pass
        else:
            # Fake confidence if missing for demo? Or skip.
            # Strategy says "Scatter Plot". Let's assume it exists or use dummy/proxy (e.g. length of summary?)
            pass

        # Table of Top Sources
        st.subheader("Top Publishing Sources")
        if 'source' in df_filtered.columns:
            top_sources = df_filtered['source'].value_counts().head(10).reset_index()
            top_sources.columns = ['Source Name', 'Article Count']
            st.dataframe(top_sources, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("CE49X Final Project Dashboard | Developed by Antigravity")

if __name__ == "__main__":
    main()
