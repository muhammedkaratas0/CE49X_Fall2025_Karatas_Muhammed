import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import json
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.supabase_client import SupabaseManager

OUTPUT_DIR = os.path.join(project_root, "output", "v2")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def load_data():
    print("Fetching data...")
    db = SupabaseManager()
    data = db.fetch_dataset_articles()
    df = pd.DataFrame(data)
    
    # Unpack JSON
    for col in ['ce_areas', 'ai_technologies']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            
    print(f"Loaded {len(df)} articles.")
    return df

def generate_bar_chart_v2(df):
    print("Generating Bar Chart (V2)...")
    plt.figure(figsize=(10, 6))
    counts = df['primary_ce_area'].value_counts()
    
    # Filter specific small outliers if count < 5 OR exclude 'Other' if needed
    # Let's keep top 15 essentially or filter min count
    counts = counts[counts >= 5]
    
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title("Number of Articles per Civil Engineering Area (Min 5 Articles)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_bar_chart_ce_areas_v2.png"), dpi=300)
    plt.close()

def generate_heatmap_v2(df):
    print("Generating Heatmap (V2)...")
    # CE Area vs AI Tech
    heatmap_data = []
    for _, row in df.iterrows():
        ce = row['primary_ce_area']
        tech = row['primary_ai_tech']
        if ce and tech:
            heatmap_data.append({'CE Area': ce, 'AI Tech': tech})
            
    if heatmap_data:
        hm_df = pd.DataFrame(heatmap_data)
        pivot = pd.crosstab(hm_df['CE Area'], hm_df['AI Tech'])
        
        # FILTERING:
        # 1. Remove rows/cols with very low totals
        min_ce_count = 10
        min_ai_count = 10
        
        pivot = pivot[pivot.sum(axis=1) >= min_ce_count]
        pivot = pivot.loc[:, pivot.sum(axis=0) >= min_ai_count]
        
        plt.figure(figsize=(14, 10)) # Larger
        sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', linewidths=.5)
        plt.title("Co-occurrence: CE Areas vs AI Technologies (Filtered)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_heatmap_ce_ai_v2.png"), dpi=300)
        plt.close()

def generate_network_graph_v2(df):
    print("Generating Network Graph (V2)...")
    G = nx.Graph()
    
    edge_counts = {}
    node_freq = {}
    node_type = {} # Track type: 'CE' or 'AI'
    
    for _, row in df.iterrows():
        ce_items = []
        ai_items = []
        
        if isinstance(row.get('ce_areas'), list):
            ce_items = [i.get('area') for i in row['ce_areas'] if i.get('area')]
        if isinstance(row.get('ai_technologies'), list):
            ai_items = [i.get('technology') for i in row['ai_technologies'] if i.get('technology')]
            
        # Update node freq and type
        for x in ce_items:
            node_freq[x] = node_freq.get(x, 0) + 1
            node_type[x] = 'CE'
        for x in ai_items:
            node_freq[x] = node_freq.get(x, 0) + 1
            node_type[x] = 'AI'
            
        for ce in ce_items:
            for ai in ai_items:
                pair = tuple(sorted((ce, ai)))
                edge_counts[pair] = edge_counts.get(pair, 0) + 1
                
    # Filter edges - Keep only strong connections
    # Increase threshold for readability
    filtered_edges = {k: v for k,v in edge_counts.items() if v >= 5}
    
    # Sort and take top 50
    sorted_edges = sorted(filtered_edges.items(), key=lambda x: x[1], reverse=True)[:50]
    
    for (u, v), w in sorted_edges:
        G.add_edge(u, v, weight=w)
    
    # Also ensure nodes are added with attributes if they are part of edges
    for n in G.nodes():
        G.nodes[n]['type'] = node_type.get(n, 'Unknown')
        
    plt.figure(figsize=(16, 16))
    # 'k' controls spacing
    pos = nx.spring_layout(G, k=0.8, seed=101, iterations=50) 
    
    # Separate nodes by type for coloring
    ce_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'CE']
    ai_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'AI']
    
    # Draw Edges
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [(w / max_w) * 3 for w in weights] # Scale width
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color="grey")
    
    # Draw Nodes
    # Size based on frequency
    ce_sizes = [node_freq.get(n, 10) * 15 for n in ce_nodes]
    ai_sizes = [node_freq.get(n, 10) * 15 for n in ai_nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=ce_nodes, node_color='#1E3A8A', node_size=ce_sizes, label='CE Area', alpha=0.9) # Dark Blue
    nx.draw_networkx_nodes(G, pos, nodelist=ai_nodes, node_color='#E17055', node_size=ai_sizes, label='AI Tech', alpha=0.9) # Orange
    
    # Labels with background box for readability
    labels = {n: n for n in G.nodes()}
    # Filter labels? only large ones? No, user wants understandable, let's keep all but small font
    
    text_items = nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family="sans-serif", font_weight="bold")
    # Add outline to text
    import matplotlib.patheffects as path_effects
    for t in text_items.values():
        t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
        
    plt.title("Key Connections: Civil Engineering (Blue) & AI (Orange)", fontsize=16)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_network_graph_v2.png"), dpi=300)
    plt.close()

def generate_word_clouds_v2(df):
    print("Generating Word Clouds (V2)...")
    # Only Top 4 Areas
    top_areas = df['primary_ce_area'].value_counts().head(4).index.tolist()
    
    for area in top_areas:
        subset = df[df['primary_ce_area'] == area]
        
        # Collect keywords
        text = " ".join(subset['title'].dropna().tolist())
        techs = " ".join(subset['primary_ai_tech'].dropna().tolist() * 3)
        full_text = text + " " + techs
        
        # Remove common "noise" words specifically for this project if noticed
        stopwords = set(['Civil', 'Engineering', 'Using', 'Based', 'System', 'Analysis'])
        
        wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma', stopwords=stopwords).generate(full_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud: {area}")
        plt.tight_layout()
        safe_name = area.replace(" ", "_").lower()
        plt.savefig(os.path.join(OUTPUT_DIR, f"4_wordcloud_{safe_name}_v2.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    try:
        df = load_data()
        if not df.empty:
            generate_bar_chart_v2(df)
            generate_heatmap_v2(df)
            generate_network_graph_v2(df)
            generate_word_clouds_v2(df)
            print("V2 assets generated in 'output/v2/' directory.")
        else:
            print("No data loaded.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
