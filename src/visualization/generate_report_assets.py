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

OUTPUT_DIR = os.path.join(project_root, "output")
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

def generate_bar_chart(df):
    print("Generating Bar Chart...")
    plt.figure(figsize=(10, 6))
    counts = df['primary_ce_area'].value_counts()
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title("Number of Articles per Civil Engineering Area")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_bar_chart_ce_areas.png"), dpi=300)
    plt.close()

def generate_heatmap(df):
    print("Generating Heatmap...")
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
        
        # Filter top 10x10 for readability
        top_ce = pivot.sum(axis=1).nlargest(10).index
        top_ai = pivot.sum(axis=0).nlargest(10).index
        pivot = pivot.loc[top_ce, top_ai]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', linewidths=.5)
        plt.title("Co-occurrence: CE Areas vs AI Technologies")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_heatmap_ce_ai.png"), dpi=300)
        plt.close()

def generate_network_graph(df):
    print("Generating Network Graph...")
    G = nx.Graph()
    
    edge_counts = {}
    
    for _, row in df.iterrows():
        ce_items = []
        ai_items = []
        
        # Extraction logic matching app.py
        if isinstance(row.get('ce_areas'), list):
            ce_items = [i.get('area') for i in row['ce_areas'] if i.get('area')]
        if isinstance(row.get('ai_technologies'), list):
            ai_items = [i.get('technology') for i in row['ai_technologies'] if i.get('technology')]
            
        for ce in ce_items:
            for ai in ai_items:
                pair = tuple(sorted((ce, ai)))
                edge_counts[pair] = edge_counts.get(pair, 0) + 1
                
    # Filter edges
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    top_edges = sorted_edges[:50] # Top 50 connections
    
    for (u, v), w in top_edges:
        G.add_edge(u, v, weight=w)
        
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=1.0, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight']*0.2 for u,v in G.edges()], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    plt.title("Network Graph: CE & AI Co-occurrences")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_network_graph.png"), dpi=300)
    plt.close()

def generate_word_clouds(df):
    print("Generating Word Clouds...")
    top_areas = df['primary_ce_area'].value_counts().head(4).index.tolist()
    
    for area in top_areas:
        subset = df[df['primary_ce_area'] == area]
        
        # Collect keywords (using AI tech and title words as proxy)
        text = " ".join(subset['title'].dropna().tolist())
        # Add AI techs repeatedly to weight them
        techs = " ".join(subset['primary_ai_tech'].dropna().tolist() * 3)
        full_text = text + " " + techs
        
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(full_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud: {area}")
        plt.tight_layout()
        safe_name = area.replace(" ", "_").lower()
        plt.savefig(os.path.join(OUTPUT_DIR, f"4_wordcloud_{safe_name}.png"), dpi=300)
        plt.close()

def calculate_conclusions(df):
    print("Calculating Conclusions...")
    # Ranking by Volume
    ranking = df['primary_ce_area'].value_counts().to_string()
    
    with open(os.path.join(OUTPUT_DIR, "5_conclusions_ranking.txt"), "w") as f:
        f.write("=== CE Area Ranking by Article Volume ===\n")
        f.write(ranking)
        f.write("\n\n=== Top AI Technologies ===\n")
        f.write(df['primary_ai_tech'].value_counts().head(10).to_string())

if __name__ == "__main__":
    try:
        df = load_data()
        if not df.empty:
            generate_bar_chart(df)
            generate_heatmap(df)
            generate_network_graph(df)
            generate_word_clouds(df)
            calculate_conclusions(df)
            print("All assets generated in 'output/' directory.")
        else:
            print("No data loaded.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
