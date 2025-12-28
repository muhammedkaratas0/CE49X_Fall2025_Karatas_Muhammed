import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path so we can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is .../src/visualization
# We want .../ (root)
# ../ is src, ../../ is root
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: Project Root added to path: {project_root}")

from src.storage.supabase_client import SupabaseManager

def create_dashboard():
    print("--- Generating Dashboard ---")
    
    # Fetch Data
    db = SupabaseManager()
    data = db.fetch_all_articles()
    
    if not data:
        print("No data found in Supabase. Checking local fallback...")
        if os.path.exists("data/processed/latest_run.csv"):
            df = pd.read_csv("data/processed/latest_run.csv")
            print("Loaded local data.")
        else:
            print("No data available to visualize.")
            return
    else:
        df = pd.DataFrame(data)

    # Data Cleaning for Visualization
    # Filter out 'None' values to focus on relevant classifications
    if 'category' in df.columns:
        df = df[df['category'] != 'None']
        df = df[df['category'] != 'Other'] # Optional: User might want to remove 'Other' too if it's vague, but 'None' was specific request. Let's keep 'Other' unless requested, but usually 'None' is the junk one. 
        # Actually user said "None kategorisini", usually implies the one named "None".
        # Let's check the code... llm_processor defaults category to "Other", ai_tech to "None".
        # So "None" likely refers to ai_tech="None" appearing in the heatmap, or maybe the LLM output "None" for category?
        # To be safe, let's filter "None" from both.
        
    if 'ai_tech' in df.columns:
        df = df[df['ai_tech'] != 'None']
        
    if df.empty:
        print("DataFrame is empty after filtering 'None'.")
        return

    # Ensure output directory
    os.makedirs("output/plots", exist_ok=True)

    # 1. Bar Chart: Articles per Category
    if 'category' in df.columns:
        plt.figure(figsize=(10, 6))
        df['category'].value_counts().plot(kind='bar', color='skyblue')
        plt.title("Articles per Civil Engineering Category")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/plots/category_distribution.png")
        print("Saved category_distribution.png")
        plt.close()

    # 2. Heatmap: Category vs AI Tech
    if 'category' in df.columns and 'ai_tech' in df.columns:
        pivot = pd.crosstab(df['category'], df['ai_tech'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt='d')
        plt.title("Category vs AI Technology Heatmap")
        plt.ylabel("Category")
        plt.xlabel("AI Technology")
        plt.tight_layout()
        plt.savefig("output/plots/tech_heatmap.png")
        print("Saved tech_heatmap.png")
        plt.close()

    # 3. Word Cloud per Category (Requirement: Word Clouds)
    # Using summaries to generate word clouds
    from wordcloud import WordCloud
    
    if 'category' in df.columns and 'summary' in df.columns:
        categories = df['category'].unique()
        for cat in categories:
            try:
                text_corpus = " ".join(df[df['category'] == cat]['summary'].fillna("").astype(str))
                if len(text_corpus) < 100: continue # Skip if not enough text
                
                wc = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                # Clean filename
                safe_name = "".join(c for c in cat if c.isalnum() or c in (' ','_')).strip().replace(' ','_')
                plt.title(f"Word Cloud: {cat}")
                plt.tight_layout()
                plt.savefig(f"output/plots/wordcloud_{safe_name}.png")
                print(f"Saved wordcloud_{safe_name}.png")
                plt.close()
            except Exception as e:
                print(f"Error generating wordcloud for {cat}: {e}")

    # 4. Network Graph (Requirement: Network Graph)
    # Linking Categories to AI Tech
    try:
        import networkx as nx
        if 'category' in df.columns and 'ai_tech' in df.columns:
            G = nx.Graph()
            
            # Add edges
            for _, row in df.iterrows():
                cat = row['category']
                tech = row['ai_tech']
                if cat and tech and cat != 'Other' and tech != 'None':
                    if G.has_edge(cat, tech):
                        G[cat][tech]['weight'] += 1
                    else:
                        G.add_edge(cat, tech, weight=1)
            
            # Prune small edges to reduce clutter
            edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2]
            G.remove_edges_from(edges_to_remove)
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.3)
            
            # Draw nodes
            # Color by type if we knew them, here we guess based on name or simply color all same
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgreen', alpha=0.7)
            
            # Draw edges with varying width
            weights = [G[u][v]['weight'] * 0.1 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')
            
            # Labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
            
            plt.title("Civil Engineering Areas <-> AI Tech Network")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("output/plots/network_graph.png")
            print("Saved network_graph.png")
            plt.close()
    except Exception as e:
        print(f"Error generating network graph: {e}")

    # 5. Top 20 Words Analysis (Requirement: NLP Stats)
    try:
        from collections import Counter
        import re
        
        all_text = " ".join(df['summary'].fillna("").astype(str)).lower()
        # Simple tokenization
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        
        # Basic stopwords list since we didn't download NLTK data to be fast/offline safe
        stopwords = set(['the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was', 'were', 'have', 'has', 'will', 'can', 'construction', 'engineering', 'civil',
                        # Turkish Stopwords
                        'bu', 'bir', 've', 'ile', 'için', 'olarak', 'daha', 'en', 'kadar', 'gibi', 'makale', 'çalışma', 'yeni', 'proje', 'olan', 'tarafından', 'var', 'yok', 'ama', 'fakat', 'ancak', 'veya', 'ise', 'çünkü']) 
        # Added 'construction', 'engineering', 'civil' as they are domain stopwords likely to dominate
        
        filtered_words = [w for w in words if w not in stopwords]
        
        common_words = Counter(filtered_words).most_common(20)
        
        # Save to text report
        with open("output/plots/top_20_words.txt", "w") as f:
            f.write("TOP 20 FREQUENT WORDS (Excluding Stopwords):\n")
            f.write("-------------------------------------------\n")
            for word, count in common_words:
                f.write(f"{word}: {count}\n")
        print("Saved top_20_words.txt")
        
    except Exception as e:
         print(f"Error analyzing top words: {e}")

    print("--- Dashboard Update Complete ---")

if __name__ == "__main__":
    create_dashboard()
