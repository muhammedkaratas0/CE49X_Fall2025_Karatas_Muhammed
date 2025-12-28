import feedparser
import urllib.parse
from datetime import datetime
import time
import random

def fetch_google_news_targeted():
    """
    Fetches articles from Google News RSS using specific queries to ensure high relevance
    to AI and Civil Engineering.
    """
    queries = [
        # --- High Precision (Tier 1) ---
        '"Artificial Intelligence" AND "Civil Engineering"',
        '"Machine Learning" AND "Structural Engineering"',
        '"Computer Vision" AND "Construction Safety"',
        '"Generative AI" AND "Architecture"',
        '"Robotics" AND "Construction Site"',
        '"AI" AND "Geotechnical Engineering"',
        '"Smart City" AND "Traffic Management AI"',
        '"Digital Twin" AND "Infrastructure"',
        '"Deep Learning" AND "Concrete crack detection"',
        '"AI" AND "Building Information Modeling"',
        
        # --- Broad Scope (Tier 2 - Gap Fillers) ---
        '"Automation" AND "Bridge Inspection"',
        '"Technology" AND "Highway Maintenance"',
        '"Smart Materials" AND "Construction"',
        '"Predictive Analytics" AND "Project Management"',
        '"Drones" AND "Surveying"',
        '"3D Printing" AND "Civil Construction"',
        # --- Niche & Emerging (Tier 3 - Final Push) ---
        '"Geographic Information System" AND "Civil Engineering"',
        '"Augmented Reality" AND "Construction Site"',
        '"BIM" AND "Facility Management"',
        '"Smart Water Management" AND "AI"',
        '"Waste Management" AND "Construction Technology"',
        '"Energy Efficiency" AND "Building Automation"',
        '"Disaster Resilience" AND "AI"',
        '"Urban Planning" AND "Machine Learning"',
        '"Self-healing Concrete" AND "Technology"',
        '"Modular Construction" AND "Robotics"',
        
        # --- Targeted Balancing (Tier 4 - Underrepresented Areas) ---
        '"Structural Health Monitoring" AND "AI"',
        '"Geotechnical Engineering" AND "Soil Analysis"',
        '"Environmental Engineering" AND "AI"',
        '"Transportation Engineering" AND "Traffic Flow"',
        '"Hydraulic Engineering" AND "Simulation"',
        '"Water Resources" AND "Machine Learning"',
        '"Earthquake Engineering" AND "AI Prediction"',
        '"Sustainable Infrastructure" AND "AI"',
        '"Pavement Engineering" AND "Smart Sensors"',
        '"Coastal Engineering" AND "Climate Model"'
    ]

    base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
    
    all_articles = []
    seen_links = set()

    print(f"--- Google News Targeted Fetch Started ({len(queries)} queries) ---")

    for query in queries:
        encoded_query = urllib.parse.quote(query)
        url = base_url.format(encoded_query)
        
        try:
            print(f"Fetching Google News: {query}...")
            feed = feedparser.parse(url)
            
            count = 0
            for entry in feed.entries:
                link = entry.get("link", "")
                if link in seen_links:
                    continue
                
                title = entry.get("title", "No Title")
                # Google News titles often have " - SourceName" at the end, clean it if needed
                # But source info is useful.
                
                pub_date = entry.get("published", datetime.now().isoformat())
                
                # Google News RSS summaries are often HTML snippets
                summary = entry.get("summary", "") or entry.get("description", "")
                
                articles_data = {
                    "title": title,
                    "link": link,
                    "published": pub_date,
                    "summary": summary, # Will be cleaned/analyzed by LLM
                    "source_name": "GoogleNews_Targeted",
                    "query_tag": query
                }
                
                all_articles.append(articles_data)
                seen_links.add(link)
                count += 1
            
            print(f"  -> Found {count} unique articles.")
            time.sleep(random.uniform(0.5, 1.0)) # Polite delay between queries

        except Exception as e:
            print(f"Error fetching query '{query}': {e}")

    print(f"Total Google News Target articles: {len(all_articles)}")
    return all_articles

if __name__ == "__main__":
    results = fetch_google_news_targeted()
    print(f"Sample: {results[0]['title'] if results else 'None'}")
