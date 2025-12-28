import time
from src.ingestion.rss_fetcher import fetch_rss_feeds
from src.intelligence.llm_processor import analyze_article
from src.storage.supabase_client import SupabaseManager

def run_pipeline():
    print("--- Starting Pipeline ---")
    
    # 1. Fetch Data
    articles = fetch_rss_feeds()
    if not articles:
        print("No articles found.")
        return

    # 2. Process with LLM
    print("Analyzing articles with LLM...")
    enriched_articles = []
    
    # Initialize DB (if available)
    db = SupabaseManager()
    
    for article in articles:
        print(f"Processing: {article['title'][:50]}...")
        
        # Analyze
        analysis = analyze_article(article['title'], article['summary'])
        
        # Skip if not relevant
        if not analysis.get("is_relevant", True):
            continue

        # Merge results
        enriched_data = {
            "title": article['title'],
            "url": article['link'],
            "published_at": article['published'],
            "source_name": article['source_name'],
            "summary": analysis.get("summary", ""),
            "category": analysis.get("category", "Other"),
            "ai_tech": analysis.get("ai_tech", "None"),
            "sentiment": analysis.get("sentiment", "Neutral")
        }
        
        # Save one by one immediately
        if db.client:
            db.insert_articles([enriched_data])
        
        # Paid Tier: Fast processing
        time.sleep(0.2) 

    print("--- Pipeline Completed ---")

if __name__ == "__main__":
    run_pipeline()
