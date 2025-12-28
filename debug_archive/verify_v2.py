import sys
import os
import requests

# Add project root to path
sys.path.append(os.path.abspath("/Users/alikaratas/Desktop/49x Final Project"))

def verify_v2():
    print("--- Verifying V2 Components ---")
    
    # 1. Test Scraper
    try:
        from src.ingestion.content_scraper import scrape_article_content
        print("SUCCESS: Imported scrape_article_content.")
        
        # Test on a stable URL (e.g. example.com or a tech blog)
        # Using a very simple one to avoid network blocks in test
        url = "https://www.example.com"
        content = scrape_article_content(url)
        if content:
            print(f"SUCCESS: Scraper fetched content ({len(content)} chars).")
        else:
            print("WARNING: Scraper returned empty content (might be network issue).")
            
    except Exception as e:
        print(f"ERROR: Scraper Check Failed: {e}")
        
    # 2. Test DB Client target
    try:
        from src.storage.supabase_client import SupabaseManager
        db = SupabaseManager()
        # We can't easily introspect the table name without reading code or making a failed query
        # But we trust the code update. 
        # Let's try to fetch 1 row from articles_v2
        if db.client:
            try:
                res = db.client.table("articles_v2").select("id").limit(1).execute()
                print("SUCCESS: Connected to 'articles_v2' (Table Exists).")
            except Exception as e:
                print(f"WARNING: Could not select from 'articles_v2'. Did you create the table? Error: {e}")
    except Exception as e:
         print(f"ERROR: DB Check Failed: {e}")

    # 3. Test Orchestration Import
    try:
        from massive_sync import massive_sync
        print("SUCCESS: massive_sync imported.")
    except Exception as e:
        print(f"ERROR: massive_sync import failed: {e}")

if __name__ == "__main__":
    verify_v2()
