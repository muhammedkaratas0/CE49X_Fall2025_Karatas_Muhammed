import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.storage.supabase_client import SupabaseManager
from src.ingestion.content_scraper import scrape_article_content, scrape_with_selenium

# Browsers are heavy, so reduce concurrency
MAX_WORKERS = 4

def backfill_full_text():
    print("--- STARTING BACKFILL: FULL TEXT (SELENIUM POWERED) ---")
    db = SupabaseManager()
    
    print("Fetching RELEVANT articles from DB (Score > 60) with missing text...")
    
    try:
        # Fetch items where full_text is null OR empty AND relevance_score > 60
        # Supabase-py chaining for OR is tricky, so let's do two queries or just fetch relevant ones and filter in python if needed.
        # But we can do: select * from articles_v2 where relevance_score > 60
        
        res = db.client.table("articles_v2") \
            .select("id, url, title, full_text") \
            .gt("relevance_score", 60) \
            .execute()
            
        all_relevant = res.data if res.data else []
        
        # Filter for missing full_text in Python to be safe/easy
        items = [
            x for x in all_relevant 
            if not x.get('full_text') or len(x.get('full_text', '')) < 50
        ]

    except Exception as e:
        print(f"Error fetching candidates: {e}")
        return

    print(f"Found {len(items)} RELEVANT articles with missing full_text.")
    if not items:
        print("Nothing to backfill.")
        return

    # 2. Parallel Scrape & Update
    processed = 0
    updated = 0
    failed = 0
    
    lock = threading.Lock()
    
    def process_item(item):
        nonlocal processed, updated, failed
        url = item['url']
        # print(f"Processing: {url}")
        
        # Try basic first? No, we know they failed. Use Selenium directly.
        # But for non-google links basic might be faster.
        # Let's use Selenium for all backfill since we know these are the "failed" ones usually.
        try:
            text = scrape_with_selenium(url)
        except Exception as e:
            text = ""
            print(f"Worker Error: {e}")
        
        with lock:
            processed += 1
            idx = processed
            
        if text and len(text) > 100:
            # Update DB
            try:
                # Calculate tokens
                word_count = len(text.split())
                token_count = int(word_count * 1.3)
                
                db.client.table("articles_v2").update({
                    "full_text": text,
                    "token_count": token_count
                }).eq("id", item['id']).execute()
                
                with lock:
                    updated += 1
                    print(f"[{idx}/{len(items)}] UPDATED: {item['title'][:30]}")
            except Exception as e:
                print(f"Error updating ID {item['id']}: {e}")
                with lock:
                    failed += 1
        else:
            with lock:
                failed += 1
                print(f"[{idx}/{len(items)}] FAILED/EMPTY: {item['title'][:30]}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for f in as_completed(futures):
            pass

    print(f"--- BACKFILL COMPLETE: {updated} Updated, {failed} Failed ---")

if __name__ == "__main__":
    backfill_full_text()
