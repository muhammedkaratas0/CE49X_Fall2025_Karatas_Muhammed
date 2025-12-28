import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ingestion.rss_fetcher import fetch_rss_feeds
from src.ingestion.archive_crawler import crawl_sciencedaily_archives
from src.ingestion.google_news_fetcher import fetch_google_news_targeted
from src.ingestion.content_scraper import scrape_article_content
from src.intelligence.llm_processor import analyze_article
from src.storage.supabase_client import SupabaseManager

MAX_ARTICLES_SAFETY_LIMIT = 7500 # V2 Scale
MAX_WORKERS = 10 # Parallel threads

# Thread-safe counters
class RunStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.skipped_dup = 0
        self.skipped_irrelevant = 0
        self.skipped_cat_full = 0
        self.category_counts = {}
        self.existing_urls = set()

stats = RunStats()

def process_single_article(article, db, cat_limit):
    """
    Worker function to process a single article in a thread.
    """
    url = article['link']
    
    # 1. Duplicate Check (Thread-safe read)
    with stats.lock:
        if url in stats.existing_urls:
            stats.skipped_dup += 1
            return
            
    # 2. Analyze Metadata First
    # Assuming analyze_article is thread-safe (it just calls API)
    try:
        analysis = analyze_article(article['title'], article['summary'], full_text="")
    except Exception as e:
        print(f"Error analyzing {url}: {e}")
        return

    # 3. Filter by Relevance
    score = analysis.get("relevance_score", 0)
    is_relevant = analysis.get("is_relevant", False)
    
    if not is_relevant or score < 60:
        with stats.lock:
            stats.skipped_irrelevant += 1
            # Optional: Print less often or just a dot
            # print(f"SKIP (Low Score {score}): {article['title'][:30]}")
        return

    # 4. Scrape Full Text
    print(f" -> [Thread] SCRAPING: {article['title'][:30]} (Score: {score})")
    full_text = scrape_article_content(url)
    
    # 5. Metrics
    word_count = len(full_text.split()) if full_text else 0
    token_count_est = int(word_count * 1.3)

    # Sanitize inputs
    pub_date = article['published']
    if str(pub_date).strip() == "Historical":
        pub_date = "2020-01-01T00:00:00Z"

    enriched_data = {
        "title": article['title'],
        "url": url,
        "published_at": pub_date,
        "source_name": article['source_name'],
        "summary": analysis.get("summary", ""),
        "full_text": full_text, 
        "category": analysis.get("category", "Other"),
        "ai_tech": analysis.get("ai_tech", "None"),
        "sentiment": analysis.get("sentiment", "Neutral"),
        "relevance_score": score,
        "token_count": token_count_est
    }
    
    cat = enriched_data["category"]
    
    # 6. Balance Check & Update (Thread-safe)
    with stats.lock:
        current_cat_count = stats.category_counts.get(cat, 0)
        if current_cat_count >= cat_limit:
            stats.skipped_cat_full += 1
            return
            
        stats.category_counts[cat] = current_cat_count + 1
        stats.processed += 1
        stats.existing_urls.add(url)
        count_display = stats.processed

    # 7. Save (DB client usually thread-safe for simple requests, or we create one per thread if needed. 
    # Supabase-py uses httpx which is fine. But to be super safe, we can lock inserts or just risk it.)
    # Locking inserts to avoid any "connection busy" race conditions in the client if it's not fully async-safe.
    # Actually, let's lock just the insert call to be safe.
    with stats.lock:
        if db.client:
             db.insert_articles([enriched_data])
             
    print(f"[{count_display}] SAVED: {article['title'][:40]}")


def massive_sync(target=1000):
    if target > MAX_ARTICLES_SAFETY_LIMIT:
        target = MAX_ARTICLES_SAFETY_LIMIT

    print(f"--- MASSIVE SYNC (V2 Parallel) STARTED (Target: {target} Successful Items) ---")
    db = SupabaseManager()
    
    # 0. Pre-fetch Context
    print("Step 0: Fetching existing context from DB...")
    all_data = db.fetch_all_articles()
    if all_data:
        for row in all_data:
            stats.existing_urls.add(row.get('url'))
            c = row.get('category', 'Other')
            stats.category_counts[c] = stats.category_counts.get(c, 0) + 1
            
    print(f" -> Found {len(stats.existing_urls)} existing articles.")
    
    # 1. Collection
    print("Step 1: Gathering Candidates...")
    # Parallel fetch of feeds? Maybe later. For now sequential fetch is fast enough compared to analysis.
    gnews_items = fetch_google_news_targeted()
    rss_items = fetch_rss_feeds()
    
    primary_items = gnews_items + rss_items
    
    # We need a lot of candidates to hit 1000 successes given filters
    # Let's say we need 4x candidates
    needed_candidates = target * 4
    
    archive_target = 0
    if len(primary_items) < needed_candidates: 
        archive_target = needed_candidates - len(primary_items)
        if archive_target < 200: archive_target = 200
        
    archives = crawl_sciencedaily_archives(target_count=archive_target)
    all_raw = primary_items + archives
    
    print(f"--- Total raw candidates: {len(all_raw)} ---")
    print(f"--- Starting Parallel Processing ({MAX_WORKERS} Workers) ---")
    
    # 2. Parallel Processing
    CAT_LIMIT = 1200 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, article in enumerate(all_raw):
            # Stop submitting if we reached target? 
            # Hard to know exact "saved" count async, but we can check stats.processed
            if stats.processed >= target:
                print(f"Target of {target} reached. Stopping submission.")
                break
                
            future = executor.submit(process_single_article, article, db, CAT_LIMIT)
            futures.append(future)
            
        # Wait for all
        for f in as_completed(futures):
            pass # exceptions are handled in worker or ignored
            
    print("\n--- MASSIVE SYNC (V2 Parallel) COMPLETE ---")
    print(f"Total Saved (New): {stats.processed}")
    print(f"Skipped - Duplicate: {stats.skipped_dup}")
    print(f"Skipped - Irrelevant: {stats.skipped_irrelevant}")
    print(f"Skipped - Cat Full: {stats.skipped_cat_full}")

if __name__ == "__main__":
    massive_sync(target=1000)
