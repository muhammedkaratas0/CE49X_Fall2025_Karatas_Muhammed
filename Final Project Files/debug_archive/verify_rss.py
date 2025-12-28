import sys
import os
import feedparser

# Add project root to path
sys.path.append(os.path.abspath("/Users/alikaratas/Desktop/49x Final Project"))

try:
    from src.ingestion.rss_fetcher import RSS_SOURCES
    print(f"SUCCESS: Imported RSS_SOURCES. Count: {len(RSS_SOURCES)}")
except Exception as e:
    print(f"ERROR: Failed to import RSS_SOURCES: {e}")
    sys.exit(1)

print("\n--- Testing RSS Feed Reachability (Sample) ---")
# Test random 3 to ensure network/parsing works
import random
keys = list(RSS_SOURCES.keys())
sample_keys = random.sample(keys, 3)

for k in sample_keys:
    url = RSS_SOURCES[k]
    print(f"Checking {k}...")
    try:
        f = feedparser.parse(url)
        if f.bozo:
             print(f"  -> Warning: Bozo (XML error) but reachable.")
        else:
             print(f"  -> OK. Entries: {len(f.entries)}")
    except Exception as e:
        print(f"  -> FAIL: {e}")

try:
    import massive_sync
    print("\nSUCCESS: massive_sync imported successfully (Syntax Check).")
except Exception as e:
    print(f"ERROR: massive_sync import failed: {e}")
