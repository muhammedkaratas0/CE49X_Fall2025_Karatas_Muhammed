import requests
from bs4 import BeautifulSoup
import time
import random

def crawl_sciencedaily_archives(target_count=300):
    """
    Crawls historical pages of ScienceDaily's Civil Engineering news.
    Supports multi-page pagination.
    """
    base_url = "https://www.sciencedaily.com/news/matter_energy/civil_engineering/"
    articles = []
    article_links = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    print(f"--- ScienceDaily Archive Deep Crawl Started ---")
    
    # 1. Gather Links across multiple pages
    # We need roughly 3000-5000 candidates to get 1000 relevant hits after LLM filtering
    page = 1
    max_pages = 100 # Safety cap
    
    while len(article_links) < target_count * 5 and page <= max_pages:
        url = base_url if page == 1 else f"{base_url}page_{page}.htm"
        print(f"Scanning Page {page}: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Reached end of archives or received error {response.status_code}.")
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            page_links = 0
            for l in links:
                href = l['href']
                if "/releases/" in href and href.startswith("/"):
                    full_url = "https://www.sciencedaily.com" + href
                    if full_url not in article_links:
                        article_links.append(full_url)
                        page_links += 1
            
            print(f"Extracted {page_links} new links from page {page}.")
            if page_links == 0:
                break # No more links found
                
            page += 1
            time.sleep(1) # Frequency control
            
        except Exception as e:
            print(f"Error scanning page {page}: {e}")
            break

    print(f"Total candidate links gathered: {len(article_links)}")
    
    # 2. Scrape Content from gathered links
    for url in article_links:
        if len(articles) >= target_count:
            break
            
        try:
            print(f"Scraping [{len(articles)+1}/{target_count}]: {url}")
            res = requests.get(url, headers=headers, timeout=10)
            art_soup = BeautifulSoup(res.content, 'html.parser')
            
            title = art_soup.find('h1', id='headline').text if art_soup.find('h1', id='headline') else "No Title"
            date = art_soup.find('div', id='date_posted').text if art_soup.find('div', id='date_posted') else "Historical"
            text_div = art_soup.find('div', id='story_text') or art_soup.find('div', id='text')
            text = text_div.text if text_div else ""
            
            if text:
                articles.append({
                    "title": title.strip(),
                    "link": url,
                    "published": date.strip(),
                    "summary": text.strip()[:1000], # Send more context to Gemini for better filtering
                    "source_name": "ScienceDaily_Archive"
                })
            
            # Polite delay
            time.sleep(random.uniform(0.3, 0.8)) # Faster now with Paid Tier
                
        except Exception as e:
            print(f"Skip {url}: {e}")
            continue
            
    return articles

if __name__ == "__main__":
    # Test with small batch
    results = crawl_sciencedaily_archives(target_count=5)
    print(f"Fetched {len(results)} samples.")
