import requests
from bs4 import BeautifulSoup
import random
import time

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15"
]

def scrape_article_content(url):
    """
    Fetches the full text content of an article URL.
    Returns the text content or an empty string if failed.
    """
    if not url:
        return ""

    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }

    try:
        # Timeout 5 seconds to be fast
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            print(f"Scrape Failed {response.status_code}: {url}")
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')

        # Heuristic to find the main article text
        # 1. Look for <article> tags
        article_body = soup.find('article')
        
        # 2. If not, look for main content divs
        if not article_body:
            article_body = soup.find('div', {'class': ['entry-content', 'post-content', 'article-content', 'main-content']})
            
        # 3. Fallback: Just grab all paragraphs if not too many navigational ones
        if article_body:
            paragraphs = article_body.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        # Filter and join paragraphs
        text_content = []
        for p in paragraphs:
            text = p.get_text().strip()
            # Heuristic: skip very short lines (menus, navigation, credits)
            if len(text) > 40:
                text_content.append(text)

        full_text = "\n\n".join(text_content)
        
        return full_text

    except Exception as e:
        print(f"Scrape Error {url}: {e}")
        return ""

# --- SELENIUM ADDITION ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    
    def scrape_with_selenium(url):
        """
        Uses Headless Chrome to scrape. Essential for Google News redirects.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        # Anti-detection
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")
        
        driver = None
        try:
            # Install/Get Driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set timeout
            driver.set_page_load_timeout(30)
            
            # Go
            driver.get(url)
            
            # Wait for JS redirect if it's a google link
            if "google.com" in url:
                time.sleep(3) # Initial wait
                # Check if we are still on google
                if "consents" in driver.current_url or "google.com" in driver.current_url:
                     # Maybe wait a bit more
                     time.sleep(2)
            
            # Now parse content
            # We can pass driver.page_source to BS4 for better parsing than innerText
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Re-use heuristics
            article_body = soup.find('article')
            if not article_body:
                article_body = soup.find('div', {'class': ['entry-content', 'post-content', 'article-content', 'main-content']})
            
            text_content = []
            if article_body:
                paragraphs = article_body.find_all('p')
            else:
                paragraphs = soup.find_all('p')
                
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 40:
                    text_content.append(text)
                    
            full_text = "\n\n".join(text_content)
            return full_text

        except Exception as e:
            print(f"Selenium Error {url}: {e}")
            return ""
        finally:
            if driver:
                driver.quit()

except ImportError:
    def scrape_with_selenium(url):
        print("Selenium not installed.")
        return ""

if __name__ == "__main__":
    # Test
    test_url = "https://www.sciencedaily.com/releases/2023/10/231017123456.htm" 
    print(f"Testing Scraper on: {test_url}")

