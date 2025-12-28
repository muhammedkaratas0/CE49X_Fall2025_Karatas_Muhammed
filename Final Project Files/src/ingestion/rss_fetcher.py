import feedparser
import pandas as pd
import ssl
from datetime import datetime

# RSS Sources (Can be moved to config)
RSS_SOURCES = {
    # Civil & Structural (Expanded)
    "ScienceDaily_Civil": "https://www.sciencedaily.com/rss/matter_energy/civil_engineering.xml",
    "ASCE_Civil_Eng": "https://source.asce.org/feed/",
    "Civil_Eng_Mag": "https://www.civilengineeringmag.com/feed",
    "Structural_Eng_Blog": "https://structuralengineering.blog/feed/",
    "TheStructuralEngineer": "https://www.thestructuralengineer.info/news/feed",
    "StructureMag": "https://www.structuremag.org/feed/",
    "BridgeDesignEng": "https://www.bridgeweb.com/rss/news",
    
    # Geotechnical (New)
    "GeoEngineer": "https://www.geoengineer.org/news/feed",
    "Geosynthetics": "https://geosyntheticsmagazine.com/feed/",
    "TunnelTalk": "https://www.tunneltalk.com/rss/news.xml",
    
    # Environmental & Water (New)
    "WaterWorld": "https://www.waterworld.com/rss/topics/technologies.xml",
    "Enviro_Leader": "https://www.environmentalleader.com/feed/",
    "Water_Technology": "https://www.water-technology.net/feed/",
    
    # Transportation (Expanded)
    "Transport_Topics": "https://www.ttnews.com/rss.xml",
    "Railway_Technology": "https://www.railway-technology.com/feed/",
    "Geospatial_World": "https://www.geospatialworld.net/feed/",
    "Traffic_Technology": "https://www.traffictechnologytoday.com/feed",
    "Thinking_Highways": "https://hwy.news/feed/",
    
    # Construction Tech & Management (Existing - Keeping but others will balance)
    "ConstructionDive": "https://www.constructiondive.com/feeds/news/",
    "BIMPlus": "https://www.bimplus.co.uk/feed/",
    "Construction_Enquirer": "https://www.constructionenquirer.com/feed/",
    "ConstructConnect": "https://www.constructconnect.com/blog/rss.xml",
    "ENR_News": "https://www.enr.com/rss/articles",
    "Robotics_Tomorrow": "https://www.roboticstomorrow.com/feeds/news-rss.xml", # Relevant for automation
    
    # AI & Tech General (Filtered by LLM later)
    "TechCrunch_AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "MIT_Tech_Review": "https://www.technologyreview.com/feed/",
    "VentureBeat_AI": "https://venturebeat.com/category/ai/feed/",
    "ZDNet_Robotics": "https://www.zdnet.com/topic/robotics/rss.xml",
    "VentureBeat_BigData": "https://venturebeat.com/category/big-data/feed/"
}

# Fix for SSL certificate issues in some environments
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def fetch_rss_feeds():
    """
    Fetches articles from all defined RSS sources.
    Returns a list of dictionaries with article data.
    """
    print(f"Fetching updates from {len(RSS_SOURCES)} RSS feeds...")
    articles = []

    for source_name, url in RSS_SOURCES.items():
        try:
            print(f"Parsing: {source_name}...")
            feed = feedparser.parse(url)
            
            if feed.bozo:
                print(f"Warning: Potential issue with {source_name} feed parsing.")
                
            for entry in feed.entries[:20]: # Limit to 20 latest per feed (Increased from 10)
                article = {
                    "title": entry.get("title", "No Title"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", datetime.now().isoformat()),
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                    "source_name": source_name,
                    "raw_id": entry.get("id", entry.get("link", ""))
                }
                articles.append(article)
                
        except Exception as e:
            print(f"Error fetching {source_name}: {e}")
            
    print(f"Total articles fetched: {len(articles)}")
    return articles

if __name__ == "__main__":
    # Test run
    data = fetch_rss_feeds()
    for d in data[:3]:
        print(f"- {d['title']} ({d['source_name']})")
