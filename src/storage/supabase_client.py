import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

class SupabaseManager:
    def __init__(self):
        if not url or not key:
            print("Error: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(url, key)
                print("Connected to Supabase.")
            except Exception as e:
                print(f"Failed to connect to Supabase: {e}")
                self.client = None

    def insert_articles(self, articles):
        """
        Inserts a list of articles into the 'articles_v2' table.
        Avoids duplicates based on 'url' (if unique constraint exists).
        """
        if not self.client:
            print("Supabase client not initialized.")
            return

        if not articles:
            return

        # print(f"Upserting {len(articles)} articles to DB (v2)...")
        try:
            # We use upsert on 'url' or 'id' if possible. 
            # Assuming 'url' is unique in your schema.
            # If plain insert is preferred: self.client.table("articles_v2").insert(articles).execute()
            data = self.client.table("articles_v2").upsert(articles, on_conflict="url").execute()
            # print("Insert successful.")
        except Exception as e:
            print(f"DB Insert Error: {e}")

    def fetch_all_articles(self):
        if not self.client:
            return []
        try:
            response = self.client.table("articles_v2").select("*").execute()
            return response.data
        except Exception as e:
            print(f"DB Fetch Error: {e}")
            return []
            
    def get_existing_urls(self):
        """
        Fetches only the URLs of existing articles to create a local cache for duplicate checking.
        Processing-efficient.
        """
        if not self.client:
            return set()
        try:
            # Pagination might be needed if > 1000 rows, but for now let's hope Supabase py client handles simple select well or we just get first 1000.
            # Supabase API usually limits to 1000 by default. We should loop if we expect > 1000.
            # For this project, let's just grab as many as reasonable or assume the client handles it.
            # Actually, let's try to get a large number.
            response = self.client.table("articles_v2").select("url").execute()
            # If response.data is truncated, we might need range() queries.
            # But let's start simple.
            return {item['url'] for item in response.data} if response.data else set()
        except Exception as e:
            print(f"DB Url Fetch Error: {e}")
            return set()

    def fetch_dataset_articles(self):
        """
        Fetches all articles from the 'articles_data' table with pagination.
        Guarantees retrieving all rows even if > 1000.
        """
        if not self.client:
            return []
        try:
            all_data = []
            offset = 0
            limit = 1000
            
            while True:
                # print(f"Fetching rows {offset} to {offset+limit}...")
                response = self.client.table("articles_data").select("*").range(offset, offset + limit - 1).execute()
                batch = response.data
                if not batch:
                    break
                    
                all_data.extend(batch)
                
                if len(batch) < limit:
                    break
                    
                offset += limit
                
            print(f"Total articles fetched: {len(all_data)}")
            return all_data
        except Exception as e:
            print(f"DB Fetch Error (articles_data): {e}")
            return []            
if __name__ == "__main__":
    # Test connection
    db = SupabaseManager()
