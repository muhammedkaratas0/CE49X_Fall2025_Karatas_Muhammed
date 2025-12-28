from src.storage.supabase_client import SupabaseManager
db = SupabaseManager()
res = db.client.table("articles_v2").select("url").gt("relevance_score", 60).limit(1).execute()
if res.data:
    print(res.data[0]['url'])
else:
    print("No relevant articles found.")
