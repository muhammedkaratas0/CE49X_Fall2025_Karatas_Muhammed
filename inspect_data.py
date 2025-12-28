import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("Error: SUPABASE_URL or SUPABASE_KEY not found.")
    exit()

supabase = create_client(url, key)

print("--- Inspecting 'articles_data' Table ---")

try:
    # Fetch a few rows to understand the schema
    response = supabase.table("articles_data").select("*").limit(5).execute()
    data = response.data
    
    if not data:
        print("Table 'articles_data' is empty or does not exist.")
    else:
        print(f"Found {len(data)} rows (showing 1st):")
        first_row = data[0]
        print("Columns found:")
        for col, val in first_row.items():
            print(f" - {col}: {type(val).__name__} (Example: {str(val)[:50]}...)")
            
except Exception as e:
    print(f"Error: {e}")
