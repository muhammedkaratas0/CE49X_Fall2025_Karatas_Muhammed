import os
from dotenv import load_dotenv

# Force reload of .env
load_dotenv(override=True)

print("--- DIAGNOSTIC RESULT ---")

# 1. Check Gemini Key
gemini_key = os.environ.get("GEMINI_KEY")
if gemini_key:
    # Print first few chars to verify it's loaded but keep secret safe
    print(f"✅ GEMINI_KEY found: {gemini_key[:4]}...****")
else:
    print("❌ GEMINI_KEY NOT found in environment.")
    print("   Please make sure your .env file has a line like: GEMINI_KEY=AIza...")

# 2. Check Supabase Config
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if url and key:
    print(f"✅ Supabase Config found: URL={url[:15]}...")
else:
    print("❌ Supabase URL or KEY missing in .env")

# 3. Check Directory
print(f"Current Working Directory: {os.getcwd()}")
if os.path.exists(".env"):
    print("✅ .env file exists in current directory.")
else:
    print("❌ .env file NOT found. Make sure you are running this from the project root.")
