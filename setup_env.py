import os
import shutil

def setup_env():
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("Created .env file from template.")
            print("Please open .env and fill in your SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY.")
        else:
            print("Error: .env.example not found.")
    else:
        print(".env file already exists.")

if __name__ == "__main__":
    setup_env()
