import os
from dotenv import load_dotenv, dotenv_values

# Explicit path logic
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env") # in root

print(f"DEBUG: Checking .env at {env_path}")
if os.path.exists(env_path):
    # This reads the file parsing logic from python-dotenv directly
    config = dotenv_values(env_path)
    print("--- KEYS DETECTED IN .ENV FILE ---")
    for key in config.keys():
        print(f"Key: '{key}' (Length: {len(key)})")
    
    if not config:
        print("Empty or unparseable .env file content.")
else:
    print(".env file not found at this location.")
