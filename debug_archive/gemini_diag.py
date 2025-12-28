import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.environ.get("GEMINI_API_KEY")

print(f"--- Gemini Diagnostic ---")
print(f"Key Found: {'Yes' if api_key else 'No'}")

if api_key:
    genai.configure(api_key=api_key)
    
    print("\n1. Testing Model List...")
    try:
        models = genai.list_models()
        print("Available Models (Subset):")
        for i, m in enumerate(models):
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
            if i > 10: break
    except Exception as e:
        print(f"Error listing models: {e}")

    print("\n2. Testing Single Direct Call...")
    try:
        # Try the specific model requested by the user
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content("Hello, respond with one word: Success.")
        print(f"Success! Response: {response.text.strip()}")
    except Exception as e:
        print(f"Detailed Error: {e}")
        if "429" in str(e):
            print("\n!!! 429 ERROR DETECTED !!!")
            print("This usually means:")
            print("A) You are still on Free Tier and hit the 15 RPM limit.")
            print("B) The billing hasn't propagated yet (takes ~10-15 mins).")
            print("C) The API Key is not associated with the project where billing is enabled.")
else:
    print("No GEMINI_API_KEY found in .env")
