import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Force load from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
env_path = os.path.join(project_root, ".env")

print(f"DEBUG: Looking for .env at: {env_path}")
if os.path.exists(env_path):
    print("DEBUG: .env file found.")
else:
    print("DEBUG: .env file NOT found at expected path.")

load_dotenv(dotenv_path=env_path, override=True)

api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    print(f"DEBUG: GEMINI_API_KEY found (Length: {len(api_key)})")
    genai.configure(api_key=api_key)
    # Using gemini-2.5-flash-lite as requested by the user
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash-lite',
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    )
else:
    print("DEBUG: GEMINI_API_KEY is None.")
    model = None

def analyze_article(title, summary="", full_text=""):
    """
    Sends article title, summary, and optionally full_text to Google Gemini for analysis.
    Checks for English language requirement.
    """
    if not model:
        return {"is_relevant": False, "summary": "API Key eksik."}

    # Prepare content - if full text is available, leverage it but truncate to stay within safe token limits for Flash model
    # Gemini 2.5 Flash has huge context, but let's be reasonable (e.g. 15k chars ~ 4k tokens) to keep it fast.
    content_source = ""
    if full_text and len(full_text) > 200:
        content_source = f"Title: {title}\n\nFull Article Text (Truncated):\n{full_text[:15000]}"
    else:
        content_source = f"Title: {title}\n\nSummary: {summary}"

    system_prompt = """
    You are an expert Civil Engineer and Data Analyst. Analyze the provided article content.
    
    CRITICAL REQUIREMENT 1 (LANGUAGE): The content MUST be in ENGLISH. If it is not in English, set "is_relevant": false immediately.
    
    CRITICAL REQUIREMENT 2 (RELEVANCE): Determine if the article is relevant to BOTH:
    1. Civil Engineering / Construction / Infrastructure
    2. Artificial Intelligence / Digital Technology / Smart Systems
    
    Assign a "relevance_score" from 0 to 100 based on how strong the connection is.
    - 0-40: Irrelevant or tangentially related.
    - 41-70: Moderately relevant (mentions both but maybe superficial).
    - 71-100: Highly relevant (core topic is AI in Civil/Construction).
    
    Output strictly valid JSON:
    {
        "is_relevant": true/false,
        "relevance_score": 0,
        "summary": "1-sentence summary in ENGLISH",
        "category": "Construction Management, Structural, Geotechnical, Transportation, Environmental, Water Resources, Other",
        "ai_tech": "Computer Vision, Machine Learning, Robotics, Generative AI, IoT, NLP, Optimization, None",
        "sentiment": "Positive, Neutral, Negative",
        "language": "English"
    }
    """

    user_content = content_source
    combined_prompt = f"{system_prompt}\n\n{user_content}"

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Use a slightly longer delay between calls even in paid tier to be safe
            time.sleep(0.5)
            
            response = model.generate_content(combined_prompt)
            
            if not response or not response.text:
                print(f"DEBUG: Empty response from Gemini (Attempt {attempt+1})")
                continue

            result_text = response.text
            # Clean up potential markdown blocks
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(result_text)

        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Gemini call failed: {error_msg}")
            
            if "429" in error_msg:
                # If paid tier is active, this shouldn't happen often. 
                # If it does, we wait longer as per Free Tier rules.
                wait_time = 15 * (attempt + 1)
                print(f"DEBUG: Rate limit hit (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # For other errors, return failure immediately
            break
    
    return {
        "is_relevant": False,
        "summary": "Analiz başarısız (Limit veya Bağlantı hatası)."
    }

if __name__ == "__main__":
    # Test
    sample = analyze_article("AI used to detect cracks in bridges", "New computer vision model helps engineers.")
    print(sample)
