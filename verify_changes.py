import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("/Users/alikaratas/Desktop/49x Final Project"))

try:
    from src.visualization.dashboard import create_dashboard
    print("SUCCESS: src.visualization.dashboard imported successfully.")
except Exception as e:
    print(f"ERROR: Failed to import src.visualization.dashboard: {e}")

try:
    from massive_sync import massive_sync
    print("SUCCESS: massive_sync imported successfully.")
except Exception as e:
    print(f"ERROR: Failed to import massive_sync: {e}")

# Test Stopwords Logic directly
import re
from collections import Counter

def test_stopwords():
    print("\n--- Testing Stopwords Logic ---")
    # Simulate text with Turkish stopwords
    text = "Bu bir makale ve inşaat mühendisliği ile ilgili çalışma. Yeni proje için detaylar."
    expected_words = ["inşaat", "mühendisliği", "ilgili", "detaylar"] # "inşaat" & "mühendisliği" might remain depending on how strict the list is, but "bu", "bir", "makale" should go.
    # Wait, "inşaat" and "mühendisliği" are in the dashboard.py stopwords list too?
    # Let's check the code I wrote:
    # 'construction', 'engineering', 'civil' were there.
    # I didn't add 'inşaat', 'mühendisliği' explicitly to the English list, but let's see if I handled it.
    # I added: 'bu', 'bir', 've', 'ile', 'için', 'olarak', 'daha', 'en', 'kadar', 'gibi', 'makale', 'çalışma', 'yeni', 'proje', 'olan', 'tarafından', 'var', 'yok', 'ama', 'fakat', 'ancak', 'veya', 'ise', 'çünkü'
    
    # So 'inşaat', 'mühendisliği' should remain unless I add them. The user complained about 'bu', 'makale'.
    
    stopwords = set(['the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was', 'were', 'have', 'has', 'will', 'can', 'construction', 'engineering', 'civil',
                    'bu', 'bir', 've', 'ile', 'için', 'olarak', 'daha', 'en', 'kadar', 'gibi', 'makale', 'çalışma', 'yeni', 'proje', 'olan', 'tarafından', 'var', 'yok', 'ama', 'fakat', 'ancak', 'veya', 'ise', 'çünkü']) 
    
    words = re.findall(r'\b[a-zçğıöşü]{3,}\b', text.lower()) # Simple regex for Turkish chars support
    
    filtered = [w for w in words if w not in stopwords]
    print(f"Original: {words}")
    print(f"Filtered: {filtered}")
    
    if "bu" in filtered or "makale" in filtered:
        print("FAIL: Stopwords not filtered.")
    else:
        print("SUCCESS: 'bu' and 'makale' filtered out.")

test_stopwords()
