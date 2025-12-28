import base64
import re

url = "https://news.google.com/rss/articles/CBMioAFBVV95cUxOdlNic3hTcTB3QWJnTzFqUXQzS0xxXzZKUThXWmFLLVVMS3d0Z09fbnpIRldjQzlyMXZDMkc0ektXbjY3M2ItOHZBSVhSTXJPQ1F6NEhhMGNWall1RHVKM2V5UGVQZU9rbWxaNDd5T2FwV2pUYzg0anZISEU4Rkotb1NyTmFibTI5MTdmWnlLZGNRTVFoOVdnUi0tWEFhd2Nu?oc=5"

# Extract the ID part
match = re.search(r'articles/([^?]+)', url)
if match:
    cbm_str = match.group(1)
    print(f"CBM String: {cbm_str}")
    
    # Needs padding
    padding = len(cbm_str) % 4
    if padding:
        cbm_str += "=" * (4 - padding)
        
    try:
        decoded = base64.urlsafe_b64decode(cbm_str)
        print(f"Decoded (Raw): {decoded}")
        
        # Look for http url inside
        # It's binary, so we need to be careful printing
        try:
             # Try to find http string
             decoded_str = decoded.decode('latin1') # usage of latin1 preserves bytes
             url_match = re.search(r'(https?://[a-zA-Z0-9./-]+)', decoded_str)
             if url_match:
                 print(f"Found URL: {url_match.group(1)}")
             else:
                 print("No URL found in decoded string.")
        except Exception as e:
            print(f"String Decode Error: {e}")
            
    except Exception as e:
        print(f"Base64 Error: {e}")
