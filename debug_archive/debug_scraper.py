import requests

url = "https://news.google.com/rss/articles/CBMioAFBVV95cUxOdlNic3hTcTB3QWJnTzFqUXQzS0xxXzZKUThXWmFLLVVMS3d0Z09fbnpIRldjQzlyMXZDMkc0ektXbjY3M2ItOHZBSVhSTXJPQ1F6NEhhMGNWall1RHVKM2V5UGVQZU9rbWxaNDd5T2FwV2pUYzg0anZISEU4Rkotb1NyTmFibTI5MTdmWnlLZGNRTVFoOVdnUi0tWEFhd2Nu?oc=5"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}

try:
    print(f"Fetching: {url}")
    response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
    print(f"Final URL: {response.url}")
    print(f"Status Code: {response.status_code}")
    print(f"Content Length: {len(response.content)}")
    print(f"Content Preview: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
