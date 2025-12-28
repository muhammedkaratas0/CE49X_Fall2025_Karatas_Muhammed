import requests

url = "https://news.google.com/rss/articles/CBMioAFBVV95cUxOdlNic3hTcTB3QWJnTzFqUXQzS0xxXzZKUThXWmFLLVVMS3d0Z09fbnpIRldjQzlyMXZDMkc0ektXbjY3M2ItOHZBSVhSTXJPQ1F6NEhhMGNWall1RHVKM2V5UGVQZU9rbWxaNDd5T2FwV2pUYzg0anZISEU4Rkotb1NyTmFibTI5MTdmWnlLZGNRTVFoOVdnUi0tWEFhd2Nu?oc=5"
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(url, headers=headers)
with open("debug_google.html", "w") as f:
    f.write(resp.text)
print("Saved debug_google.html")
