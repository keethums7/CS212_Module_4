# fetch_data.py: download the HC3 files into a folder named data.
# Run it once with:  uv run fetch_data.py
import os
import socket
from urllib.request import urlretrieve

try:
    import truststore  # only needed if you get a certificate error (see the lab instructions)
    truststore.inject_into_ssl()
except ImportError:
    pass

socket.setdefaulttimeout(60)  # give up with an error instead of waiting forever

BASE_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/"

os.makedirs("data", exist_ok=True)
for name in ["wiki_csai.jsonl", "open_qa.jsonl"]:
    print("Downloading", name)
    urlretrieve(BASE_URL + name, os.path.join("data", name))
print("Done.")
