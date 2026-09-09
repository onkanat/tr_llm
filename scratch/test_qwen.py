import urllib.request
import json

url = "http://localhost:11434/api/chat"
payload = {
    "model": "qwen3.5:2b", 
    "messages": [
        {"role": "user", "content": "Merhaba!"}
    ],
    "stream": False
}

try:
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    print("Sending request with 180s timeout...")
    # Long timeout to allow model loading
    with urllib.request.urlopen(req, timeout=180) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("Success! Response:")
        print(res["message"]["content"])
except Exception as e:
    print(f"Error: {e}")
