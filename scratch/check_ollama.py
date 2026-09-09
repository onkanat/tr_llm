import urllib.request
import json

for host in ["localhost", "127.0.0.1", "192.168.1.14"]:
    try:
        url = f"http://{host}:11434/api/tags"
        print(f"Connecting to {url}...")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            print(f"Ollama connection successful on {host}!")
            print("Available models:")
            for model in data.get("models", []):
                print(f"  - {model['name']}")
            break
    except Exception as e:
        print(f"Failed to connect to {host}: {e}")
