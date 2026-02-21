import requests

r = requests.get("http://localhost:11434")
print(r.text)
