import requests

url = 'http://127.0.0.1:8000/metrics'

response = requests.get(url, timeout=5)
response.raise_for_status()

ketwords = [
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_percent",
    "e2e_request_latency_seconds",
]

for line in response.text.splitlines():
    if line.startswith("#"):
        continue
    if any(word in line for word in ketwords):
        print(line)



