"""Contract-aware inference smoke test; does not require or assume a trained model.
Run against a permitted staging endpoint:
 python smoke_test_inference.py --url http://127.0.0.1:8080/predict --expected-features 30
"""
import argparse
import json
from urllib.request import Request, urlopen

def payload(expected_features: int) -> bytes:
    if not 1 <= expected_features <= 100000:
        raise ValueError("Invalid expected feature dimension")
    return json.dumps({"features": [0.1] * expected_features}).encode("utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--expected-features", type=int, default=30)
    p.add_argument("--timeout", type=float, default=15)
    a = p.parse_args()
    if a.url.startswith("https://") is False and not a.url.startswith("http://127.0.0.1:"):
        p.error("Use HTTPS unless sending to local loopback")
    request = Request(a.url, data=payload(a.expected_features),
                      headers={"Content-Type": "application/json"},
                      method="POST")
    with urlopen(request, timeout=a.timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Unexpected HTTP status {response.status}")
        print(response.read().decode("utf-8"))

if __name__ == "__main__":
    main()
