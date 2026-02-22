import argparse
import json
import os
from urllib import request, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--endpoint", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: missing API key. Use --api-key or set DASHSCOPE_API_KEY")
        return

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 16,
    }

    req = request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            print(f"HTTP {resp.status}")
            print(resp.read().decode("utf-8", errors="replace"))
    except error.HTTPError as e:
        print(f"HTTP {e.code}")
        print(e.read().decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
