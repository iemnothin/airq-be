# scripts/generate_docs.py
import requests
import json

SWAGGER_URL = "http://localhost:8000/openapi.json"
OUTPUT_FILE = "backend/docs/api_generated.md"

def generate_markdown_from_openapi():
    data = requests.get(SWAGGER_URL).json()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# 📘 Auto-generated API Docs\n\n")
        for path, methods in data["paths"].items():
            for method, details in methods.items():
                f.write(f"### `{method.upper()} {path}`\n")
                f.write(f"**Summary:** {details.get('summary', '-')}\n\n")
                f.write("---\n\n")

if __name__ == "__main__":
    generate_markdown_from_openapi()
    print(f"✅ API documentation generated at {OUTPUT_FILE}")
