import requests

url = "http://localhost:11434/api/chat"
payload = {
    "model": self.ollama_model,  # ej: "llama3.1"
    "messages": [
        {"role": "user", "content": enhanced_prompt}
    ],
    "stream": False
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())