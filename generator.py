import os
import datetime
from huggingface_hub import InferenceClient

# Modelo que SÍ admite chat_completion
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("Configura tu token con: setx HF_API_TOKEN 'tu_token'")

client = InferenceClient(model=MODEL, token=HF_TOKEN)

def generate_article(topic="AI Tools & Tutorials", lang="en"):
    prompt = (
        f"Write a detailed, SEO-optimized blog post about {topic} in {lang}. "
        f"Include headings, numbered lists, and practical steps. "
        f"Make it at least 700 words long."
    )

    try:
        completion = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.7,
        )
        text = completion.choices[0].message["content"]
    except Exception as e:
        print("❌ Error al generar texto:", e)
        return

    date = datetime.date.today().isoformat()
    path = f"content/{lang}/{topic.replace(' ', '_').lower()}_{date}.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{text}")

    print(f"✅ Artículo generado y guardado en: {path}")

if __name__ == "__main__":
    generate_article("How to use AI Tools for Passive Income", "en")
