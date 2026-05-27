"""Quick test: which API keys actually work? (no retries)"""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

PROMPT = "What is 2 + 2? Answer with only the number."

# --- GPT-4.1-mini ---
print("\n--- gpt-4.1-mini ---")
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    r = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user","content":PROMPT}], max_tokens=10, temperature=0.1)
    print(f"  OK: '{r.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- Claude ---
print("\n--- claude-3.5-sonnet ---")
try:
    from anthropic import Anthropic
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    r = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=10, temperature=0.1, messages=[{"role":"user","content":PROMPT}])
    print(f"  OK: '{r.content[0].text.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- Gemini ---
print("\n--- gemini-2.5-flash ---")
try:
    from google import genai
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    r = client.models.generate_content(model="gemini-2.5-flash", contents=PROMPT, config={"temperature":0.1,"max_output_tokens":100})
    text = r.text if r.text else r.candidates[0].content.parts[0].text
    print(f"  OK: '{text.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- Groq (Llama) ---
print("\n--- llama-3.3-70b (Groq) ---")
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":PROMPT}], max_tokens=10, temperature=0.1)
    print(f"  OK: '{r.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- DeepSeek ---
print("\n--- deepseek-v3 ---")
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    r = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":PROMPT}], max_tokens=10, temperature=0.1)
    print(f"  OK: '{r.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- Mistral ---
print("\n--- mistral-large ---")
try:
    from mistralai import Mistral
    client = Mistral(api_key=config.MISTRAL_API_KEY)
    r = client.chat.complete(model="mistral-large-latest", messages=[{"role":"user","content":PROMPT}], temperature=0.1, max_tokens=10)
    print(f"  OK: '{r.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

# --- DeepSeek (Together) ---
print("\n--- deepseek-v3 (Together) ---")
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
    r = client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role":"user","content":PROMPT}], max_tokens=50, temperature=0.1)
    print(f"  OK: '{r.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n--- DONE ---")
