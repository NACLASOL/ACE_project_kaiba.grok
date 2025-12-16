import os
import json
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
from opik import track
from parse_data import rubric_bullets


load_dotenv()


MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_GROQ_2 = 'groq/moonshotai/kimi-k2-instruct'
MODEL_GOOGLE = 'gemini/gemini-2.5-flash'

client = LiteLLMClient(model=MODEL_GOOGLE, api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1, max_tokens=8192)

# Optional custom curator prompt; used in case default curator prompt fails.
custom_curator_prompt = '''
Progress: {progress}
Stats: {stats}
Reflection: {reflection}
Playbook: {playbook}
Context: {question_context}
Decide what changes to make. Return valid and parseable JSON with delta operations.
'''

generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)
playbook = Playbook()

SEED_BULLETS = [
    "OUTPUT FORMAT: Respond with a raw JSON object ONLY: {\"reasoning\": \"...\", \"bullet_ids\": [...], \"final_answer\": \"RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0\"}. NO Markdown fences (```json), NO extra text, NO explanations outside JSON.",
    "CRITICAL: In 'final_answer', use EXACTLY: RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0 at end. Ensure a valid and parseable JSON overall.",
    "JSON RULES: Use \\n for line breaks in 'reasoning'. Escape quotes properly. No apostrophes in strings.",
    "CHAIN OF THOUGHT: 1. Quote evidence. 2. Map to rubric level. 3. Justify score. Put in 'reasoning'.",
    "DETERMINISM: Ignore creativity; stick to rubric descriptors verbatim.",
    "RUBRIC BULLETS: Use these for grading.", # rubric_bullets will be added dynamically.
]

for bullet in SEED_BULLETS:
    playbook.add_bullet(bullet, content='essential')