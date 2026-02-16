import os
import json
import logging
import re

from datetime import datetime
from dotenv import load_dotenv
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
from typing import Optional, Dict
from opik import track


from parse_data import rubric_bullets
from constants import LATEST_PLAYBOOK_FILE, SEED_BULLETS


load_dotenv()


MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_GROQ_2 = 'groq/moonshotai/kimi-k2-instruct'
MODEL_GOOGLE = 'gemini/gemini-2.5-flash'
MODEL_UPV = 'openai/poligpt'

class UPVGrader:
    """UPV B2 Article Grader with ACE framework support."""

    def __init__(self, task_prompt: str, custom_playbook: Optional[Dict] = None):
        """
        Initialize the UPV Grader.

        Args:
            task_prompt: The grading task description
            custom_playbook: Optional pre-existing playbook (for resume)
        """
        self.client = LiteLLMClient(model=MODEL_UPV, api_key=os.getenv("UPV_API_KEY"), api_base="https://api.poligpt.upv.es/v1", temperature=0.1, max_tokens=8192, timeout=60, max_retries=3)

        # Initialize ACE components
        self.generator = Generator(self.client)
        self.reflector = Reflector(self.client)
        self.curator = Curator(self.client)

        # Initialize or load playbook
        if custom_playbook:
            self.playbook = Playbook.from_dict(custom_playbook)
        else:
            self.playbook = Playbook() # Initalize with seed bullets
            for bullet in SEED_BULLETS:
                self.playbook.add_bullet(section='essential', content=bullet)
        
        self.task_prompt = task_prompt
    
    def get_generator(self):
        """Return generator instance."""
        return self.generator

    def get_reflector(self):
        """Return reflector instance."""
        return self.reflector

    def get_curator(self):
        """Return curator instance."""
        return self.curator
