import json
import os
import re
from dotenv import load_dotenv
from parse_data import extract_rubric
from datetime import datetime
from ace import *
from opik import *
from opik.opik_context import get_current_span_data
from pathlib import Path
from upv_grader import *

load_dotenv()

# === CONFIGURATION ===
RUBRIC_FILE = "parse/internship.enhance_b2_article_writing_rubric_structured.pdf"
PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

summary = json.load(open(LOG_DIR / "adaptation_summary.json"))

# Load evolved playbook (from last adaptation_summary.json)
latest_playbook = Playbook.load_from_file("logs/latest_playbook.json")

# Create playbook prompt.
PLAYBOOK_PROMPT = "\n".join(str(latest_playbook.bullets())) if latest_playbook else "No evolved playbook found - using default" # Convert to string.

RUBRIC = extract_rubric(RUBRIC_FILE)

TASK_PROMPT = """You have just seen the following advertisement in the university magazine:
Share with us what you think about the importance of physically attending classes in-person for university students instead of online classes. We're looking for articles about the benefits of studying in a classroom with a teacher. The best article will be published in our university magazine and the winner will receive a €200 gift card.
You have decided to contribute, Write an ARTICLE in which you:
• explain why physically attending classes can improve learning.
• mention why attending class at a university also provides the student with other resources and facilities.
• suggest ways in which teachers can encourage students to go to class.
Give your article a title. Write your ARTICLE in 180-220 words."""


def extract_scores_from_response(final_answer: str) -> dict:
    """
    Extract CEFR B2 scores from LLM response using flexible regex.

    Args:
        final_answer: LLM response containing score in format:
                      "RESULT,TA:3.0,CC:4.0,GR:3.0,LR:4.0,OWP:4.0".
    
    Returns:
        dict with keys TA, CC, GR, LR, OWP and float values [1.0-5.0].

    Raises:
        ValueError: If regex doesn't match or values out of range.
    """

    """
    attributes={
            "valid_range": "[1.0, 5.0]",
            "all_scores_valid": all(1.0 <= v <= 5.0 for v in scores.values()),
        }
    """

    pattern = r'RESULTS\s*,?\s*TA\s*:?\s*([\d.]+)\s*,\s*CC\s*:?\s*([\d.]+)\s*,\s*GR\s*:?\s*([\d.]+)\s*,\s*LR\s*:?\s*([\d.]+)\s*,\s*OWP\s*:?\s*([\d.]+)'

    match = re.search(pattern, final_answer, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Score extraction failed. Expect format: "
            f"'RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0'\n"
            f"Got: {final_answer[:200]}"
        )

    scores = {
        "TA": float(match.group(1)),
        "CC": float(match.group(2)),
        "GR": float(match.group(3)),
        "LR": float(match.group(4)),
        "OWP": float(match.group(5)),
    }
    """
    opik_context.update_current_span(
        metadata={
            "valid_range": "[1.0, 5.0]",
            "all_scores_valid": all(1.0 <= v <= 5.0 for v in scores.values()),
        }
    )
    """
    
    # Validate all scores with CEFR range [1.0, 5.0].
    for score_name, score_value in scores.items():
        if not (1.0 <= score_value <= 5.0):
            raise ValueError(
                f"Invalid score {score_name} = {score_value}"
                f"Must be in range [1.0, 5.0]"
            )
    
    return scores

@track(project_name=PROJECT_NAME)
def grade_article(article_title: str, article_text: str) -> dict:
    """
    Grade a B2 English article using CEFR rubric and evolved playbook.

    Args:
        article_title: Student's chosen article title (str).
        article_text: Full article body (str).

    Returns:
        dict with:
            - scores: {TA, CC, GR, LR, OWP}.
            - justification: str with model's reasoning.
            - metadata: dict with exectuion context.

    Raises:
        ValueError: If input validation fails or score extraction fails.
    """

    # === VALIDATION ===
    
    """
    attributes={
            "article_title": article_title,
            "article_length": len(article_text),
            "title_length": len(article_title),
            "cefr_level": "B2",
            "task": "article_grading",
            "timestamp": datetime.now().isoformat()
        }
    """

    with start_as_current_span(
        name="validation_span",
        metadata={
            "article_title": article_title,
            "article_length": len(article_text),
            "title_length": len(article_title),
            "cefr_level": "B2",
            "task": "article_grading",
            "timestamp": datetime.now().isoformat()
        }
    ) as validation_span:
        # Minimum length check.
        if not article_text or len(article_text) < 50:
            raise ValueError(
                f"Article text too short: {len(article_text)} chars."
                f"Minimum 50 required."
            )
        
        if not article_title or len(article_title) < 3:
            raise ValueError(
                f"Article title too short: {len(article_title)} chars."
                f"Minimum 3 required."
            )
        
        # Character encoding validation (UTF-8).
        try:
            article_text.encode('utf-8')
            article_text.encode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid character encoding: {str(e)}")
        
    # === PROMPT CONSTRUCTION ===
    """
    attributes={
            "has_task_prompt": True,
            "has_rubric": RUBRIC is not None,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0, 
        }
    """

    with start_as_current_span(
        name="prompt_construction",
        metadata={
            "has_task_prompt": True,
            "has_rubric": RUBRIC is not None,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
        }
    ) as prompt_span:
        # Get playbook strategies as string.
        playbook_str = "\n".join(str(b) for b in latest_playbook.bullets()) if latest_playbook else "No playbook available"
        
        # Construct full prompt.
        prompt = f"""TASK: {TASK_PROMPT}

STUDENT ARTICLE TITLE: {article_title}

STUDENT ARTICLE:

{article_text}

RUBRIC:

{RUBRIC}

INSTRUCTIONS: Grade this article using the rubric. Analyze step-by-step in 'reasoning' field. Output raw JSON ONLY (no markdown): {{"reasoning": "detailed
analysis with evidence from article and rubric", "final_answer":
"RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0"}}

PLAYBOOK STRATEGIES:

{playbook_str}
"""
        
    # === LLM GENERATION ===

    """
    attributes={
            "model": client.model,
            "temperature": 0.1,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
            "prompt_length": len(prompt),
            "article_length": len(article_text),
        }
    """

    with start_as_current_span(
        name="llm_generation",
        metadata={
            "model": client.model,
            "temperature": 0.1,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
            "prompt_length": len(prompt),
            "article_length": len(article_text),
        }
    ) as llm_generation_span:
        try:
            response = generator.generate(
                question=prompt,
                context="",
                playbook=latest_playbook
            )
            
        except Exception as e:
            # Capture error in main trace metadata for observability.
            raise ValueError(f"LLM generation failed: {str(e)}")

    # === SCORE EXTRACTION ===

    """
    attributes={
        "output_format": "CEFR_5_POINT_SCALE",
        "expected_scores": 5, # TA, CC, GR, LR, OWP.
        "response_length": len(response.final_answer) if response.final_answer else 0,
    }
    """

    with start_as_current_span(
        name="score_extraction",
        metadata={
            "output_format": "CEFR_5_POINT_SCALE",
            "expected_scores": 5, # TA, CC, GR, LR, OWP.
            "response_length": len(response.final_answer) if response.final_answer else 0,
        },
    ) as score_extraction_span:
        try:
            scores = extract_scores_from_response(response.final_answer)
        except ValueError as e:
            # Add debugging info to trace.
            raise ValueError(f"Score extraction failed: {str(e)}")
                
    # === AUDIT LOGGING ===
    """
    attributes={
            "log_type": "grade_record",
            "timestamp": datetime.now().isoformat(),
            "log_path": str(LOG_DIR / "grades.log")
        }
    """
    
    with start_as_current_span(
        name="audit_logging",
        metadata={
            "log_type": "grade_record",
            "timestamp": datetime.now().isoformat(),
            "log_path": str(LOG_DIR / "grades.log")
        },
    ) as logging_span:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "article_title": article_title,
            "article_length": len(article_text),
            "scores": scores,
            "justification_length": len(response.reasoning) if response.reasoning else 0,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
        }

        log_path = LOG_DIR / "grades.log"
        try:
            with open(log_path, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except IOError as e:
            raise ValueError(f"Failed to write grade log: {str(e)}")

    # === ANNOTATE MAIN TRACE WITH METADATA ===
    """return {
            "article_title": article_title,
            "article_length": len(article_text),
            "scores": scores,
            "average_score": sum(scores.values()) / len(scores),
            "cefr_level": "B2",
            "grading_model": client.model,
            "playbook_version": "latest",
            "timestamp": datetime.now().isoformat(),
        }
    """

    # === RETURN RESULT ===
    return {
        "scores": scores,
        "justification": response.reasoning if response.reasoning else "",
        "metadata": {
            "article_length": len(article_text),
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
            "cefr_level": "B2",
            "timestamp": datetime.now().isoformat(),
            "model": client.model
        
        }
    }
        
if __name__ == "__main__":
    # Example usage with test article.
    title = "Being at classroom"
    text = """The goal of this article is to emphasize the importance of attending classes physically. 
    With the COVID-19 crisis arrival, every academic exam, task or class had to be done online. 
    The use of platforms like Microsoft Teams and Zoom from teachers increased massively. 
    Remote classes have a huge advantage over physical in terms of availability and time dedication. 
    However, online classes are less effective for improving learning. In my opinion, it is way better 
    attending a lesson in a classroom, where you are face-to-face with your teacher, and your doubts 
    can be solved. Besides, it is harder to see on camera when your professor writes something on the board. 
    In conclusion, we need to make sure the advantages of online classes do not persuade us about attending physically. 
    Moreover, teachers have to attract students somehow. Maybe giving some gifts or rewarding them in other ways."""
    
    result = grade_article(title, text)
    print(json.dumps(result, indent=2))