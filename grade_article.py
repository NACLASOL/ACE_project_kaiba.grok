import json
import os
import re

from dotenv import load_dotenv

from datetime import datetime
from ace import *
from opik import *
from opik.opik_context import update_current_span
from pathlib import Path

from upv_grader import *
from parse_data import extract_rubric
from constants import (
    RUBRIC_FILE,
    TASK_PROMPT,
    OPIK_PROJECT_NAME,
    LOG_DIR,
    ADAPTATION_SUMMARY_FILE,
    LATEST_PLAYBOOK_FILE,
    GRADES_LOG_FILE
)

load_dotenv()

# === CONFIGURATION ===
PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

# Load adaptation summary (if exists).
if ADAPTATION_SUMMARY_FILE.is_file():
    summary = json.load(open(ADAPTATION_SUMMARY_FILE))
else:
    summary = {}

# Load playbook (from last adaptation_summary.json).
if LATEST_PLAYBOOK_FILE.is_file():
    latest_playbook = Playbook.load_from_file(str(LATEST_PLAYBOOK_FILE))
else:
    latest_playbook = None

# Create playbook prompt.
PLAYBOOK_PROMPT = ("\n".join(str(latest_playbook.bullets())) if latest_playbook and latest_playbook.bullets() 
                   else "No evolved playbook found - using default") # Convert to string.

RUBRIC = extract_rubric(RUBRIC_FILE)

TASK_PROMPT = TASK_PROMPT


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

    update_current_span(
        name="validation_span",
        metadata={
            "article_title": article_title,
            "article_length": len(article_text.split()),
            "title_length": len(article_title.split()),
            "cefr_level": "B2",
            "task": "article_grading",
            "timestamp": datetime.now().isoformat()
        }
    )
    # Minimum length check.
    if not article_text or len(article_text) < 50:
        raise ValueError(
            f"Article text too short: {len(article_text)} chars."
            f"Minimum 50 required."
        )
    
    if not article_title or len(article_title) < 3:
        raise ValueError(
            f"Article title too short: {len(article_title)} chars."
            f"Minimum 3 chars required."
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

    update_current_span(
        name="prompt_construction",
        metadata={
            "has_task_prompt": True,
            "has_rubric": RUBRIC is not None,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
        }
    )
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

    update_current_span(
        name="llm_generation",
        metadata={
            "model": client.model,
            "temperature": 0.1,
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
            "prompt_length": len(prompt.split()),
        }
    )
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

    update_current_span(
        name="score_extraction",
        metadata={
            "output_format": "CEFR_5_POINT_SCALE",
            "expected_scores": 5, # TA, CC, GR, LR, OWP.
            "response_length": len(response.final_answer.split()) if response.final_answer else 0,
        },
    )
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
    
    update_current_span(
        name="audit_logging",
        metadata={
            "scores": scores,
            "log_type": "grade_record",
            "timestamp": datetime.now().isoformat(),
            "log_path": str(GRADES_LOG_FILE)
        },
    )
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "article_title": article_title,
        "article_length": len(article_text.split()),
        "scores": scores,
        "justification_length": len(response.reasoning.split()) if response.reasoning else 0,
        "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
    }

    try:
        with open(GRADES_LOG_FILE, "a", encoding="utf-8") as f:
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
            "article_length": len(article_text.split()),
            "playbook_size": len(latest_playbook.bullets()) if latest_playbook else 0,
            "cefr_level": "B2",
            "timestamp": datetime.now().isoformat(),
            "model": client.model
        
        }
    }
        
if __name__ == "__main__":
    # Example usage with test article.
    title = "Being at classroom"
    text = """
In today's digital age, the debate between online and in-person learning continues to intensify. However, attending classes physically
offers undeniable advantages that enhance the educational experience significantly.
First and foremost, physically attending lectures fosters better concentration and engagement. When students are present in a
classroom, they're less likely to succumb to distractions that plague online learning environments. The direct interaction with
professors enables immediate clarification of doubts, promoting deeper understanding of complex concepts.
Moreover, universities provide invaluable resources beyond the classroom. Libraries offer extensive collections and quiet study
spaces, while laboratories facilitate hands-on experiments crucial for scientific disciplines. Campus facilities like study rooms,
computer labs, and sports centers create a holistic learning environment that nurtures both academic and personal development.
To encourage attendance, educators could implement interactive teaching methodologies. Incorporating group discussions, practical
demonstrations, and real-world case studies makes classes more engaging. Additionally, recognition systems that reward consistent
attendance with participation grades motivate students to prioritize physical presence.
In conclusion, while online classes offer convenience, the comprehensive benefits of in-person attendance—from enhanced focus to
access to university facilities—make it the superior choice for serious learners committed to academic excellence.
    """
    
    result = grade_article(title, text)
    print(json.dumps(result, indent=2))