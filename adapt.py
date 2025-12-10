import re
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from opik import *
from opik.opik_context import update_current_span
from parse_data import train_samples, test_samples, one_sample
from upv_grader import *

# === CONFIGURATION ===
PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

EPOCHS = 1
LOG_DIR = Path("logs")
EPOCHS_DIR = LOG_DIR / "epochs"
LOG_DIR.mkdir(exist_ok=True)
EPOCHS_DIR.mkdir(exist_ok=True)

mismatches_log = []
deltas_log = {}
playbooks_log = []

Path('logs').mkdir(exist_ok=True) # Ensure directory.

Path('logs/epochs').mkdir(exist_ok=True) # Create 'epochs' directory.

def log_failure(prompt_snippet: str, error_message: str) -> None:
    """
    Log generation/parsing failures for debugging.

    Args:
        prompt_snipper: Chars of failed prompt.
        error message: Exception message.
    """
    log_path = LOG_DIR / "epoch_failure.log"
    with open('logs/epoch_failure.log', 'w') as f:
        f.write(f"---\n{datetime.now()}\nError: {error_message}\n\n{prompt_snippet}\n---\n")

@track(
    project_name=os.getenv("OPIK_PROJECT_NAME"),
    metadata={
        "task": "adaptation_loop",
        "component": "score_extraction"
    }
)
def extract_scores_from_response(final_answer: str) -> dict:
    """
    Extract CEFR scores from LLM response using corrected felxible regex

    Args:
        final_answer: LLM response contianing scores.

    Returns:
        dict: {TA, CC, GR, LR, OWP} with float values [1.0-5.0]

    Raises:
        ValueError: If regex doesn't match or values out of range.
    """

    pattern = r'RESULTS\s*,?\s*TA\s*:?\s*([\d.]+)\s*,\s*CC\s*:?\s*([\d.]+)\s*,\s*GR\s*:?\s*([\d.]+)\s*,\s*LR\s*:?\s*([\d.]+)\s*,\s*OWP\s*:?\s*([\d.]+)'

    match = re.search(pattern, final_answer, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Score extraction failed. Expected format: 'RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0'\n"
            f"Got: {final_answer[:300]}"
        )
    
    scores = {
        'TA': float(match.group(1)),
        'CC': float(match.group(2)),
        'GR': float(match.group(3)),
        'LR': float(match.group(4)),
        'OWP': float(match.group(5)),
    }

    # Validate all scores in CEFR range.
    for score_name, score_value in scores.items():
        if not (1.0 <= score_value <= 5.0):
            raise ValueError(
                f"Score {score_name}={score_value} outside valid range [1.0, 5.0]"
            )
        
    return scores

@track(
        project_name=PROJECT_NAME,
        metadata={
            "task": "adaptation_loop",
            "component": "main_adaptation",
        }     
    )
def adaptation_epoch(epoch: int, sample_set) -> dict:
    """Run a single adaptation epoch with tracing."""

    epoch_mismatches = []
    epoch_deltas_count = 0

    # === SAMPLE PROCESSING ===
    for sample_idx, sample in enumerate(sample_set):
        # === GENERATE ===
        update_current_span(
            name="sample_processing",
            metadata={
                "epoch": epoch,
                "sample_id": sample_idx,
            }
        )
        update_current_span(
            name="generation",
        )
        try:
            prompt = sample['input'] + "\n**PLAYBOOK STRATEGIES:**\n" + str(playbook)
            
            result = generator.generate(
                question=prompt,
                context="",
                playbook=playbook
            )

            update_current_span(
                metadata={"prompt_length": len(prompt.split()), "reasoning_length": len(result.reasoning.split()), "success": True}
                )
        
        except Exception as e:
            log_failure(sample['input'][:500], str(e))
            # Fallback scores.
            ai_scores = {
                'TA': 0.0,
                'CC': 0.0,
                'GR': 0.0,
                'LR': 0.0,
                'OWP': 0.0,
            }

            mismatch = 3.0
            epoch_mismatches.append(mismatch)
            continue
            
        # === PARSE SCORES ===
        update_current_span(
            name=f"parse_scores_sample_{sample_idx}",
            metadata={
                    "epoch": epoch,
                    "sample_index": sample_idx,
                    "output_length": len(result.final_answer) if result.final_answer else 0,
            }
        )
        try:
            ai_scores = extract_scores_from_response(result.final_answer)
        except ValueError as e:
            log_failure(result.final_answer, f"Parse failed: {str(e)}")
            ai_scores = {
                'TA': 0.0,
                'CC': 0.0,
                'GR': 0.0,
                'LR': 0.0,
                'OWP': 0.0
            }
            log_failure(result.final_answer[:500], "Regex parse failed - no match.")
        
        # === COMPUTE MISMATCH ===
        update_current_span(
            name="mismatch_computation_sample_{sample_idx}",
            metadata={
            }
        )
        human = sample['ground_truth']
        mismatch = mean(abs(ai_scores[k] - human[k]) for k in human)

        epoch_mismatches.append(mismatch)

        update_current_span(
            metadata={
                "mismatch": mismatch,
                "human_scores": human,
                "ai_scores": ai_scores, 
                "score_differences": {k: abs(ai_scores[k] - human[k]) for k in human},
            }
        )

        feedback = f"Human: {human}. AI: {ai_scores}. Avg mismatch: {mismatch:.2f}"

        # === REFLECT ===
        update_current_span(
            name="reflection",
            metadata={
                "epoch": epoch,
                "sample_index": sample_idx,
                "mismatch": mismatch 
            }
        )
        try:
            reflection = reflector.reflect(
                question="",
                generator_output=result,
                playbook=playbook,
                ground_truth=str(human),
                feedback=feedback,
            )
        except Exception as e:
            log_failure(feedback, f"Reflect failed: {str(e)}")
            reflection - "" # Skip the reflection.

        # === CURATE & APPLY DELTA ===
        update_current_span(
            name="curation",
            metadata={
                    "epoch": epoch,
                    "sample_index": sample_idx,
            }
        )
        try:
            deltas = curator.curate(
                reflection=reflection,
                playbook=playbook,
                question_context="",
                progress=f"Epoch {epoch}, Sample {sample_idx}",
            )

            playbook.apply_delta(deltas.delta)

            delta_json = deltas.delta.to_json()
            new_deltas = [
                str(d) for d in delta_json.get('operations', [])
                if 'ADD' in str(d) or 'TAG' in str(d)
            ]

            epoch_deltas_count += len(new_deltas)
            deltas_log.update({
                'epoch': epoch,
                'sample': sample_idx,
                'deltas': new_deltas,
            })

            update_current_span(
                metadata={"delta_count": len(new_deltas), "play_size_after": len(playbook.bullets()), "delta_operations": new_deltas[:3] } # Log first 3.
            )
        
        except Exception as e:
                log_failure(feedback, f"Curate failed: {str(e)}")
                update_current_span(
                metadata={"status": "failure", "error": str(e)}
            )
                raise


    # === EPOCH SUMMARY ===
    avg_mismatch = mean(epoch_mismatches) if epoch_mismatches else 0.0
    mismatches_log.append(avg_mismatch)
    playbooks_log.append(playbook.bullets())

    epoch_result = {
            'epoch': epoch,
            'avg_mismatch': avg_mismatch,
            'playbook_size': len(playbook.bullets()),
            'deltas_applied': epoch_deltas_count,
            'samples_processed': len(sample_set),
            'timestamp': datetime.now().isoformat(),
    }

    # Write epoch summary.
    epoch_path = EPOCHS_DIR / f"epoch-{epoch:02d}.json"
    with open(epoch_path, 'w') as f:
        json.dump(epoch_result, f, indent=2)

    # Annotate main trace.
    update_current_span(
            metadata={
                'epoch': epoch,
                'avg_mismatch': avg_mismatch,
                'playbook_size': len(playbook.bullets()),
                'deltas_applied': epoch_deltas_count,
                'timestamp': datetime.now().isoformat()    
            }
    )

    print(f"Epoch {epoch}: Avg mismatch {avg_mismatch:.3f}, "
            f"Playbook size {len(playbook.bullets())}"
            f"Deltas {epoch_deltas_count}")
    
    return epoch_result
    
def main():
    """
    Execute full adaptation loop across all epochs.
    """

    for epoch in range(EPOCHS):
        try:
            epoch_result = adaptation_epoch(epoch, test_samples)
            print(f"✅ Epoch {epoch} completed successfully")
        except Exception as e:
            print(f"❌ Epoch {epoch} failed: {str(e)}")
            raise

    start_as_current_span(
        name="aggregation",
        metadata={
            "total_epochs": EPOCHS,
            "total_avg_mismatch": mean(mismatches_log) if mismatches_log else 0.0,
            "total_deltas": len(deltas_log),
        }
    )
    summary_path = LOG_DIR / "adaptation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'mismatches_log': mismatches_log,
            'deltas_log': deltas_log,
            'total_epochs': EPOCHS,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    playbook.save_to_file(str(LOG_DIR / "latest_playbook.json"))

    print("✅ Adaptation Complete!")
    print(
        f"Average mismatch across all epochs: {mean(mismatches_log):.3f}"
    )
    print(f"Total deltas applied: {len(deltas_log)}")
    print(f"Final playbook size: {len(playbook.bullets())} bullets")

if __name__ == "__main__":
    main()
