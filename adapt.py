import os
import re
import json
import time

from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List
from opik import *
from opik.opik_context import update_current_span

from checkpoint import AdaptationCheckpoint
from parse_data import samples
from upv_grader import *
from constants import (
    OPIK_PROJECT_NAME,
    CHECKPOINT_ENABLED,
    LOG_DIR,
    EPOCHS_DIR,
    EPOCH_FAILURE_LOG,
    TASK_PROMPT,
    DEBUG_ADAPT,
    EPOCHS,
)

mismatches_log = []
deltas_log = []
playbooks_log = []

Path(EPOCHS_DIR).mkdir(exist_ok=True) # Create 'epochs' directory.


def log_failure(prompt_snippet: str, error_message: str) -> None:
    """
    Log generation/parsing failures for debugging.

    Args:
        prompt_snipper: Chars of failed prompt.
        error message: Exception message.
    """
    with open(EPOCH_FAILURE_LOG, 'a') as f:
        f.write(f"---\n{datetime.now()}\nError: {error_message}\n\n{prompt_snippet}\n---\n")


def load_training_samples(samples_path: str = "logs/samples/train_samples.json") -> List[Dict]:
    """Load training samples from JSON file."""
    samples_path = Path(samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(f"Training samples not found: {samples_path}")
    
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"✅ Loaded {len(samples)} training samples from {samples_path}")
    return samples
    

@track(
    project_name=OPIK_PROJECT_NAME,
    metadata={
        "task": "adaptation_loop",
        "component": "score_extraction"
    }
)
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
    
    # Validate all scores with CEFR range [1.0, 5.0].
    for score_name, score_value in scores.items():
        if not (1.0 <= score_value <= 5.0):
            raise ValueError(
                f"Invalid score {score_name} = {score_value}"
                f"Must be in range [1.0, 5.0]"
            )
    
    return scores


@track(
        project_name=OPIK_PROJECT_NAME,
        metadata={
            "task": "adaptation_loop",
            "component": "main_adaptation",
        }     
    )
def adaptation_epoch(grader: UPVGrader, samples, epoch: int) -> dict:
    """
    Main adaptation that runs a single adaptation epoch.

    Args:
        grader: Grader used in the adaptation program
        samples: The samples used for the adaptation process
        epoch: How many epochs the adaptation program will run.

    """

    epoch_mismatches = []
    epoch_deltas_count = 0

    # Get ACE components from grader.
    generator = grader.get_generator()
    reflector = grader.get_reflector()
    curator = grader.get_curator()
    playbook = grader.playbook

    # === SAMPLE PROCESSING ===
    for sample_idx, sample in enumerate(samples):
        # === GENERATE ===
        update_current_span(
            name=f"sample_processing_{sample_idx}",
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
            
            if DEBUG_ADAPT:
                print(f"\n🔍 DEBUG PROMPT ANALYSIS:")
                print(f"  - Total prompt length: {len(prompt)} characters")
                print(f"  - Prompt word count: {len(prompt.split())} words")
                print(f"  - Playbook string length: {len(str(playbook))} characters")
                print(f"  - Estimated tokens (rough): ~{len(prompt.split()) * 1.3:.0f}")

                try:
                    bullets = playbook.bullets()
                    print(f"  - Playbook bullet count: {len(bullets)}")
                except:
                    print(f"  - Playbook bullet count: UNKNOWN")
                print()

            result = generator.generate(
                question=prompt,
                context="",
                playbook=playbook
            )

            update_current_span(
                metadata={"prompt_length": len(prompt.split()), "reasoning_length": len(result.reasoning.split()), "success": True}
                )
        
        except TypeError as e:
            print(f"❌ LLM API returned empty response (likely token limit exceeded)")
            print(f"   Prompt length: {len(prompt)} chars, ~{len(prompt.split())*1.3:.0f} tokens")
            raise ValueError(f"LLM API failure - empty response. Prompt may exceed token limit.") from e

        except Exception as e:
            print(f"❌ Generation failed: {type(e).__name__}: {str(e)}")
            log_failure(sample['input'][:500], f"Generation: {str(e)}")
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
        except (ValueError, AttributeError) as e:
            print(f"❌ Sample {sample_idx}: Score extraction failed")
            answer_snippet = (result.final_answer[:500] if result.final_answer else "<Empty>")
            log_failure(answer_snippet, f"Parse failed: {str(e)}")
            continue
        
        # === COMPUTE MISMATCH ===
        update_current_span(
            name=f"mismatch_computation_sample_{sample_idx}",
            metadata={
                "epoch": epoch,
                "sample_index": sample_idx
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

        feedback = f"""
Human scores: {human}. 
AI scores: {ai_scores}.
Avg mismatch: {mismatch:.2f}
"""

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
                question=sample['input'],
                generator_output=result,
                playbook=playbook,
                ground_truth=str(human),
                feedback=feedback,
            )
        except Exception as e:
            print(f"❌ Reflection failed: {type(e).__name__}: {str(e)}")
            log_failure(feedback, f"Reflect failed: {str(e)}")
            continue

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

            if not deltas or not deltas.delta:
                print("⚠️  Curator returned no deltas")
                continue

            playbook.apply_delta(deltas.delta)

            delta_json = deltas.delta.to_json()
            new_deltas = [
                str(d) for d in delta_json.get('operations', [])
                if 'ADD' in str(d) or 'TAG' in str(d)
            ]

            epoch_deltas_count += len(new_deltas)
            deltas_log.append({
                'epoch': epoch,
                'sample': sample_idx,
                'deltas': new_deltas,
            })

            update_current_span(
                metadata={"delta_count": len(new_deltas), "play_size_after": len(playbook.bullets()), "delta_operations": new_deltas[:3] } # Log first 3.
            )
        
        except Exception as e:
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
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
            'samples_processed': len(samples),
            'timestamp': datetime.now().isoformat(),
    }

    # Write epoch summary.
    epoch_path = EPOCHS_DIR / f"epoch-{epoch:02d}.json"
    try:
        with open(epoch_path, 'w') as f:
            json.dump(epoch_result, f, indent=2)
    except IOError as e:
        print(f"⚠️ Failed to write epoch file: {e}")

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
    

def adaptation_with_checkpoint(epochs: int = EPOCHS):
    """
    Main adaptation loop with checkpoint/resume support.

    Args:
        epochs: Total number of epochs to run
    """

    print("="*70)
    print("🚀 ADAPTIVE CONTEXT ENGINEERING - CHECKPOINT MODE")
    print("="*70)

    # Initialize checkpoint manager
    checkpoint = AdaptationCheckpoint()

    # Load training samples
    print("\n📂 Loading training samples...")
    train_samples = load_training_samples()
    print(f"✅ Loaded {len(train_samples)} training articles")

    # Check for existing checkpoint
    existing_checkpoint = checkpoint.load()

    if DEBUG_ADAPT and existing_checkpoint:
        print("🔍 DEBUG: Checkpoint playbook structure:")
        print(json.dumps(existing_checkpoint['current_playbook'], indent=2)[:500])

    if existing_checkpoint and checkpoint.should_resume():
        # RESUME MODE
        print("\n" + "="*70)
        print("🔄 RESUMING FROM CHECKPOINT")
        print("="*70)

        checkpoint.print_status()

        # Validate checkpoint matches current configuration.
        if existing_checkpoint['total_epochs'] != epochs:
            print(f"⚠️ WARNING: Checkpoint has {existing_checkpoint['total_epochs']} epochs")
            print(f"        but current config specifies {epochs} epochs")
            response = input("\n    Continue with checkpoint's epoch count? (y/n): ")
            if response.lower() != 'y':
                print("Aborting. Delete checkpoint manually to start fresh.")
                return
            epochs = existing_checkpoint['total_epochs']

        if existing_checkpoint['training_samples_count'] != len(train_samples):
            print(f"⚠️ WARNING: Training samples count changed!")
            print(f"        Checkpoint: {existing_checkpoint['training_samples_count']}")
            print(f"        Current: {len(train_samples)}")
            response = input("\n    Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting. Delete checkpoint manually to start fresh")
                return
            epochs = existing_checkpoint['total_epochs']
            
        # Increment resume counter
        existing_checkpoint['resume_count'] += 1
        existing_checkpoint['resumed_at'] = datetime.now().isoformat()
        existing_checkpoint['interrupted'] = False # Clear interrupted flag.
        checkpoint.save()

        # Initialzie grader with restored playbook
        print("\n🔧 Initializing grader with checkpoint playbook...")
        grader = UPVGrader(
            task_prompt=TASK_PROMPT,
            custom_playbook=existing_checkpoint['current_playbook']
        )

        start_epoch = checkpoint.get_resume_epoch()
        print(f"▶️ Resuming from epoch {start_epoch}\n")

    else:
        # === FRESH START MODE ===
        if existing_checkpoint and checkpoint.is_complete():
            print("\n✅ Previous run already completed all epochs!")
            print("Delete checkpoint to start fresh, or increase EPOHCS to continue.")
            response = input("\nStart fresh (y/n): ")      
            if response.lower() != 'y':
                return
            checkpoint.delete()
            
        print("\n🆕 Starting fresh adaptation run")

        # Initailize grader with seed playbook
        grader = UPVGrader(task_prompt=TASK_PROMPT)

        if DEBUG_ADAPT:
            print("🔍 DEBUG: Fresh playbook.to_dict() structure:")
            print(json.dumps(grader.playbook.to_dict(), indent=2)[:500])   
        

        # Initialize checkpoint
        checkpoint.initialize(
            total_epochs=epochs,
            initial_playbook=grader.playbook.to_dict(),
            training_samples_count=len(train_samples)
        )

        start_epoch = 0
        print(f"▶️ Starting from epoch {start_epoch}\n")

    # Create epochs directory
    Path(EPOCHS_DIR).mkdir(parents=True, exist_ok=True)

    # Main adaptation loop
    print("="*70)
    print("ADAPTATION LOOP STARTING")
    print("="*70)
    print("💡 Press Ctrl+C to pause gracefully (progress will be saved)\n")

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{epochs-1}")
        print(f"{'='*70}")

        try:
            # Run one epoch
            result = adaptation_epoch(grader, train_samples, epoch)

            required_keys = ['epoch', 'avg_mismatch', 'playbook_size', 'deltas_applied', 'samples_processed', 'timestamp']

            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                raise ValueError(
                    f"adaptation_epoch() result missing keys: {missing_keys}"
                )

            # Save epoch file (unchanged from original)
            epoch_file = Path(EPOCHS_DIR) / f"epoch-{epoch:02d}.json"
            with open(epoch_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch': epoch,
                    'avg_mismatch': result['avg_mismatch'],
                    'playbook_size': len(grader.playbook.bullets()),
                    'deltas_applied': result['deltas_applied'],
                    'samples_processed': len(train_samples),
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_time_seconds': round(time.time() - epoch_start_time, 2)
                }, f, indent=2)

            # Update checkpoint
            checkpoint.update_epoch(
                epoch=epoch,
                avg_mismatch=result['avg_mismatch'],
                playbook_size=result['playbook_size'],
                deltas_applied=result['deltas_applied'],
                updated_playbook=grader.playbook.to_dict()
            )

            print(f"✅ Epoch {epoch} complete - Checkpoint saved")
            print(
                f"      Mismatch: {result['avg_mismatch']:.3f} | "
                f"      Playbook: {result['playbook_size']} bullets | "
                f"      Deltas: {result['deltas_applied']}"
            )
                
        except Exception as e:
            print(f"\n❌ ERROR during epoch {epoch}: {e}")
            print(f"    Error type: {type(e).__name__}")
            print(f"    Checkpoint saved at epoch {checkpoint.data.get('last_completed_epoch', -1)}")
            print("Progress saved. Fix the issue and rerun to resume.")
            raise

    # All epochs completed sucessfully.
    print("\n" + "="*70)
    print("🎉 ADAPTATION COMPLETE")
    print("="*70)


    # Generate final summary
    summary = {
        'total_epochs': epochs,
        'training_samples': len(train_samples),
        'total_deltas': checkpoint.data['total_deltas_applied'],
        'final_playbook_size': checkpoint.data['adaptation_history'][-1]['playbook_size'],
        'final_mismatch': checkpoint.data['adaptation_history'][-1]['avg_mismatch'],
        'started_at': checkpoint.data['started_at'],
        'completed_at': datetime.now().isoformat(),
        'resume_count': checkpoint.data['resume_count'],
        'epoch_history': checkpoint.data['adaptation_history'],
        'deltas_log': deltas_log
    }

    summary_path = Path(LOG_DIR) / "adaptation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Summary saved to: {summary_path}")

    # Save final playbook
    playbook_path = Path(LOG_DIR) / "latest_playbook.json"
    with open(playbook_path, 'w', encoding='utf-8') as f:
        json.dump(grader.playbook.to_dict(), f, indent=2)

    print(f"📚 Final playbook saved to: {playbook_path}")

    # Delete checkpoint (no longer needed)
    checkpoint.delete()

    print("\n✨ All done! You can now run post-adaptation testing.")


def main():
    from constants import ADAPTATION_SUMMARY_FILE, LATEST_PLAYBOOK_FILE

    """
    Execute full adaptation loop across all epochs without checkpointing.
    """

    print("="*70)
    print("🚀 AGENTIC CONTEXT ENGINEERING - STANDARD MODE")
    print("="*70)

    # Load samples
    train_samples = load_training_samples()

    # Initialize grader
    grader = UPVGrader(task_prompt=TASK_PROMPT)
    
    # Run epochs
    for epoch in range(EPOCHS):
        try:
            epoch_result = adaptation_epoch(grader, train_samples, epoch)
            print(f"✅ Epoch {epoch} completed successfully")
            print(epoch_result)
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
    with open(LOG_DIR / ADAPTATION_SUMMARY_FILE, 'w') as f:
        json.dump({
            'deltas_log': deltas_log,
            'total_epochs': EPOCHS,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    grader.playbook.save_to_file(str(LATEST_PLAYBOOK_FILE))

    print("="*70)
    print("✅ Adaptation Complete!")
    print(
        f"Average mismatch across all epochs: {mean(mismatches_log):.3f}"
    )
    print(f"Total deltas applied: {len(deltas_log)}")
    print(f"Final playbook size: {len(grader.playbook.bullets())} bullets")
    print("="*70)

if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    epochs = EPOCHS
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
            print(f"Using custom epoch count: {epochs}")
        except ValueError:
            print(f"Invalid epoch coumt. Using default: {EPOCHS}")
    
    if CHECKPOINT_ENABLED:
        adaptation_with_checkpoint(epochs)
    else:
        main()
