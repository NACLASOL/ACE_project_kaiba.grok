import re
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from opik import track
from parse_data import train_samples, test_samples, one_sample
from upv_grader import *

EPOCHS = 1  
mismatches_log = []
deltas_log = []
playbooks_log = []

Path('logs').mkdir(exist_ok=True) # Ensure directory.

def log_failure(final_answer, error_message=""):
    with open('logs/epoch_failure.log', 'w') as f:
        f.write(f"---\n{datetime.now()}\nError: {error_message}\n\n{final_answer}\n---\n")
      
for epoch in range(EPOCHS):
    epoch_mismatches = []
    for i, sample in enumerate(test_samples):

        try:
            # 1. GENERATE.
            prompt = sample['input'] + "\n**PLAYBOOK STRATEGIES:**\n" + str(playbook)
            result = generator.generate(question=prompt, context= "", playbook=playbook)
        except Exception as e:
            # Debug: Log if parse fails.
            log_failure(str(sample['input']+ "..."), str(e)) # Log prompt snippet.
            # Skip this sample or retry (for now, its sets low scores)
            ai_scores= {'TA': 1.0, 'CC': 1.0, 'GR': 1.0, 'LR': 1.0, 'OWP': 1.0}
            mismatch = 3.0 # High score for penalty.
            epoch_mismatches.append(mismatch)
            feedback = f"Generation failed: {str(e)}. Human: {sample['ground_truth']}. Use fallback scores."
            continue

        # 2. PARSE AI SCORES (Flexible regex for labels/spaces/colons).
        results_match = re.search(
            r'RESULTS\s*,\s*(?:TA\s*[\: ]*)?(\d\.\d)\s*,\s*(?:CC\s*[\: ]*)?(\d\.\d)\s*,\s*(?:GR\s*[\: ]*)?(\d\.\d)\s*,\s*(?:LR\s*[\: ]*)?(\d\.\d)\s*,\s*(?:OWP\s*[\: ]*)?(\d\.\d)',
            result.final_answer, re.IGNORECASE,
        )
        if results_match:
            ai_scores = {
                'TA' : float(results_match.group(1)),
                'CC' : float(results_match.group(2)),
                'GR' : float(results_match.group(3)),
                'LR' : float(results_match.group(4)),
                'OWP' : float(results_match.group(5)),
            }
        else:
            ai_scores = {'TA': 0.0, 'CC' : 0.0, 'GR' : 0.0, 'LR' : 0.0, 'OWP' : 0.0} # Fail-safe.
            log_failure(result.final_answer, "Regex parse failed")

        # 3. MISMATCH.
        human = sample['ground_truth']
        mismatch = mean(abs(ai_scores[k] - human[k]) for k in human)
        epoch_mismatches.append(mismatch)

        feedback = f"Human: {human}. AI: {ai_scores}. Avg mismatch: {mismatch:.2f}. Fix reasoning gaps."

        # 4. REFLECT.
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
            reflection = "" # Skip this reflection.
        
        # 5. CURATE & APPLY DELTA.
        try:
            deltas = curator.curate(
                reflection=reflection,
                playbook=playbook,
                question_context="",
                progress="",
            )
            playbook.apply_delta(deltas.delta)
            delta_json = deltas.delta.to_json()
            new_deltas = [str(d) for d in delta_json.get('operations', []) if 'ADD' in str(d) or 'TAG' in str(d)] # Log changes.
            deltas_log.append({'epoch' : epoch, 'deltas' : new_deltas})
        except Exception as e:
            log_failure(feedback, f"Curate failed: {str(e)}")
            raise

    avg_mismatch = mean(epoch_mismatches)
    mismatches_log.append(avg_mismatch)
    playbooks_log.append(playbook.bullets()) # Snapshot of playbook.

    # LOG
    epoch_log = {'epoch': epoch, 'avg_mismatch': avg_mismatch, 'playbook_size': len(playbook.bullets())}
    json.dump(epoch_log, open(f'logs/epoch-{epoch:02d}.json', 'w'), indent=2)

    print(f"Epoch {epoch}: Avg mismatch {avg_mismatch:.3f}, Playbook size {len(playbook.bullets())}")

json.dump({'mismatches_log': mismatches_log, 'deltas_log': deltas_log}, open('logs/adaptation_summary.json', 'w'), indent=2)

playbook.save_to_file("logs/latest_playbook.json")


print("✅ Adaptation Complete!")