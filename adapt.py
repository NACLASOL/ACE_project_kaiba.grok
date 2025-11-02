import re
import json
import datetime
from parse_data import train_samples
from statistics import mean
from upv_grader import *

EPOCHS = 1
mismatches_log = []
deltas_log = []
playbooks_log = []


def log_failure(final_answer):
    with open('logs/result_failures.log', 'w') as f:
        f.write(f"---\n{datetime.now()}\n\n{final_answer}\n---\n")
        
for epoch in range(EPOCHS):
    epoch_mismatches = []
    for sample in train_samples:

        try:
            # 1. GENERATE.
            prompt = sample['input'] + "\n**PLAYBOOK STRATEGIES:**\n" + str(playbook)
            result = generator.generate(question=prompt, context= "", playbook=playbook)
        except Exception as e:
            # Debug: Log if parse fails.
            log_failure(e)
            raise

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

        # 3. MISMATCH.
        human = sample['ground_truth']
        mismatch = mean(abs(ai_scores[k] - human[k]) for k in human)
        epoch_mismatches.append(mismatch)

        feedback = f"Human: {human}. AI: {ai_scores}. Avg mismatch: {mismatch:.2f}. Fix reasoning gaps."

        # 4. REFLECT.
        reflection = reflector.reflect(
            question="",
            generator_output=result,
            playbook=playbook,
            ground_truth=str(human),
            feedback=feedback,
        )
        
        # 5. CURATE & APPLY DELTA.
        deltas = curator.curate(
            reflection=reflection,
            playbook=playbook,
            question_context="",
            progress="",
        )
        old_size = len(playbook.bullets)
        playbook.apply_delta(deltas)
        new_deltas = [d for d in deltas if 'ADD' in str(d) or 'MODIFY' in str(d)] # Log changes.
        deltas_log.append({'epoch' : epoch, 'deltas' : new_deltas})

    avg_mismatch = mean(epoch_mismatches)
    mismatches_log.append(avg_mismatch)
    playbooks_log.append(playbook.bullets) # Snapshot of playbook.

    # LOG
    json.dump({'epoch' : epoch, 'avg_mismatch' : avg_mismatch, 'playbook_size' : len(playbook.bullets)},
            open(f'logs/epoch-{epoch:02d}.json', 'w'))

print("✅ Adaptation Complete!")