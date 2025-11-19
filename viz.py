import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# === 1. CONFIGURATION ===
LOG_DIR = Path("logs")
SUMMARY_FILE = LOG_DIR / "adaptation_summary.json"
EPOCH_FILES = sorted(LOG_DIR.glob("epoch-*.json")) # For epoch-00.json, epoch-01.json, ...

# Plot style.
sns.set_style('whitegrid')
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["font.size"] = 12
SAVE_FIGS = False    # Set to False to only show interactively.
FIG_FMT = "png"      # "png" | "pdf" | "svg"

# 2. === LOAD DATA ===
def load_summary() -> Dict[str, Any]:
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_epoch(epoch: int) -> Dict[str, Any]:
    file = LOG_DIR / f"epoch-{epoch:02d}.json"
    if not file.exists():
        raise FileNotFoundError(file)
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def gather_all_epochs() -> List[Dict[str, Any]]:
    """Return a list ordered by epoch number."""
    epochs = []
    for fp in EPOCH_FILES:
        # The filename format: epoch-03.json -> epoch = 3.
        epoch_num = int(fp.stem.split("-")[1])
        epochs.append((epoch_num, load_epoch(epoch_num)))
    epochs.sort(key=lambda x: x[0])
    return [data for num, data in epochs]

# 3. === PLOTTING FUNCTIONS ===
def plot_mismatch_and_playbook(epochs_data: List[Dict[str, Any]]):
    epochs = [d["epoch"] for d in epochs_data]
    mismatches = [d["avg_mismatch"] for d in epochs_data]
    playbook_sizes = [d["playbook_size"] for d in epochs_data]

    fig, ax1 = plt.subplots()

    color = sns.color_palette("husl", 2)[0]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Mismatch", color=color)
    ax1.plot(epochs, mismatches, marker="o", color=color, label="Avg Mismatch")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(mismatches) *1.15)

    ax2 = ax1.twinx()
    color = sns.color_palette("husl", 2)[1]
    ax2.set_ylabel("Playboo Size", color=color)
    ax2.bar(epochs, playbook_sizes, alpha=0.6, color=color, label="Playbook Size")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Adaptation Progress - Mistmatch & Playbook Growth")
    fig.tight_layout()
    if SAVE_FIGS:
        plt.savefig(LOG_DIR / f"mismatch_playbook.{FIG_FMT}", dpi=300)
        plt.show()

def plot_deltas(summary: Dict[str, Any]):
    """Stacked bar of ADD / MODIFY / DELETE per epoch"""
    delta_per_epoch: List[Dict[str, int]] = []
    for entry in summary.get("deltas_log", []):
        epoch = entry["epoch"]
        ops = Counter()
        for op in entry["deltas"]:
            # "ops" look like: "ADD(bullet-123)" or "MODIFY(bullet-456)"
            if "ADD" in op:
                ops["ADD"] += 1
            elif "MODIFY" in op:
                ops["MODIFY"] += 1
            elif "DELETE" in op:
                ops["DELETE"]
        delta_per_epoch.append({"epoch": epoch, **ops})

    if not delta_per_epoch:
        print("Warning: No deltas found in adaptation_summary.json")
        return
            
    # Build the DataFrame for seaborn
    df = pd.DataFrame(delta_per_epoch).fillna(0)
    df = df.sort_values("epoch")

    ax = df.plot(
        x="epoch",
        kind="bar",
        stacked=True,
        color={"ADD": "#60cdaa", "MODIFY": "#fc8d62", "DELETE": "#8da0cb"},
        width=0.7,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Delta Operations")
    ax.set_title("Playbook Delta Operations per Epoch")
    plt.legend(title="Operation")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(LOG_DIR / f"deltas.{FIG_FMT}", dpi=300)
    plt.show

def plot_playbook_evolution(epochs_data: List[Dict[str, Any]]):
    """Simple line showing the playbook size growth"""
    epochs = [d["epoch"] for d in epochs_data]
    sizes = [d["playbook_size"] for d in epochs_data]

    plt.plot(epochs, sizes, marker="s", color="#1f78b4", linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Playbook Size (bullets)")
    plt.title("Playbook Growth Over Adaption")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(LOG_DIR / f"playbook_growth.{FIG_FMT}", dpi=300)
    plt.show()

# 4. === MAIN EXECUTION ===
def main():
    if not LOG_DIR.exists():
        raise FileExistsError(f"Log directory {LOG_DIR} not found - run adapt.py first")

    summary = load_summary()
    epoch_data = gather_all_epochs()

    if not epoch_data:
        print("Warning: No epoch-*.json files found - nothing to visualise")
        return
    
    print(f"Found {len(epoch_data)} epoch(s) and {len(summary.get('deltas_log', []))} delta entries")

    plot_mismatch_and_playbook(epoch_data)
    plot_playbook_evolution(epoch_data)
    plot_deltas(summary)

    print(f"Plots saved to {LOG_DIR} (format: {FIG_FMT})")

if __name__ == "__main__":
    main()