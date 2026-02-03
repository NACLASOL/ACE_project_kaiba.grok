import json
import signal
import sys
import platform
import os

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

from constants import EPOCHS_DIR, LOG_DIR


class AdaptationCheckpoint:
    """Manages checkpoint for adaptation loop with resume capability."""

    def __init__(self, checkpoint_path: str = "logs/adaptation_checkpoint.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.data: Optional[Dict] = None
        self.interrupted = False
        self._save_in_progress = False
        self._signal_count = 0 # Track multiple interrupts

        # Register signal handlers (skip on windows)
        if platform.system() != "Windows":
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)
            print("✅ Signal handlers registered (Ctrl+C will save progress)")


    def _handle_interrupt(self, signum, frame):
        """
        Handle interrupt signals WITHOUT calling sys.exit().

        Strategy:
        1. First interrupt: Save checkpoint, raise KeyboardInterrupt
        2. Second interrupt: Force immediate exit (user impatience)
        """
        self._signal_count += 1

        # User presed Ctrl+C twice -> force exit
        if self._signal_count > 1:
            print("\n⚠️ SECOND INTERRUPT - Forcing immediate exit")
            print("⚠️ Progress may be lost!")
            sys.exit(1) # Only on second interrupt.

        print("\n" + "="*70)
        print("⚠️ INTERRUPT SIGNAL RECIEVED (Ctrl+C)")
        print("="*70)

        # Prevent re-entrance if save is in progress
        if self._save_in_progress:
            print("⚠️ Checkpoint save in progress - please wait...")
            print("    (Press Ctrl+C again to force exit)")
            return # Let save complete
        
        print("💾 Saving current progress...")
        self.interrupted = True

        if self.data:
            self.data['interrupted'] = True
            self.data['interrupted_at'] = datetime.now().isoformat()
            self.data['interrupted_at_epoch'] = self.data.get('last_completed_epoch', -1)

            try:
                self.save()
                print(f"✅ Progress saved to: {self.checkpoint_path}")
                print(f"📊 Completed epochs: {self.data['last_completed_epoch'] + 1}")
                print(f"📈 Total deltas: {self.data.get('total_deltas_applied', 0)}")
                print(f"⏱️ Interrupted at: {self.data['interrupted_at']}")
            except Exception as e:
                print(f"❌ Checkpoint save failed: {type(e).__name__}: {str(e)}")
                print(f"⚠️ Progress at epoch {self.data.get('last_completed_epoch' -1)} may be lost")
        
        else:
            print("⚠️ No checkpoint data to save (adaptation hasn't started yet)")
        
        print("💡 To resume: Run adapt.py again")
        print("     Checkpoint will be auto-detected and resumed")
        print("="*70)

        # ✅ CORRECT: Raise KeyboardInterrupt, DON'T call sys.exit()
        # This propagates to caller, allowing finally blocks to execute

        raise KeyboardInterrupt("User interrupted adaptation")
    
    def save(self):
        """
        Save checkpoint with atomic write and proper fsync.

        Protections:
        - Writes to temp file first
        - Uses os.fsync() to ensure data hist disk
        - Atomic rename
        - Sets flag to prevent interrupt during save
        """
        self._save_in_progress = True

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.checkpoint_path.with_suffix('.tmp')

            # Write to temp file with explicit flush
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                f.flush() # Flush Python buffer
                os.fsync(f.fileno()) # Force OS to write to disk NOW

            # Atomic rename
            os.replace(str(temp_path), str(self.checkpoint_path))
        
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise # Re-raise original exception

        finally:
            self._save_in_progress = False # Always reset flag

    def load(self) -> Optional[Dict]:
        """Load checkpoint with validation"""
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                required_keys = ['checkpoint_version', 'last_completed_epoch', 'current_playbook', 'adaptation_history']
                if not all(k in data for k in required_keys):
                    print(f"⚠️ Checkpoint missing required keys: {required_keys}")
                    return None
                
                self.data = data

                print(f"📂 Load checkpoint from: {self.checkpoint_path}")
                return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Checkpoint corrupted (invalid JSON): {e}")
            print("     Starting from scratch")
            return None
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()
        
    def initialize(self, total_epochs: int, initial_playbook: Dict, training_samples_count: int) -> Dict:
        """
        Initialize a new checkpoint for a fresh run.

        Args:
            total_epochs: Total number of epochs to run
            initial_playbook: Starting playbook state
            training_samples_count: Number of training samples

        Returns:
            Initialized checkpoint data
        """
        self.data = {
            'checkpoint_version': '1.0',
            'last_completed_epoch': -1, # No epochs completed yet.
            'total_epochs': total_epochs,
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'training_samples_count': training_samples_count,
            'current_playbook': initial_playbook,
            'adaptation_history': [],
            'total_deltas_applied': 0,
            'interrupted': False,
            'resume_count': 0
        }

        self.save()
        return self.data
    
    def save(self):
        """Save current checkpoint state to disk with atomic write."""
        if self.data is None:
            raise ValueError("Cannot save: checkpoint data not initialized")
        
        self.data['last_updated'] = datetime.now().isoformat()

        # Atomic write: write to temp file, then rename
        temp_path = self.checkpoint_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)

            # Atomic rename (overwrites existing checkpoint)
            temp_path.replace(self.checkpoint_path)

        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def update_epoch(self, epoch: int, avg_mismatch: float, playbook_size: int, deltas_applied: int, updated_playbook: Dict):
        """
        Record completion of an epoch.

        Args:
            epoch: Epoch number just completed
            avg_mismatch: Average mismatch for this epoch
            playbook_size: Current playbook size
            deltas_applied: Number of deltas applied this epoch
            updated_playbook: Current playbook state as dict
        """
        if self.data is None:
            raise ValueError("Checkpoint not initialized")
        
        self.data['last_completed_epoch'] = epoch
        self.data['current_playbook'] = updated_playbook
        self.data['total_deltas_applied'] += deltas_applied

        # Append to history
        self.data['adaptation_history'].append({
            'epoch': epoch,
            'avg_mismatch': round(avg_mismatch, 4),
            'playbook_size': playbook_size,
            'deltas_applied': deltas_applied,
            'timestamp': datetime.now().isoformat()
        })

        self.save()

    def should_resume(self) -> bool:
        """Check if we should resume from checkpoint."""
        if self.data is None:
            return False
        return  self.data['last_completed_epoch'] < self.data['total_epochs'] - 1
    
    def get_resume_epoch(self) -> int:
        """Get the next epoch to run (last completed + 1)."""
        if self.data is None:
            return 0
        return self.data['last_completed_epoch'] + 1
    
    def is_complete(self) -> bool:
        """Check if all epochs have been completed."""
        if self.data is None:
            return False
        return self.data['last_completed_epoch'] >= self.data['total_epochs'] - 1
    
    def delete(self):
        """Remove checkpoint file (used after successful completion)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"🗑️ Checkpoint deleted: {self.checkpoint_path}")
    
    def print_status(self):
        """Print human-readable checkpoint status."""
        if self.data is None:
            print("No checkpoint data loaded")
            return
        
        completed = self.data['last_completed_epoch'] + 1
        total = self.data['total_epochs']
        percent = (completed / total * 100) if total > 0 else 0

        print("\n" + "="*70)
        print("📋 CHECKPOINT STATUS")
        print("="*70)
        print(f"Progress: {completed}/{total} epochs ({percent:.1f}%)")
        print(f"Started: {self.data['started_at']}")
        print(f"Last Updated: {self.data['last_updated']}")
        print(f"Training Samples: {self.data['training_samples_count']}")
        print(f"Total Deltas Applied: {self.data['total_deltas_applied']}")
        print(f"Resume Count: {self.data['resume_count']}")

        if self.data.get('interrupted', False):
            print(f"⚠️ Last run was INTERRUPTED at: {self.data.get('interrupted_at', 'unknown')}")

        if completed > 0:
            last_epoch = self.data['adaptation_history'][-1]
            print(f"\nLast Completed Epoch ({last_epoch['epoch']}):")
            print(f"    - Avg Mismatch: {last_epoch['avg_mismatch']}")
            print(f"    - Playbook Size: {last_epoch['playbook_size']}")
            print(f"    - Deltas Applied: {last_epoch['deltas_applied']}")

        print("="*70 + "\n")
