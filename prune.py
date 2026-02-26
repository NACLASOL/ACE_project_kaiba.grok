"""
Playbook Pruning Modeul for the ACE Framework

Provides multiple strategies for reducing playbook size while maintaining quality:
- Similarity-based: Remove semantically duplicate bullets
- Usage-based: Remove bullets never referenced in bullet_ids
- Performance-based: Remove bullets not correlated with score improvements
- Age-based: Prioritize recent, effective bullets
"""

import json
import hashlib
import numpy as np

from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from opik import track
from opik.opik_context import update_current_span
from ace import Playbook
from ace.delta import DeltaBatch, DeltaOperation

from constants import (
    LOG_DIR,
    OPIK_PROJECT_NAME,
    PRUNING_ENABLED,
    PRUNING_STRATEGY,
    PRUNING_SIMILARITY_THRESHOLD,
    PRUNING_MIN_USAGE_COUNT,
    PRUNING_AGE_THRESHOLD_EPOCHS,
    PRUNING_MIN_PLAYBOOK_SIZE,
    PRUNING_MAX_REMOVE_PER_EPOCH,
    PRUNING_DRY_RUN,
    PRUNING_LOG_FILE
)

# === DATA STRUCURES ===
@dataclass
class BulletMetadata:
    """Metadata for tracking bullet usage and performance."""
    bullet_id: str
    content: str
    created_epoch: int
    last_used_epoch: int = -1
    usage_count: int = 0
    total_references: int = 0
    avg_score_when_used: float = 0.0
    avg_score_when_not_used: float = 0.0
    section: str = "essential"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
@dataclass
class PruningResult:
    """Result of a pruning operation."""
    epoch: int
    strategy: str
    bullets_before: int
    bullets_after: int
    bullets_removed: int
    removed_bullet_ids: List[str] = field(default_factory=list)
    removed_bullets_content: List[Dict[str,str]] = field(default_factory=list)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    usage_stats: Dict[str, int] = field(default_factory=dict)
    dry_run: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

# === PRUNING STRATEGIES ===

class PlaybookPruner:
    """
    Manages playbook pruning with multiple strategies.

    Usage:
        pruner = PlaybookPruner(strategy='similarity')
        result = pruner.prune(playbook, epoch=5, usage_tracker=tracker)
    """

    def __init__(
            self,
            strategy: str = PRUNING_STRATEGY,
            similarity_threshold: float = PRUNING_SIMILARITY_THRESHOLD,
            min_usage_count: int = PRUNING_MIN_USAGE_COUNT,
            age_threshold: int = PRUNING_AGE_THRESHOLD_EPOCHS,
            min_playbook_size: int = PRUNING_MIN_PLAYBOOK_SIZE,
            max_remove_per_epoch: int = PRUNING_MAX_REMOVE_PER_EPOCH,
            dry_run: bool = PRUNING_DRY_RUN,
            embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Args:
            strategy: Pruning strategy ('similarity', 'usage', 'performance', 'age', 'hybrid')
            similarity_threshold: Cosine similarity threshold for duplicate detection (0.0-1.0)
            min_usage_count: Minimum times a bullet must be used to avoid pruning.
            age_threshold: Remove bullets unused for this many epochs
            min_playbook_size: Never prune below this size (safety)
            max_remove_per_epoch: Maximum bullets to remove per pruning operation
            dry_run: determines whether to log decisions but doesn't modify the playbook
            embedding_model: SentenceTransformer model for semantic similarity
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.min_usage_count = min_usage_count
        self.age_threshold = age_threshold
        self.min_playbook_size = min_playbook_size
        self.max_remove_per_epoch = max_remove_per_epoch
        self.dry_run = dry_run

        # Initialize embedding model for similarity-based pruning.
        if strategy in ['similarity', 'hybrid']:
            print(f"📦 Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

        # Metadata tracking
        self.bullet_metadata: Dict[str, BulletMetadata] = {}

        print(f"✅ PlaybookPruner initialized (strategy={strategy}, dry_run={dry_run})")

    @track(project_name=OPIK_PROJECT_NAME, metadata={"component": "pruning"})
    def prune(
        self,
        playbook: Playbook,
        epoch: int,
        usage_tracker: Optional['UsageTracker'] = None
    ) -> PruningResult:
        """
        Prune the playbook using the configured strategy.

        Args:
        playbook: The ACE Playbook to prune
        epoch: Current adaptation epoch
        usage_tracker: Optional traacker with bullet usage statistics

        Returns:
            PruningResult with details of pruning operation
        """
        bullets_before = len(playbook.bullets())

        # Safety check: don't prune if below minimum size
        if bullets_before <= self.min_playbook_size:
            print(f"⏭️ Skipping pruning: playbook size ({bullets_before}) <= minimum ({self.min_playbook_size})")
            return PruningResult(
                epoch=epoch,
                strategy=self.strategy,
                bullets_before=bullets_before,
                bullets_after=bullets_before,
                bullets_removed=0,
                dry_run=self.dry_run
            )
        
        # Update metadata from usage tracker
        if usage_tracker:
            self._update_metadata_from_tracker(usage_tracker, epoch)
        
        # Select pruning strategy
        if self.strategy == 'similarity':
            bullets_to_remove = self._prune_by_similarity(playbook)
        elif self.strategy == 'usage':
            bullets_to_remove = self._prune_by_usage(playbook, epoch)
        elif self.strategy == 'performance':
            bullets_to_remove = self._prune_by_usage(playbook, epoch)
        elif self.strategy == 'age':
            bullets_to_remove = self._prune_by_age(playbook, epoch)
        elif self.strategy == 'hybrid':
            bullets_to_remove = self._prune_hybrid(playbook, epoch)
        else:
            raise ValueError(f"Unknown pruning strategy: {self.strategy}")
        
        # Limit removal count
        if len(bullets_to_remove) > self.max_remove_per_epoch:
            print(f"⚠️ Limiting removal: {len(bullets_to_remove)} → {self.max_remove_per_epoch} bullets")
            bullets_to_remove = bullets_to_remove[:self.max_remove_per_epoch]

        # Build pruning result
        result = PruningResult(
            epoch=epoch,
            strategy=self.strategy,
            bullets_before=bullets_before,
            bullets_after=bullets_before - len(bullets_to_remove),
            bullets_removed=len(bullets_to_remove),
            removed_bullet_ids=[b.id for b in bullets_to_remove],
            removed_bullets_content=[
                {"id": b.id, "content": b.content, "section": b.section} for b in bullets_to_remove
            ],
            dry_run=self.dry_run
        )

        # Apply remove (or just log for dry run)
        if not self.dry_run and bullets_to_remove:
            self._apply_removal(playbook, bullets_to_remove)
            print(f"✂️ Pruned {len(bullets_to_remove)} bullets from playbook")
        elif self.dry_run and bullets_to_remove:
            print(f"🔍 DRY RUN: Would remove {len(bullets_to_remove)} bullets")
        else:
            print(f"✅ No bullets to prune this epoch")

        # Log to Opik
        update_current_span(
            metadata={
                "bullets_removed": result.bullets_removed,
                "bullets_before": bullets_before,
                "bullets_after": result.bullets_after,
                "strategy": self.strategy,
                "dry_run": self.dry_run
            }
        )

        return result
    
    def _prune_by_similarity(self, playbook: Playbook) -> List:
        """
        Remove bullets that are semantically similar (duplicates).

        Three-pass strategy:
        Pass 0 - Empty/None content: Remove bullets with no meaningul content. These are unreliable for embedding and useless in the playbook.
        Pass 1 - Exact duplicate detection: Content-hash comparison before any embedding is computed. Efficient and 100% reliable.
        Pass 2 - Semantic similarity: Cosine similarity on remaining non-empty, non-duplicate bullets. Include zero-norm and NaN protection.
        """
        bullets = playbook.bullets()
        if len(bullets) <= 1:
            return []
        
        to_remove: set = set()

        # === PASS 0: Empty / None / whitespace-only content ===
        for bullet in bullets:
            content = bullet.content
            if content is None or not str(content).strip():
                to_remove.add(bullet.id)
                print(f"    🗑️ Empty content bullet flagged: "
                      f"id='{bullet.id}' section='{bullet.section}'")
        
        # === PASS 1: Exact duplicate detection via content hash ===
        seen_hashes: Dict[str,str] = {} #md5_hash -> Frst bullet ID
        for bullet in bullets:
            if bullet.id in to_remove:
                continue
            normalised = str(bullet.content or "").strip().lower()
            content_hash = hashlib.md5(normalised.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                to_remove.add(bullet.id)
                print(f"    🔁 Exact duplicate flagged: "
                      f"id='{bullet.id}'"
                      f"(duplicate of '{seen_hashes[content_hash]}')")
            else:
                seen_hashes[content_hash] = bullet.id

        # === PASS 2: Embedding-based semantic similarity ===
        encodable_bullets = [
            b for b in bullets
            if b.id not in to_remove
            and b.content
            and str(b.content).strip()
        ]

        if len(encodable_bullets) > 1:
            contents = [str(b.content).strip() for b in encodable_bullets]

            print(f"🧮 Computing embeddings for {len(contents)} bullets...")
            try:
                embeddings = self.embedding_model.encode(
                    contents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                norms = np.linalg.norm(embeddings, axis=1)
                zero_norm_mask = (norms == 0.0)
                if zero_norm_mask.any():
                    zero_count = zero_norm_mask.sum()
                    print(f"    ⚠️ {zero_count} bullet(s) produced zero-norm "
                          f"embeddings - flagging directly.")
                    for idx, is_zero in enumerate(zero_norm_mask):
                        if is_zero:
                            to_remove.add(encodable_bullets[idx].id)
                            print(f"    🗑️ Zero-norm bullet flagged: "
                                  f"'{encodable_bullets[idx].id}'")
                    # Refilter to valid (non-zero-norm) bullets only
                    valid_idx = [i for i, z in enumerate(zero_norm_mask) if not z]
                    encodable_bullets = [encodable_bullets[i] for i in valid_idx]
                    embeddings = embeddings[valid_idx]
                
                if len(encodable_bullets) > 1:
                    similarity_matrix = cosine_similarity(embeddings)

                    # NaN gaurd: defensive protection against any residual
                    similarity_matrix = np.nan_to_num(
                        similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0
                    )

                    for i in range(len(encodable_bullets)):
                        if encodable_bullets[i].id in to_remove:
                            continue
                        for j in range(i + 1, len(encodable_bullets)):
                            if encodable_bullets[j].id in to_remove:
                                continue
                            similarity = float(similarity_matrix[i][j])
                            if similarity >= self.similarity_threshold:
                                to_remove.add(encodable_bullets[j].id)
                                print(
                                    f"  📊 Semantic duplicate"
                                    f"(similarity={similarity:.3f}): "
                                    f"Keeping   '{str(encodable_bullets[i].content)[:50]}'"
                                    f"Removing  '{str(encodable_bullets[j].content)[:50]}'"
                                )
            except Exception as e:
                print(f"    ⚠️ Embedding similarity pass failed: {type(e).__name__}: {e}")
                print(f"    Pass 0 and Pass 1 results are still applied.")

        return [b for b in bullets if b.id in to_remove]        
        
    def _prune_by_usage(self, playbook: Playbook, epoch: int) -> List:
        """
        Remove bullets that are rarely or never used.
        
        Strategy:
        1. Track how often each bullet is referenced in bullet_ids
        2. Remove bullets with usage_count < min_usage_count
        3. Protect seed bullets (created at epoch 0)
        """
        bullets = playbook.bullets()
        to_remove = []

        for bullet in bullets:
            metadata = self.bullet_metadata.get(bullet.id)

            # Protect seed bullets
            if metadata and metadata.created_epoch == 0:
                continue

            # Check usage
            usage_count = metadata.usage_count if metadata else 0
            if usage_count < self.min_usage_count:
                to_remove.append(bullet)
                print(f"    📉 Low usage: '{bullet.content[:50]}...' (used {usage_count} times)")

            
        return to_remove
    
    def _prune_by_performance(self, playbook: Playbook, epoch: int) -> List:
        """
        Remove bullets that don't correlate with score improvements.
        
        Strategy:
        1. Track avg scores when bullet is used vs. not used
        2. Remove bullets where avg_score_when_used <= avg_score_when_not_used
        3. Requires sufficient data (min 5 samples)
        """
        bullets = playbook.bullets()
        to_remove = []

        for bullet in bullets:
            metadata = self.bullet_metadata.get(bullet.id)

            # Skip if insufficient data
            if not metadata or metadata.usage_count < 5:
                continue

            # Check performance impact
            if metadata.avg_score_when_used <= metadata.avg_score_when_not_used:
                to_remove.append(bullet)
                improvement = metadata.avg_score_when_used - metadata.avg_score_when_not_used
                print(f"    📊 No improvement: '{bullet.content[:50]}...'"
                      f"(Δscore={improvement:.3f}")
                
        return to_remove
    
    def _prune_by_age(self, playbook: Playbook, epoch: int) -> List:
        """
        Remove old bullets that haven't been used recently.

        Strategy:
        1. Remove bullets where (epoch - last_used_epoch) > age_threshold
        2. Protect seed bullets and recently added bullets
        """
        bullets = playbook.bullets()
        to_remove = []

        for bullet in bullets:
            metadata = self.bullet_metadata.get(bullet.id)

            if not metadata:
                continue

            # Protect seed bullets
            if metadata.created_epoch == 0:
                continue

            # Check staleness
            epochs_since_use = epoch - metadata.last_used_epoch
            if epochs_since_use > self.age_threshold and metadata.last_used_epoch != -1:
                to_remove.append(bullet)
                print(f"    ⏰ Stale bullet: '{bullet.content[:50]}...'"
                      f"(unused for {epochs_since_use} epochs)")
                
        return to_remove
    
    def _prune_hybrid(self, playbook: Playbook, epoch: int) -> List:
        """
        Combine multiple strategies for robust pruning.
        
        Strategy:
        1. First pass: Remove exact duplicates (similarity > 0.95)
        2. Second pass: Remove low-usage bullets (usage < 2)
        3. Third pass: Remove stale bullets (unused for 3+ epochs)
        """
        to_remove_ids = set()
        bullets = playbook.bullets()

        # Pass 1: high similarity (duplicates)
        if self.embedding_model:
            high_sim_bullets = self._prune_by_similarity_threshold(playbook, threshold=0.60)
            to_remove_ids.update(b.id for b in high_sim_bullets)
            print(f"    🔍 Hybrid Pass 1 (similarity): {len(high_sim_bullets)} candidates")

        # Pass 2: Very low usage
        low_usage_bullets = [
            b for b in bullets
            if b.id not in to_remove_ids
            and self.bullet_metadata.get(b.id)
            and self.bullet_metadata[b.id].usage_count < 2
            and self.bullet_metadata[b.id].created_epoch > 0 # Protect seed bullets
        ]
        to_remove_ids.update(b.id for b in low_usage_bullets)
        print(f"    🔍 Hybrid Pass 2 (usage): {len(low_usage_bullets)} candidates")

        # Pass 3: Staleness
        stale_bullets = [
            b for b in bullets
            if b.id not in to_remove_ids
            and self.bullet_metadata.get(b.id)
            and (epoch - self.bullet_metadata[b.id].last_used_epoch) > 3
            and self.bullet_metadata[b.id].last_used_epoch != -1
        ]
        to_remove_ids.update(b.id for b in stale_bullets)
        print(f"    🔍 Hybrid Pass 3 (staleness): {len(stale_bullets)} candidates")

        return [b for b in bullets if b.id in to_remove_ids]
    
    def _prune_by_similarity_threshold(self, playbook: Playbook, threshold: float) -> List:
        """Helper for similarity pruning with custom threshold."""
        original_threshold = self.similarity_threshold
        self.similarity_threshold = threshold
        result = self._prune_by_similarity(playbook)
        self.similarity_threshold = original_threshold
        return result
    
    def _apply_removal(self, playbook: Playbook, bullets_to_remove: List) -> None:
        """Apply bullet removal to playbook using Delta operations."""
        if not bullets_to_remove:
            return
        
        # Build one REMOVE operation per bullet
        operations = []
        for bullet in bullets_to_remove:
            op = DeltaOperation(
                type='REMOVE',
                section=bullet.section,
                bullet_id=bullet.id,
                content=None,
                metadata={}
            )
            operations.append(op)

        batch = DeltaBatch(
            reasoning=f"Pruning {len(operations)} reduntant/unused bullet(s) from playbook.",
            operations=operations
        )

        # Apply REMOVE operation through standard Playbook interface
        playbook.apply_delta(batch)

    def _update_metadata_from_tracker(self, tracker: 'UsageTracker', epoch: int) -> None:
        """Update bullet metadata from usage tracker."""
        for bullet_id, stats in tracker.get_all_stats().items():
            if bullet_id not in self.bullet_metadata:
                self.bullet_metadata[bullet_id] = BulletMetadata(
                    bullet_id=bullet_id,
                    content=stats.get('content', ''),
                    created_epoch=stats.get('created_epoch', epoch),
                    section=stats.get('section', 'essential')
                )
            
            metadata = self.bullet_metadata[bullet_id]
            metadata.usage_count = stats.get('usage_count', 0)
            metadata.total_references = stats.get('total_refereces', 0)
            metadata.last_used_epoch = stats.get('last_used_epoch', -1)
            metadata.avg_score_when_used = stats.get('avg_score_when_used', 0.0)
            metadata.avg_score_when_not_used = stats.get('avg_score_when_not_used', 0.0)
        
    def save_metadata(self, path: Path) -> None:
        """Save bullet metadata to JSON file."""
        data = {
            bullet_id: metadata.to_dict()
            for bullet_id, metadata in self.bullet_metadata.items()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_metadata(self, path: Path) -> None:
        """Load bullet metadata from JSON file."""
        if not path.exists():
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bullet_metadata = {
            bullet_id: BulletMetadata(**metadata_dict)
            for bullet_id, metadata_dict in data.items()
        }

# === USAGE TRACKING ===
class UsageTracker:
    """
    Tracks bullet usage across adaptation loop.

    Integrates with adapt.py to monitor which bullets are referenced in bullet_ids during generation.
    """

    def __init__(self):
        self.usage_log: List[Dict[str, Any]] = []
        self.bullet_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'usage_count': 0,
            'total_references': 0,
            'last_used_epoch': -1,
            'scores_when_used': [],
            'scores_when_not_used': [],
            'created_epoch': -1,
            'content': '',
            'section': 'essential'
        })
    
    def record_usage(
            self,
            epoch: int,
            sample_idx: int,
            bullet_ids_used: List[str],
            all_bullet_ids: List[str],
            score: float,
            playbook: Playbook
    ) -> None:
        """
        Record bullet usage for a sample.
        
        Args:
            epoch: Current epoch
            sample_idx: Sample index
            bullet_ids_used: Bullets referenced in this sample
            all_bullet_ids: All available bullet IDs
            score: Average score for this sample
            playbook: Current playbook state
        """
        # Log usage event
        self.usage_log.append({
            'epoch': epoch,
            'sample': sample_idx,
            'bullet_ids_used': bullet_ids_used,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })

        # Update stats for used bullets
        for bullet_id in bullet_ids_used:
            stats = self.bullet_stats[bullet_id]
            stats['usage_count'] += 1
            stats['total_references'] += 1
            stats['last_used_epoch'] = epoch
            stats['scores_when_used'].append(score)

            # Update content if not set
            if not stats['content']:
                bullet = playbook.get_bullet(bullet_id)
                if bullet:
                    stats['content'] = bullet.content
                    stats['section'] = bullet.section
                    stats['created_epoch'] = epoch # Approximate
        
        for bullet_id in all_bullet_ids:
            if bullet_id not in bullet_ids_used:
                self.bullet_stats[bullet_id]['scores_when_not_used'].append(score)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get computed statistics for all bullets."""
        stats = {}
        for bullet_id, raw_stats in self.bullet_stats.items():
            # Compute averages
            avg_when_used = (
                np.mean(raw_stats['scores_when_used'])
                if raw_stats['scores_when_used'] else 0.0
            )

            avg_when_not_used = (
                np.mean(raw_stats['scores_when_not_used'])
                if raw_stats['scores_when_not_used'] else 0.0
            )

            stats[bullet_id] = {
                'usage_count': raw_stats['usage_count'],
                'total_references': raw_stats['total_references'],
                'last_used_epoch': raw_stats['last_used_epoch'],
                'avg_score_when_used': float(avg_when_used),
                'avg_score_when_not_used': float(avg_when_not_used),
                'created_epoch': raw_stats['created_epoch'],
                'content': raw_stats['content'],
                'section': raw_stats['section']
            }

        return stats
    
    def save_log(self, path: Path) -> None:
        """Save usage log to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.usage_log, f, indent=2, ensure_ascii=False)
    
    def save_stats(self, path: Path) -> None:
        """Save computed statstics to file."""
        stats = self.get_all_stats()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

# === LOGGING ===
 
class PruningLogger:
    """Manages pruning operation logging."""
    
    def __init__(self, log_file: Path = PRUNING_LOG_FILE):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"pruning_operations": []}, f)

    def log_pruning(self, result: PruningResult) -> None:
        """Append pruning result to log file."""
        # Read existing log
        with open(self.log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        # Append new result
        log_data['pruning_operations'].append(result.to_dict())

        # Write back
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"📝 Logged pruning operation to {self.log_file}")

# === MAIN FUNCTION ===

def prune_playbook(
        playbook: Playbook,
        epoch: int,
        usage_tracker: Optional[UsageTracker] = None,
        strategy: str = PRUNING_STRATEGY
) -> PruningResult:
    """
    Function for pruning a playbook.
    
    Args:
        playbook: Playbook to prune
        epoch: Current epoch
        usage_tracker: Optional usage tracker
        strategy: Pruning strategy

    Returns:
        PruningResult
    """
    pruner = PlaybookPruner(strategy=strategy)
    result = pruner.prune(playbook, epoch, usage_tracker)

    # Log result
    logger = PruningLogger()
    logger.log_pruning(result)

    return result

if __name__ == "__main__":
    print("⚠️   This module is not meant to be run directly.")
    print("     Import and use PlaybookPruner or prune_playbook() in adapt.py")