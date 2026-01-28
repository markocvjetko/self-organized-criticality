"""Utility functions for state hashing and I/O."""
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def compute_state_hash(state: torch.Tensor) -> bytes:
    """Compute MD5 hash of tensor state for cycle detection."""
    state_bytes = state.cpu().numpy().tobytes()
    return hashlib.md5(state_bytes).digest()


def save_results(
    results: List[dict],
    output_dir: Path,
    experiment_id: str,
    config: Dict[str, Any],
    final_state: Optional[torch.Tensor] = None,
) -> Path:
    """Save experiment results to JSON and optionally pickle the final state."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics for plot reconstruction
    steps_list = [r["steps"] for r in results]
    changes_list = [r["changes"] for r in results]
    unique_cells_list = [r["unique_cells"] for r in results]
    summary = {
        "steps_min": min(steps_list) if steps_list else 0,
        "steps_max": max(steps_list) if steps_list else 0,
        "steps_mean": sum(steps_list) / len(steps_list) if steps_list else 0,
        "changes_min": min(changes_list) if changes_list else 0,
        "changes_max": max(changes_list) if changes_list else 0,
        "changes_mean": sum(changes_list) / len(changes_list) if changes_list else 0,
        "unique_cells_min": min(unique_cells_list) if unique_cells_list else 0,
        "unique_cells_max": max(unique_cells_list) if unique_cells_list else 0,
        "unique_cells_mean": sum(unique_cells_list) / len(unique_cells_list) if unique_cells_list else 0,
    }

    # Save results as JSON (human-readable)
    json_path = output_dir / f"{experiment_id}_results.json"
    output_data = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "num_perturbations": len(results),
        "summary": summary,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Optionally save final state as pickle (for resuming)
    if final_state is not None:
        pkl_path = output_dir / f"{experiment_id}_final_state.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(final_state.cpu(), f)

    return json_path


def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate unique experiment ID based on timestamp."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
