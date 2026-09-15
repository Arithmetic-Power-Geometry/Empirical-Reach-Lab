from __future__ import annotations

import json
from pathlib import Path

from empirical_reach.core import empirical_partition, reach_gain, unresolved_diameter
from empirical_reach.oed import best_experiment_by_model_ig, model_information_gain
from empirical_reach.worlds import make_hidden_bit_world


def main() -> None:
    worlds, current, candidate, outcome = make_hidden_bit_world(noise=0.1)
    model_labels = [0, 0, 1, 1]

    best, best_score = best_experiment_by_model_ig(current, model_labels)
    summary = {
        "worlds": worlds,
        "current_partition": empirical_partition(current, len(worlds)),
        "current_unresolved_diameter": unresolved_diameter(current, outcome),
        "candidate_reach_gain": reach_gain(current, candidate, outcome),
        "expanded_unresolved_diameter": unresolved_diameter([*current, candidate], outcome),
        "expanded_partition": empirical_partition([*current, candidate], len(worlds)),
        "oed_best_experiment": best.name,
        "oed_best_model_information_gain_bits": best_score,
        "oed_best_unresolved_diameter": unresolved_diameter([best], outcome),
        "candidate_model_information_gain_bits": model_information_gain(candidate, model_labels),
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
