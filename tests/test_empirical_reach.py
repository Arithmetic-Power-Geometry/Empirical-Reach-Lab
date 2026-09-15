import numpy as np

from empirical_reach.core import (
    blackwell_dominates,
    empirical_partition,
    product_experiment,
    reach_gain,
    unresolved_diameter,
)
from empirical_reach.oed import best_experiment_by_model_ig, model_information_gain
from empirical_reach.worlds import make_hidden_bit_world


def test_current_repertoire_collapses_hidden_consequential_bit():
    worlds, current, candidate, outcome = make_hidden_bit_world()
    part = empirical_partition(current, len(worlds))
    assert set(part) == {(0, 1), (2, 3)}
    assert unresolved_diameter(current, outcome) == 1.0


def test_new_interaction_strictly_expands_reach():
    worlds, current, candidate, outcome = make_hidden_bit_world()
    assert reach_gain(current, candidate, outcome) == 1.0
    assert unresolved_diameter([*current, candidate], outcome) == 0.0
    assert len(empirical_partition([*current, candidate], len(worlds))) == 4


def test_composing_current_experiments_does_not_reveal_hidden_bit():
    worlds, current, candidate, outcome = make_hidden_bit_world()
    product = product_experiment(current)
    assert unresolved_diameter([product], outcome) == 1.0
    # Rows remain identical within each x-class even after joint composition.
    assert np.allclose(product.channel[0], product.channel[1])
    assert np.allclose(product.channel[2], product.channel[3])


def test_blackwell_garbling_can_improve_or_degrade_quality_without_reach_expansion():
    worlds, current, candidate, outcome = make_hidden_bit_world(noise=0.1)
    exact_x, noisy_x, _ = current
    assert blackwell_dominates(exact_x, noisy_x)
    assert unresolved_diameter([exact_x], outcome) == 1.0
    assert unresolved_diameter([noisy_x], outcome) == 1.0


def test_oed_over_supplied_model_family_can_be_maximal_yet_miss_outcome():
    worlds, current, candidate, outcome = make_hidden_bit_world()
    # Conventional supplied model labels distinguish only visible bit x.
    model_labels = [0, 0, 1, 1]
    best, score = best_experiment_by_model_ig(current, model_labels)
    assert best.name == "exact_x"
    assert np.isclose(score, 1.0)
    # Yet that 'optimal' experiment leaves the consequential z-bit unresolved.
    assert unresolved_diameter([best], outcome) == 1.0
    # The new z-interaction carries zero information about the supplied x-model labels.
    assert np.isclose(model_information_gain(candidate, model_labels), 0.0)
    # But it completely resolves the actual consequential outcome.
    assert reach_gain(current, candidate, outcome) == 1.0


def test_discordant_twins_exist_under_current_repertoire():
    worlds, current, candidate, outcome = make_hidden_bit_world()
    part = empirical_partition(current, len(worlds))
    discordant_blocks = [b for b in part if len(set(outcome[list(b)])) > 1]
    assert set(discordant_blocks) == {(0, 1), (2, 3)}
