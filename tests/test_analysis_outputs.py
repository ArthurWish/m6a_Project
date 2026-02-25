import pandas as pd

from scripts.analysis.analyze_selective_binding import _summarize_selective
from scripts.analysis.mine_weak_binding_candidates import is_weak_candidate


def test_selective_summary_fields():
    df = pd.DataFrame(
        {
            "p_A": [0.8, 0.6],
            "p_C": [0.4, 0.5],
            "p_G": [0.3, 0.4],
            "p_U": [0.2, 0.35],
            "u_A": [0.2, 0.25],
            "u_C": [0.5, 0.55],
            "u_G": [0.6, 0.65],
            "u_U": [0.7, 0.75],
        }
    )

    out = _summarize_selective(df)
    assert out["n_sites"] == 2
    assert set(out["mean_prob"].keys()) == {"A", "C", "G", "U"}
    assert set(out["delta_prob_vs_A"].keys()) == {"C", "G", "U"}
    assert out["delta_prob_vs_A"]["C"] < 0
    assert out["delta_unc_vs_A"]["C"] > 0


def test_weak_candidate_filter():
    assert is_weak_candidate(-1, 0.35, 0.8, prob_min=0.2, prob_max=0.6, unc_min=0.6)
    assert not is_weak_candidate(1, 0.35, 0.8, prob_min=0.2, prob_max=0.6, unc_min=0.6)
    assert not is_weak_candidate(-1, 0.1, 0.8, prob_min=0.2, prob_max=0.6, unc_min=0.6)
    assert not is_weak_candidate(-1, 0.35, 0.2, prob_min=0.2, prob_max=0.6, unc_min=0.6)
