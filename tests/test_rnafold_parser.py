from pathlib import Path

from models.etd_multitask.rnafold import parse_dot_ps_ubox


def test_parse_dot_ps_ubox_square_probability(tmp_path: Path):
    dot_ps = tmp_path / "sample_dp.ps"
    dot_ps.write_text(
        """
% some header
%start of base pair probability data
1 8 0.5 ubox
3 6 0.25 ubox
showpage
""".strip()
    )

    pairs = parse_dot_ps_ubox(dot_ps)

    assert (0, 7) in pairs
    assert (2, 5) in pairs
    assert abs(pairs[(0, 7)] - 0.25) < 1e-8
    assert abs(pairs[(2, 5)] - 0.0625) < 1e-8
