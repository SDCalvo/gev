"""The programmatic data generators: contrastive policy pairs (gev.contrastive), compositional rule records (gev.composition)
No weights, no network.
Run: uv run python -m pytest tests/test_generators.py -q
"""
import pytest

from gev.contrastive import generate, paired_flip


def test_rendered_pairs_change_one_sentence_not_order():
    rows, _ = generate(20, 17)
    for a, b in zip(rows[::2], rows[1::2]):
        left = a["state"]["case"].split(". ")
        right = b["state"]["case"].split(". ")
        assert len(left) == len(right)
        assert sum(x != y for x, y in zip(left, right)) == 1

def test_pair_metric_compares_semantics_not_indices():
    rows = [
        {"pair_id": "p", "sibling": "a", "keys": ["deny", "allow"], "label": 1, "p": [0, 1]},
        {"pair_id": "p", "sibling": "b", "keys": ["allow", "deny"], "label": 1, "p": [0, 1]},
    ]
    assert paired_flip(rows)["flip_rate"] == 1
    assert paired_flip(rows)["both_correct_rate"] == 1

def test_pair_metric_refuses_missing_sibling():
    with pytest.raises(ValueError, match="incomplete"):
        paired_flip([{"pair_id": "p", "sibling": "a", "keys": ["x", "y"], "label": 0, "p": [1, 0]}])

def test_compositional_truth_tables_and_unknowns():
    from itertools import product
    from gev.composition import evaluate_rule
    atoms = [{"kind": "flag", "fields": [k], "threshold": 0} for k in ("a", "b", "c")]
    for a, b, c in product((True, False), repeat=3):
        facts = dict(a=a, b=b, c=c)
        assert evaluate_rule(("and", ("or", 0, 1), 2), atoms, facts) == ((a or b) and c)
        assert evaluate_rule(("unless", 0, 1), atoms, facts) == (a and not b)
        assert evaluate_rule(("if", 0, 1, 2), atoms, facts) == (b if a else c)
    assert evaluate_rule(("and", 0, 1), atoms, {"a": False}) is False
    assert evaluate_rule(("and", 0, 1), atoms, {"a": True}) is None
    assert evaluate_rule(("or", 0, 1), atoms, {"a": False}) is None

@pytest.mark.parametrize("kind,threshold,value,expected", [
    ("lt", 10, 10, False), ("le", 10, 10, True), ("gt", 10, 10, False),
    ("ge", 10, 10, True), ("eq", 10, 11, False), ("range", 10, 20, True), ("range", 10, 21, False),
])
def test_rule_boundary_labels(kind, threshold, value, expected):
    from gev.composition import atom_value
    assert atom_value({"kind": kind, "threshold": threshold, "fields": ["x"]}, {"x": value}) == expected

def test_compositional_pairs_validate_and_keep_invariance():
    from gev.composition import SHAPES, TRAIN_SHAPES, DEV_SHAPES, TEST_SHAPES, check_group, generate as compose
    from gev.benchmark import labels, prediction_rows
    assert not set(TRAIN_SHAPES) & (set(DEV_SHAPES) | set(TEST_SHAPES))
    records = compose(4, "test", tuple(SHAPES))
    rows = []
    for i in range(0, len(records), 4):
        group = records[i:i + 4]
        assert check_group(group)
        for r in group:
            q = r["questions"]["decision"]
            keys, y = labels(q)
            rows += prediction_rows(r, {"probabilities": {"decision": {k: int(i == y) for i, k in enumerate(keys)}}})
    summary = paired_flip(rows)
    assert summary["both_correct_rate"] == 1
    assert summary["invariance_rate"] == 1
    assert summary["invariant_both_correct_rate"] == 1
    records[0]["state"]["case"] = "The facts were changed."
    with pytest.raises(ValueError, match="rendered facts"):
        check_group(records[:4])

def test_date_and_entity_atoms():
    from gev.composition import atom_value
    a = {"kind": "elapsed", "fields": ["start", "end"], "threshold": 2}
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-01"}) is True
    assert atom_value(a, {"start": "2028-02-28", "end": "2028-03-02"}) is False
    a = {"kind": "match", "fields": ["signer", "approver"], "threshold": 0}
    assert atom_value(a, {"signer": "Mira", "approver": "Mira"}) is True
    assert atom_value(a, {"signer": "Mira", "approver": "Noah"}) is False
    assert atom_value(a, {"signer": "Mira"}) is None

def test_random_rule_structures_exclude_heldout_and_cover_negation():
    from gev.composition import SHAPES, DEV_SHAPES, TEST_SHAPES, canonical, push_negation, sample_trees, generate as compose, check_group
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])       # order/numbering-independent
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))   # De Morgan
    trees = sample_trees(30, "t")
    held = {canonical(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES}
    assert len({canonical(t) for t in trees}) == 30 and not any(canonical(t) in held or canonical(push_negation(t)) in held for t in trees)
    assert any("not(" in canonical(t) for t in trees)
    recs = compose(1, "t", styles=(3, 4), trees={f"rand{i}": t for i, t in enumerate(trees[:5])})
    for i in range(0, len(recs), 4):
        assert check_group(recs[i:i + 4])

def test_ordinal_threshold_families_are_balanced_minimal_pairs():
    import collections
    from gev.contrastive import ORDINAL_FAMILIES, generate
    from gev.data import materialize
    recs, rep = generate(30, "t", families=list(ORDINAL_FAMILIES))
    assert all(v["pairs"] == 30 for v in rep.values())
    for a, b in zip(recs[::2], recs[1::2]):
        assert a["questions"]["decision"]["type"] == "score" and a["questions"]["decision"]["label"] != b["questions"]["decision"]["label"]
        assert sum(x != y for x, y in zip(a["state"]["case"].split(". "), b["state"]["case"].split(". "))) == 1
        materialize(a)
    counts = collections.Counter((r["_meta"]["family"], r["questions"]["decision"]["label"]) for r in recs)
    assert all(counts[(f, level)] >= 8 for f in ORDINAL_FAMILIES for level in (0, 1, 2))   # every level appears in every family
