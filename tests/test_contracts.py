import pytest
from pydantic import ValidationError

from longtail.contracts import (
    TAXONOMY,
    ActorSpec,
    CandidateWindow,
    Finding,
    Frame2D,
    PredicateCall,
    PredicateResult,
    ScenarioSpec,
)


def _spec(**overrides):
    data = {
        "query": "a pedestrian crossing in front of us while we move",
        "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
        "predicates": [
            {"name": "crossing_ego_path", "actors": ["a1"], "args": {}},
            {"name": "ego_moving", "actors": [], "args": {}},
        ],
        "assumptions": ["crossing = enters the ego corridor within 3 s"],
        "clarification": None,
    }
    data.update(overrides)
    return ScenarioSpec.model_validate(data)


def _result(status="PASS", required=True, value=1.2, margin=0.7, rid="r1"):
    evidence = {"reason": "test"}
    if margin is not None:
        evidence["margin"] = margin
    return PredicateResult(
        result_id=rid,
        name="moving",
        actors={"a1": "t1"},
        status=status,
        value=value,
        threshold="> 0.5 m/s",
        evidence=evidence,
        required=required,
    )


def _finding(results, **overrides):
    data = {
        "finding_id": "f1",
        "scene_id": "002",
        "t0": 1.0,
        "t1": 3.0,
        "spec_hash": "abc",
        "bindings": {"a1": "t1"},
        "results": results,
        "score": 0.9,
        "cost_usd": 0.01,
    }
    data.update(overrides)
    return Finding(**data)


# ScenarioSpec and PredicateCall


def test_valid_spec_parses_with_defaults():
    spec = _spec()
    assert spec.min_duration_s == 0.5
    assert spec.scene_filters == {}
    assert [p.required for p in spec.predicates] == [True, True]


def test_spec_round_trips_through_json():
    spec = _spec()
    assert ScenarioSpec.model_validate_json(spec.model_dump_json()) == spec


def test_ego_predicates_take_no_actors():
    with pytest.raises(ValidationError, match="ego_moving takes 0 actor"):
        PredicateCall(name="ego_moving", actors=["a1"], args={})


def test_unknown_predicate_rejected():
    with pytest.raises(ValidationError):
        PredicateCall(name="teleporting", actors=["a1"], args={})


@pytest.mark.parametrize(
    "call, message",
    [
        ({"name": "near", "actors": ["a1"], "args": {"m": 5.0}}, "takes 2 actor"),
        ({"name": "near", "actors": ["a1", "a1"], "args": {"m": 5.0}}, "repeats an actor"),
        ({"name": "within_distance", "actors": ["a1"], "args": {}}, "takes args"),
        ({"name": "within_distance", "actors": ["a1"], "args": {"m": 5.0, "x": 1.0}}, "takes args"),
        ({"name": "within_distance", "actors": ["a1"], "args": {"m": -1.0}}, "positive"),
        ({"name": "within_distance", "actors": ["a1"], "args": {"m": "ten"}}, "positive"),
        ({"name": "relative_position", "actors": ["a1"], "args": {"side": "above"}}, "side must be"),
        ({"name": "occluded_by", "actors": ["a1"], "args": {"cls": "tree"}}, "cls must be"),
    ],
)
def test_predicate_signature_checks(call, message):
    with pytest.raises(ValidationError, match=message):
        PredicateCall(**call)


def test_integer_distance_is_coerced_to_float():
    call = PredicateCall(name="within_distance", actors=["a1"], args={"m": 10})
    assert call.args["m"] == 10.0


def test_predicate_must_reference_declared_actor():
    with pytest.raises(ValidationError, match="undeclared actors"):
        _spec(predicates=[{"name": "moving", "actors": ["a2"], "args": {}}])


def test_duplicate_actor_ids_rejected():
    actor = {"id": "a1", "cls": "car", "sam_prompt": "car"}
    with pytest.raises(ValidationError, match="duplicate actor ids"):
        _spec(actors=[actor, actor], predicates=[])


def test_class_outside_taxonomy_rejected():
    assert "pedestrian" in TAXONOMY
    with pytest.raises(ValidationError):
        ActorSpec(id="a1", cls="tree", sam_prompt="tree")


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _spec(priority="high")


def test_assumptions_and_clarification_are_required():
    with pytest.raises(ValidationError):
        ScenarioSpec(query="q", actors=[], predicates=[])


# Other models


def test_window_times_ordered():
    with pytest.raises(ValidationError, match="before t0"):
        CandidateWindow(window_id="w", scene_id="s", t0=2.0, t1=1.0, cameras=[], score=0, source="uniform")


def test_frame_box_ordered_and_optional_fields():
    frame = Frame2D(frame_idx=0, ts=1.0, box_xyxy=(1, 2, 3, 4))
    assert frame.mask_rle is None and frame.conf is None
    with pytest.raises(ValidationError, match="x1 <= x2"):
        Frame2D(frame_idx=0, ts=1.0, box_xyxy=(5, 2, 3, 4))


def test_result_margin_required_when_value_set():
    with pytest.raises(ValidationError, match="margin must be a number"):
        _result(margin=None)
    with pytest.raises(ValidationError, match="margin must be a number"):
        _result(margin=True)


def test_result_margin_absent_when_no_value():
    assert _result(status="UNVERIFIABLE", value=None, margin=None).value is None
    with pytest.raises(ValidationError, match="margin must be absent"):
        _result(status="UNVERIFIABLE", value=None, margin=0.3)


# Finding.verified


def test_verified_when_every_required_result_passes():
    assert _finding([_result(), _result(rid="r2")]).verified is True


@pytest.mark.parametrize("status", ["FAIL", "UNVERIFIABLE"])
def test_not_verified_when_a_required_result_does_not_pass(status):
    value = None if status == "UNVERIFIABLE" else 0.1
    margin = None if status == "UNVERIFIABLE" else -0.4
    failing = _result(status=status, value=value, margin=margin, rid="r2")
    assert _finding([_result(), failing]).verified is False


def test_optional_failure_does_not_block_verification():
    optional = _result(status="FAIL", required=False, value=0.1, margin=-0.4, rid="r2")
    assert _finding([_result(), optional]).verified is True


def test_not_verified_without_required_results():
    assert _finding([]).verified is False
    assert _finding([_result(required=False)]).verified is False


def test_verified_cannot_be_set_to_a_wrong_value():
    with pytest.raises(ValidationError, match="verified is computed"):
        _finding([_result(status="FAIL", value=0.1, margin=-0.4)], verified=True)


def test_verified_cannot_be_assigned():
    finding = _finding([_result()])
    with pytest.raises((AttributeError, ValidationError)):
        finding.verified = False
    assert finding.verified is True


def test_verified_follows_results():
    finding = _finding([_result()])
    finding.results.append(_result(status="FAIL", value=0.1, margin=-0.4, rid="r2"))
    assert finding.verified is False


def test_finding_round_trips_with_verified_in_json():
    finding = _finding([_result()])
    dumped = finding.model_dump()
    assert dumped["verified"] is True
    assert Finding.model_validate(dumped) == finding
    assert Finding.model_validate_json(finding.model_dump_json()).verified is True


def test_verified_not_an_input_in_json_schema():
    schema = Finding.model_json_schema(mode="validation")
    assert "verified" not in schema["properties"]
