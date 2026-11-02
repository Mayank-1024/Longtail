"""M5 cards: the validator rejects every unsupported sentence; the card falls back to a template.

Findings come from the real verifier on the synthetic scene; only the card model is scripted.
"""

import json
from pathlib import Path

import pytest
from synthetic import SyntheticScene, track_source

from longtail.contracts import ScenarioSpec
from longtail.explain.cards import make_card, template_card
from longtail.explain.validate import Card, Sentence, validate_card
from longtail.llm.baseten import SPARK_OPEN, BasetenTransport, from_chat, spark_open_client, to_chat
from longtail.llm.cache import ResponseCache
from longtail.llm.client import ModelClient, PaidCallsDisabled
from longtail.llm.ledger import Ledger
from longtail.pipeline import run

SPEC = ScenarioSpec.model_validate(
    {
        "query": "a pedestrian crossing in front of us while we move",
        "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
        "predicates": [
            {"name": "exists", "actors": ["a1"], "args": {}},
            {"name": "crossing_ego_path", "actors": ["a1"], "args": {}},
            {"name": "ego_moving", "actors": [], "args": {}},
        ],
        "assumptions": [],
        "clarification": None,
    }
)


@pytest.fixture(scope="module")
def findings(tmp_path_factory):
    crossing, empty = SyntheticScene(scene_id="crossing"), SyntheticScene(scene_id="empty")
    sources = {"crossing": track_source(crossing, {"person": ["crosser"]}), "empty": track_source(empty, {})}
    result = run(
        SPEC,
        ["crossing", "empty"],
        {"crossing": crossing, "empty": empty}.__getitem__,
        lambda scn, cam, concept: sources[scn.scene_id](scn, cam, concept),
        tmp_path_factory.mktemp("runs"),
    )
    verified = next(f for f in result.findings() if f.verified)
    rejected = next(f for f in result.findings() if not f.verified)
    return verified, rejected


class ChatScript:
    """A fake chat.completions endpoint: returns the scripted texts in order and records each payload."""

    def __init__(self, texts: list[str]) -> None:
        self.texts, self.payloads = list(texts), []

    def create(self, **payload):
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"role": "assistant", "content": self.texts.pop(0)}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 120},
        }


def client_for(tmp_path: Path, chat: ChatScript) -> ModelClient:
    return ModelClient(
        BasetenTransport("test", chat=chat),
        cache=ResponseCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        sleep=lambda s: None,
    )


def rid(finding, name):
    return next(r.result_id for r in finding.results if r.name == name)


def good_card(finding) -> str:
    return json.dumps(
        {
            "title": "Pedestrian crossing ahead",
            "sentences": [
                {
                    "text": "A pedestrian crossed into our path while we were moving.",
                    "cites": [rid(finding, "crossing_ego_path"), rid(finding, "ego_moving")],
                },
                {"text": "LiDAR confirmed the pedestrian.", "cites": [rid(finding, "exists")]},
            ],
        }
    )


def test_template_cards_always_pass(findings):
    for f in findings:
        card = template_card(f, SPEC)
        assert validate_card(card, f, SPEC) == [], card.to_json()
        assert all(s.cites for s in card.sentences)


def test_a_sentence_with_no_citation_is_rejected(findings):
    verified, _ = findings
    card = Card("Pedestrian crossing", [Sentence("The pedestrian was looking at a phone.", [])])
    assert any("cites nothing" in e for e in validate_card(card, verified, SPEC))


def test_a_failed_result_stated_as_fact_is_rejected(findings):
    _, rejected = findings
    bad = next(r for r in rejected.results if r.status != "PASS")
    card = Card("Pedestrian crossing", [Sentence("A pedestrian crossed in front of us.", [bad.result_id])])
    assert any("as fact" in e for e in validate_card(card, rejected, SPEC))
    hedged = Card("Pedestrian crossing", [Sentence("The pedestrian could not be verified.", [bad.result_id])])
    assert validate_card(hedged, rejected, SPEC) == []


def test_a_number_that_matches_no_evidence_is_rejected(findings):
    verified, _ = findings
    card = Card("Crossing", [Sentence("The pedestrian was 37.4 m away.", [rid(verified, "exists")])])
    assert any("37.4" in e for e in validate_card(card, verified, SPEC))


def test_unknown_result_ids_and_numeric_titles_are_rejected(findings):
    verified, _ = findings
    card = Card("Crossing at 9 m", [Sentence("Seen.", ["nope"])])
    errors = validate_card(card, verified, SPEC)
    assert any("unknown result ids" in e for e in errors) and any("title" in e for e in errors)


def test_a_valid_model_card_is_kept(tmp_path, findings):
    verified, _ = findings
    chat = ChatScript([good_card(verified)])
    card = make_card(verified, SPEC, client_for(tmp_path, chat))
    assert card.source == "model" and card.errors == [] and len(chat.payloads) == 1
    assert chat.payloads[0]["model"] == SPARK_OPEN


def test_a_planted_unsupported_sentence_gets_one_retry_then_the_template(tmp_path, findings):
    verified, _ = findings
    planted = json.dumps(
        {
            "title": "Crossing",
            "sentences": [{"text": "The pedestrian ran at 9.9 m/s.", "cites": [rid(verified, "exists")]}],
        }
    )
    chat = ChatScript([planted, planted])
    card = make_card(verified, SPEC, client_for(tmp_path, chat))
    assert card.source == "template" and len(chat.payloads) == 2
    assert "rejected" in chat.payloads[1]["messages"][-1]["content"]
    assert validate_card(card, verified, SPEC) == []


def test_retry_can_recover(tmp_path, findings):
    verified, _ = findings
    chat = ChatScript(["not json", f"```json\n{good_card(verified)}\n```"])
    card = make_card(verified, SPEC, client_for(tmp_path, chat))
    assert card.source == "model" and len(card.errors) == 1


def test_no_client_means_the_template_and_no_call(findings):
    verified, _ = findings
    assert make_card(verified, SPEC, None).source == "template"


def test_spark_open_client_is_off_until_paid_calls_are_allowed(monkeypatch, findings, tmp_path):
    monkeypatch.setenv("BASETEN_API_KEY", "x")
    monkeypatch.delenv("LONGTAIL_ALLOW_PAID", raising=False)
    monkeypatch.setenv("LONGTAIL_CACHE_DIR", str(tmp_path / "cache"))
    client = spark_open_client()
    client.ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(PaidCallsDisabled):
        make_card(findings[0], SPEC, client)


def test_baseten_request_and_response_shapes():
    payload = to_chat(
        {
            "model": SPARK_OPEN,
            "input": [
                {"role": "system", "content": "s"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "t"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA"},
                    ],
                },
            ],
            "max_output_tokens": 50,
            "temperature": 0,
        }
    )
    assert payload["max_tokens"] == 50 and payload["temperature"] == 0
    assert payload["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AA"},
    }
    out = from_chat(
        {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    )
    assert out["output"][0]["content"][0]["text"] == "hi"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 4}
