import json

from ecnca.real.musique import load_musique, normalise_answer, parse_record


def fixture(second_question="When was #1 founded?", final="1960"):
    return {"id": "2hop__a_b", "question": "When was it founded?",
            "answer": final, "answer_aliases": [], "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "First", "paragraph_text": "A owns B.",
                 "is_supporting": True},
                {"idx": 1, "title": "Second", "paragraph_text": "B began in 1960.",
                 "is_supporting": True}],
            "question_decomposition": [
                {"id": 1, "question": "A >> owned by", "answer": "B",
                 "paragraph_support_idx": 0},
                {"id": 2, "question": second_question, "answer": "1960",
                 "paragraph_support_idx": 1}]}


def test_linear_chain_requires_immediate_dependency_and_terminal_match():
    assert parse_record(fixture()).is_linear
    assert not parse_record(fixture(second_question="When was it founded?")).is_linear
    assert not parse_record(fixture(final="1961")).is_linear


def test_loader_fails_loudly_on_bad_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(fixture()) + "\n{bad\n")
    try:
        load_musique(path)
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("bad JSONL was silently accepted")


def test_answer_normalisation_matches_official_style():
    assert normalise_answer("The Miquette Giraudy!") == "miquette giraudy"
