"""The MuSiQue files are CC BY 4.0 and are not redistributed through this
repository, so the DGX scripts must fail fast and explicitly when they are
absent rather than part way into an expensive sweep."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from analysis.musique_audit import TRAIN_SHA256, DEV_SHA256

SCRIPT = Path("scripts/musique_data.py")
NAMES = ["musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl"]


def run(root):
    return subprocess.run([sys.executable, str(SCRIPT), "verify", "--root", str(root)],
                          capture_output=True, text=True)


def test_absent_dataset_fails_and_names_both_files(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 2
    for name in NAMES:
        assert name in r.stderr
    assert r.stderr.count("MISSING") == 2


def test_failure_points_at_the_restore_command(tmp_path):
    r = run(tmp_path)
    assert "musique_data.py restore" in r.stderr


def test_wrong_content_is_reported_as_corrupt_not_missing(tmp_path):
    for name in NAMES:
        (tmp_path / name).write_text('{"id": "x", "answerable": true}\n')
    r = run(tmp_path)
    assert r.returncode == 2
    assert "CORRUPT" in r.stderr and "MISSING" not in r.stderr
    assert TRAIN_SHA256 in r.stderr and DEV_SHA256 in r.stderr


def test_digests_come_from_the_audit_module():
    """One source of truth: the guard must not carry its own copy."""
    src = SCRIPT.read_text()
    assert "from analysis.musique_audit import TRAIN_SHA256, DEV_SHA256" in src
    assert TRAIN_SHA256 not in src and DEV_SHA256 not in src


def test_musique_stage_script_checks_data_before_importing_encoders():
    env = Path("scripts/dgx_musique.sh").read_text().split("env)")[1].split(";;")[0]
    assert env.index("musique_data.py") < env.index("sentence_transformers")
    assert "restore" in env and "verify" in env


def test_shipped_archives_are_present_and_small_enough_for_git():
    manifest = json.loads(Path("artifacts/musique/manifest.json").read_text())
    assert manifest["license"] == "CC BY 4.0" and manifest["attribution"]
    assert manifest["source"] == "https://github.com/StonyBrookNLP/musique"
    total = 0
    for name, meta in manifest["files"].items():
        archive = Path("artifacts/musique") / meta["archive"]
        assert archive.is_file()
        assert archive.stat().st_size == meta["archive_bytes"]
        # GitHub hard-rejects blobs over 100 MB and warns over 50 MB.
        assert archive.stat().st_size < 50 * 1024 * 1024
        total += meta["archive_bytes"]
    assert total < 100 * 1024 * 1024


def test_archive_digests_match_the_manifest():
    manifest = json.loads(Path("artifacts/musique/manifest.json").read_text())
    for name, meta in manifest["files"].items():
        archive = Path("artifacts/musique") / meta["archive"]
        found = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert found == meta["archive_sha256"], name
        assert meta["sha256"] == {"musique_ans_v1.0_train.jsonl": TRAIN_SHA256,
                                  "musique_ans_v1.0_dev.jsonl": DEV_SHA256}[name]


def test_restore_reproduces_the_audited_bytes(tmp_path):
    """The decompressed files must match the digests in the frozen audit."""
    r = subprocess.run([sys.executable, str(SCRIPT), "restore", "--root", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert run(tmp_path).returncode == 0
    for name, expected in [("musique_ans_v1.0_train.jsonl", TRAIN_SHA256),
                           ("musique_ans_v1.0_dev.jsonl", DEV_SHA256)]:
        h = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert h == expected


def test_restore_repairs_only_the_damaged_file(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "restore", "--root", str(tmp_path)],
                   capture_output=True, text=True)
    (tmp_path / "musique_ans_v1.0_dev.jsonl").write_text("corrupt\n")
    assert run(tmp_path).returncode == 2
    r = subprocess.run([sys.executable, str(SCRIPT), "restore", "--root", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "present: " in r.stdout and "musique_ans_v1.0_train" in r.stdout
    assert run(tmp_path).returncode == 0


def test_restore_leaves_no_partial_file_behind(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "restore", "--root", str(tmp_path)],
                   capture_output=True, text=True)
    assert not list(tmp_path.glob("*.partial"))


def test_final_driver_validates_data_before_any_expensive_stage():
    s = Path("scripts/dgx_final_iclr.sh").read_text()
    assert s.index("musique_data.py verify") < s.index("dgx_semantic_sybil.sh audit")
    assert s.index("musique_data.py verify") < s.index("gate_c_checkpoints.py verify")


def test_final_driver_reuses_a_completed_sybil_audit():
    s = Path("scripts/dgx_final_iclr.sh").read_text()
    block = s.split("Resumable:")[1].split("stage \"musique: env\"")[0]
    assert 'population"]==200' in block, "reuse must require a complete report"
    # collect is cheap and derives summary.log, so it must run either way.
    assert block.count("dgx_semantic_sybil.sh collect") == 1
    assert block.index("fi") < block.index("dgx_semantic_sybil.sh collect")
