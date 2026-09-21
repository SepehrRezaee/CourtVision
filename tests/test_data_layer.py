"""Tests for the data layer: splitting, validation, conversion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from courtvision.config import SplitConfig
from courtvision.data.conversion import (
    FilterPolicy,
    convert_sequence,
    split_relative_dirs,
    write_dataset_yaml,
)
from courtvision.data.splitting import (
    SPLIT_NAMES,
    SplitError,
    SplitManifest,
    allocate_counts,
    build_split,
    compute_split_id,
    infer_sport,
    summaries_from_names,
    verify_no_leakage,
)
from courtvision.data.synthetic import SyntheticSpec, generate_sequence
from courtvision.data.validation import (
    DatasetValidationReport,
    ValidationOptions,
    validate_dataset,
    validate_sequence,
)

SPEC = SyntheticSpec(sequences_per_sport=1, frames=6, tracks_per_sequence=2, width=160, height=120, seed=5)


def names(per_sport: int = 5) -> list[str]:
    return [f"{sport}_{index:02d}" for sport in ("basketball", "football", "volleyball") for index in range(1, per_sport + 1)]


# -- splitting ---------------------------------------------------------------


def test_no_sequence_appears_in_two_splits() -> None:
    manifest = build_split(summaries_from_names(names()), SplitConfig(seed=1))
    seen: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for entry in manifest.sequences[split]:
            assert entry.name not in seen, f"{entry.name} leaked between {seen[entry.name]} and {split}"
            seen[entry.name] = split
    assert len(seen) == len(names())


def test_leakage_detector_can_fail() -> None:
    """A guard that cannot fail is not a check."""
    entries = summaries_from_names(["basketball_01"])
    manifest = SplitManifest(
        split_id="x",
        seed=1,
        ratios={"train": 0.5, "val": 0.5, "test": 0.0},
        stratify_by_sport=True,
        generated_at="",
        strategy="test",
        assignments={"basketball_01": "train"},
        sequences={"train": entries, "val": entries, "test": []},
    )
    with pytest.raises(SplitError, match="leakage"):
        verify_no_leakage(manifest)


def test_split_is_deterministic_and_order_independent() -> None:
    forward = build_split(summaries_from_names(names()), SplitConfig(seed=7))
    backward = build_split(list(reversed(summaries_from_names(names()))), SplitConfig(seed=7))
    assert forward.assignments == backward.assignments
    assert forward.split_id == backward.split_id


def test_split_id_is_content_addressed() -> None:
    manifest = build_split(summaries_from_names(names()), SplitConfig(seed=5))
    assert manifest.split_id == compute_split_id(manifest)
    assert build_split(summaries_from_names(names()), SplitConfig(seed=5)).split_id == manifest.split_id
    assert build_split(summaries_from_names(names()), SplitConfig(seed=6)).split_id != manifest.split_id


def test_stratification_reaches_every_sport() -> None:
    manifest = build_split(summaries_from_names(names()), SplitConfig(seed=11))
    for split in SPLIT_NAMES:
        assert {entry.sport for entry in manifest.sequences[split]} == {"basketball", "football", "volleyball"}


def test_requested_splits_are_never_silently_empty() -> None:
    manifest = build_split(
        summaries_from_names(names(per_sport=2)), SplitConfig(seed=4, val_ratio=0.25, test_ratio=0.25)
    )
    for split in ("train", "val", "test"):
        assert manifest.sequences[split], f"{split} is empty despite being requested"
    assert manifest.warnings, "the adjustment must be recorded, not silent"


def test_zero_test_ratio_produces_no_test_split() -> None:
    manifest = build_split(summaries_from_names(names(per_sport=6)), SplitConfig(seed=9, test_ratio=0.0))
    assert manifest.sequences["test"] == []
    assert manifest.sequences["val"]


def test_single_sequence_stays_in_train() -> None:
    manifest = build_split(summaries_from_names(["basketball_01"]), SplitConfig(seed=1))
    assert manifest.names("train") == ["basketball_01"]


def test_zero_sequences_and_bad_ratios_are_rejected() -> None:
    with pytest.raises(SplitError, match="zero sequences"):
        build_split([], SplitConfig())
    with pytest.raises(SplitError, match="train ratio"):
        build_split(summaries_from_names(names()), ratios={"train": 0.0, "val": 0.5, "test": 0.5})
    with pytest.raises(SplitError, match="Unknown split names"):
        build_split(summaries_from_names(names()), ratios={"train": 0.5, "validation": 0.5})


def test_allocate_counts_sums_exactly_and_prefers_earlier_splits() -> None:
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    for total in range(1, 25):
        counts = allocate_counts(total, ratios)
        assert sum(counts.values()) == total
        assert counts["train"] >= 1
    assert allocate_counts(2, {"train": 0.5, "val": 0.25, "test": 0.25}) == {"train": 1, "val": 1, "test": 0}
    assert allocate_counts(0, {"train": 1.0, "val": 0.0}) == {"train": 0, "val": 0}


def test_infer_sport_does_not_guess() -> None:
    assert infer_sport("basketball_01") == "basketball"
    assert infer_sport("soccer_1") == "football"
    assert infer_sport("001") == "unknown"
    assert infer_sport("handball_1") == "handball"


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = build_split(summaries_from_names(names(per_sport=3)), SplitConfig(seed=8))
    loaded = SplitManifest.load(manifest.save(tmp_path / "splits.json"))
    assert loaded.split_id == manifest.split_id
    assert loaded.assignments == manifest.assignments


# -- validation --------------------------------------------------------------


def write_gt(sequence_dir: Path, rows: list[str]) -> None:
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def sequence(tmp_path: Path) -> Path:
    generate_sequence(tmp_path, "basketball_01", "basketball", SPEC, seed=5)
    return tmp_path / "basketball_01"


def issue_codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_clean_sequence_has_no_errors(sequence: Path) -> None:
    result = validate_sequence(sequence)
    assert result.ok, [issue.detail for issue in result.issues if issue.severity == "error"]
    assert result.stats.images == SPEC.frames
    # +1: the synthetic generator adds one confidence-0 distractor row, which validation
    # reports faithfully (the conversion policy is what drops it, not the parser).
    assert result.stats.tracks == SPEC.tracks_per_sequence + 1
    assert result.stats.width == SPEC.width


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p: (p / "seqinfo.ini").unlink(), "MISSING_SEQINFO"),
        (lambda p: (p / "gt" / "gt.txt").unlink(), "MISSING_GT"),
        (lambda p: [image.unlink() for image in (p / "img1").iterdir()], "EMPTY_IMAGE_DIR"),
        (lambda p: (p / "gt" / "gt.txt").write_text("1,1,10\n", encoding="utf-8"), "MALFORMED_ANNOTATION"),
        (lambda p: (p / "gt" / "gt.txt").write_text("1,1,10,10,0,20,1,1,1\n", encoding="utf-8"), "DEGENERATE_BOX"),
        (lambda p: (p / "gt" / "gt.txt").write_text("999,1,10,10,20,20,1,1,1\n", encoding="utf-8"), "GT_FRAME_WITHOUT_IMAGE"),
        (
            lambda p: (p / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1,1,1\n1,1,10,10,20,20,1,1,1\n", encoding="utf-8"),
            "DUPLICATE_ANNOTATION",
        ),
    ],
)
def test_issue_codes_fire(sequence: Path, mutate, code: str) -> None:
    mutate(sequence)
    assert code in issue_codes(validate_sequence(sequence))


def test_out_of_bounds_box_is_reported(sequence: Path) -> None:
    write_gt(sequence, ["1,1,-50,10,20,20,1,1,1"])
    assert "BOX_OUT_OF_BOUNDS" in issue_codes(validate_sequence(sequence))


def test_zero_confidence_row_is_informational_not_fatal(sequence: Path) -> None:
    write_gt(sequence, ["1,1,10,10,20,20,0,1,1"])
    issues = [issue for issue in validate_sequence(sequence).issues if issue.code == "ZERO_CONFIDENCE_ANNOTATION"]
    assert issues and issues[0].severity == "info"


def test_unexpected_class_only_reported_when_a_set_is_declared(sequence: Path) -> None:
    write_gt(sequence, ["2,1,10,10,20,20,1,7,1"])
    assert "UNEXPECTED_CLASS" not in issue_codes(validate_sequence(sequence))
    strict = validate_sequence(sequence, options=ValidationOptions(allowed_class_ids=(1,)))
    assert "UNEXPECTED_CLASS" in issue_codes(strict)


def test_missing_frames_and_length_mismatch(sequence: Path) -> None:
    for name in ("000003.jpg", "000004.jpg"):
        (sequence / "img1" / name).unlink()
    codes = issue_codes(validate_sequence(sequence))
    assert {"NON_CONTIGUOUS_FRAMES", "SEQ_LENGTH_MISMATCH"} <= codes


def test_dataset_report_aggregates_by_sport_and_split(tmp_path: Path) -> None:
    for index, sport in enumerate(("basketball", "football", "volleyball"), start=1):
        generate_sequence(tmp_path, f"{sport}_01", sport, SPEC, seed=index)
    report = validate_dataset(
        tmp_path, splits={"basketball_01": "train", "football_01": "val", "volleyball_01": "test"}
    )
    assert set(report.by_sport()) == {"basketball", "football", "volleyball"}
    assert set(report.by_split()) == {"train", "val", "test"}
    assert report.to_dict()["totals"]["sequences"] == 3


def test_dataset_report_round_trips(tmp_path: Path) -> None:
    generate_sequence(tmp_path, "basketball_01", "basketball", SPEC, seed=1)
    report = validate_dataset(tmp_path)
    loaded = DatasetValidationReport.load(report.save(tmp_path / "report.json"))
    assert loaded.issue_counts() == report.issue_counts()
    assert len(loaded.sequences) == len(report.sequences)


# -- conversion --------------------------------------------------------------


def test_conversion_drops_unreviewed_rows_and_writes_labels(sequence: Path, tmp_path: Path) -> None:
    result = convert_sequence(sequence, tmp_path / "out", split="train", link_mode="copy")
    assert result.images == SPEC.frames
    assert result.label_files == SPEC.frames
    assert result.dropped["zero_confidence"] == 1, "the synthetic unreviewed row must be dropped"
    labels = sorted((tmp_path / "out" / "labels" / "train").glob("*.txt"))
    for line in labels[0].read_text(encoding="utf-8").splitlines():
        fields = line.split()
        assert fields[0] == "0", "every label maps to the single YOLO class"
        assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])


def test_conversion_policy_is_enforced(sequence: Path, tmp_path: Path) -> None:
    write_gt(sequence, ["1,1,10,10,20,20,1,7,1", "1,2,40,40,20,20,1,1,0"])
    result = convert_sequence(sequence, tmp_path / "out", split="train", policy=FilterPolicy(min_visibility=0.5))
    assert result.dropped["disallowed_class"] == 1
    assert result.dropped["low_visibility"] == 1
    assert result.kept == 0


def test_conversion_clips_boxes_at_the_frame_edge(tmp_path: Path) -> None:
    """The original defect: a box crossing the edge produced a label outside the image."""
    generate_sequence(tmp_path, "basketball_09", "basketball", SPEC, seed=9)
    sequence_dir = tmp_path / "basketball_09"
    write_gt(sequence_dir, ["1,1,150,10,40,20,1,1,1"])  # extends past width 160
    result = convert_sequence(sequence_dir, tmp_path / "out", split="train", link_mode="copy")
    label = (tmp_path / "out" / "labels" / "train" / "basketball_09_000001.txt").read_text()
    cx, _cy, bw, _bh = (float(value) for value in label.split()[1:])
    assert cx + bw / 2 <= 1.0 + 1e-9, "the label must not extend past the frame"
    assert cx == pytest.approx((150 + 160) / 2 / 160), "centre of the clipped rectangle, not the edge"
    assert result.kept == 1


def test_conversion_is_idempotent(sequence: Path, tmp_path: Path) -> None:
    first = convert_sequence(sequence, tmp_path / "out", split="train", link_mode="copy")
    second = convert_sequence(sequence, tmp_path / "out", split="train", link_mode="copy")
    assert first.images == SPEC.frames
    assert second.images == 0
    assert second.skipped_existing == SPEC.frames


def test_dataset_yaml_omits_absent_splits(tmp_path: Path) -> None:
    path = write_dataset_yaml(tmp_path, split_relative_dirs(["train", "val"]))
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["train"] == "images/train"
    assert "test" not in document, "an absent split must not be advertised"
    assert document["names"] == {0: "player"}


def test_unknown_link_mode_is_rejected(sequence: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="link_mode"):
        convert_sequence(sequence, tmp_path / "out", split="train", link_mode="teleport")
