import csv

import pytest

from ddmr.protocol import (
    SDK_RANGE_PROFILE,
    UnknownRadialMappingError,
    UnsupportedRangeCodeError,
    get_range_profile,
    profile_range_mapping_nm,
)


def test_sdk_profile_maps_only_audited_codes_and_marks_limits():
    profile = get_range_profile("observed-sdk-range-index")
    assert profile is SDK_RANGE_PROFILE
    assert profile.csv_contract_verified is False
    assert profile.radial_mapping == "unknown"
    assert len(profile.range_code_to_nm) == 16
    assert profile.range_code_to_nm[2] == 0.5
    assert profile.range_max_m(2) == 926.0
    assert profile.range_max_m(4) == 1852.0
    assert profile.range_max_m(6) == 3704.0


@pytest.mark.parametrize(
    ("range_code", "range_nm"),
    [(0, 0.125), (1, 0.25), (3, 0.75), (5, 1.5), (7, 3.0)],
)
def test_sdk_profile_includes_audited_codes_zero_through_seven(range_code, range_nm):
    assert SDK_RANGE_PROFILE.range_code_to_nm[range_code] == range_nm
    assert SDK_RANGE_PROFILE.range_max_m(range_code) == range_nm * 1852.0


def test_unknown_range_code_fails_instead_of_using_a_default():
    with pytest.raises(UnsupportedRangeCodeError, match="range code 496"):
        SDK_RANGE_PROFILE.range_max_m(496)


def test_physical_bin_conversion_requires_explicit_radial_assumption():
    with pytest.raises(UnknownRadialMappingError, match="linear_full_range"):
        SDK_RANGE_PROFILE.bin_size_m(2, 868)

    assert SDK_RANGE_PROFILE.bin_size_m(
        2, 868, radial_mapping="linear_full_range"
    ) == pytest.approx(926.0 / 868)


def test_capture_config_mapping_is_explicit():
    assert profile_range_mapping_nm("observed-sdk-range-index") == {
        0: 0.125,
        1: 0.25,
        2: 0.5,
        3: 0.75,
        4: 1.0,
        5: 1.5,
        6: 2.0,
        7: 3.0,
        8: 4.0,
        9: 6.0,
        10: 8.0,
        11: 12.0,
        12: 16.0,
        13: 24.0,
        14: 32.0,
        15: 48.0,
    }

    with pytest.raises(ValueError, match="unknown range profile"):
        profile_range_mapping_nm("made-up")


def _write_capture(path, range_code=2):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        writer.writerow([1, 496, range_code, 50, 100, *([0] * 4)])


def test_extractor_does_not_assume_fixed_range_and_requires_explicit_mapping(
    tmp_path,
):
    from extract_scene import extract_scene_from_csv, load_csv

    path = tmp_path / "capture.csv"
    _write_capture(path)

    raw = load_csv(path)
    assert raw["range_max_m"] is None
    assert raw["bin_size_m"] is None
    assert raw["normalized_bin_size"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="explicit range_profile"):
        extract_scene_from_csv(path, verbose=False)

    mapped = load_csv(
        path,
        "observed-sdk-range-index",
        radial_mapping="linear_full_range",
    )
    assert mapped["range_max_m"] == 926.0
    assert mapped["bin_size_m"] == pytest.approx(926.0 / 4)


def test_extractor_rejects_unknown_code_for_physical_conversion(tmp_path):
    from extract_scene import load_csv

    path = tmp_path / "unknown.csv"
    _write_capture(path, range_code=496)
    with pytest.raises(UnsupportedRangeCodeError, match="range code 496"):
        load_csv(path, "observed-sdk-range-index", "linear_full_range")
