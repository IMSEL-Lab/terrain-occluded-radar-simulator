"""Explicit range-code and radial-calibration profiles for Furuno captures.

These profiles describe observed controller code tables. They do not assert
that a given CSV exporter wrote the `Range` field using that table, nor do they
establish how exported bins map onto physical distance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

METERS_PER_NAUTICAL_MILE = 1852.0


class UnsupportedRangeCodeError(ValueError):
    """Raised when a selected profile does not define a raw range code."""


class UnknownRadialMappingError(ValueError):
    """Raised when a physical bin conversion lacks an explicit radial model."""


@dataclass(frozen=True, slots=True)
class RangeProfile:
    """A source-backed raw range-code mapping with explicit evidence limits."""

    name: str
    range_code_to_nm: Mapping[int, float]
    source_description: str
    csv_contract_verified: bool = False
    radial_mapping: str = "unknown"

    def range_max_m(self, range_code: int) -> float:
        try:
            range_nm = self.range_code_to_nm[range_code]
        except KeyError as exc:
            supported = ", ".join(str(code) for code in self.range_code_to_nm)
            raise UnsupportedRangeCodeError(
                f"range code {range_code} is unsupported by profile {self.name!r}; "
                f"supported codes: {supported}"
            ) from exc
        return range_nm * METERS_PER_NAUTICAL_MILE

    def bin_size_m(
        self,
        range_code: int,
        num_bins: int,
        *,
        radial_mapping: str | None = None,
    ) -> float:
        """Return uniform bin width only under an explicit linear assumption."""

        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        if radial_mapping != "linear_full_range":
            raise UnknownRadialMappingError(
                "physical bin conversion requires radial_mapping="
                "'linear_full_range'; the captured-bin mapping is unverified"
            )
        return self.range_max_m(range_code) / num_bins


# Both the Gen2 SDK renderer and Gen3 protocol table map these codes to the
# selected display range. The original CSV serializer has not been identified.
SDK_RANGE_PROFILE = RangeProfile(
    name="observed-sdk-range-index",
    range_code_to_nm=MappingProxyType(
        {
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
    ),
    source_description=(
        "Gen2 NavNet SDK range table and Gen3 wire-index table agree on "
        "codes 0 through 15; CSV field provenance remains unverified."
    ),
)

RANGE_PROFILES: Mapping[str, RangeProfile] = MappingProxyType(
    {SDK_RANGE_PROFILE.name: SDK_RANGE_PROFILE}
)


def get_range_profile(name: str) -> RangeProfile:
    """Return a named, explicit profile or reject an unknown profile name."""

    try:
        return RANGE_PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(RANGE_PROFILES)
        raise ValueError(
            f"unknown range profile {name!r}; available profiles: {supported}"
        ) from exc


def profile_range_mapping_nm(name: str) -> Mapping[int, float]:
    """Return the profile's explicit range-code map for CaptureConfig."""

    return get_range_profile(name).range_code_to_nm
