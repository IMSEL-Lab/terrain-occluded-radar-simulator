"""Command line for scene generation and capture reconstruction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import __version__

_COMMANDS = {
    "synthetic": "Generate a seeded scene and render a radar sweep.",
    "scene": "Render a manually arranged scene JSON.",
    "location": "Render a .radarloc file with location-specific controls.",
    "map": "Render shoreline geometry supplied as GeoJSON.",
    "furuno": "Reconstruct and score the Furuno capture configurations.",
    "canoe": "Reconstruct CANOE capture support with a fitted intensity profile.",
    "evaluate-map": "Run the frozen CANOE shoreline evaluation.",
    "audit-geometry": "Audit the frozen 100-scene synthetic geometry grid.",
    "calibrate-canoe": "Fit the CANOE intensity profile from the frozen training scans.",
    "verify-canoe-profile": "Check the fitted CANOE profile and training masks.",
}


def _entrypoint(command: str) -> Callable[[list[str]], int | None]:
    if command == "scene":
        from . import manual_scene as entry
    elif command == "location":
        from coastline_defined import run_coastline_defined as entry
    elif command == "synthetic":
        from scripts import run_synthetic_scene_demo as entry
    elif command == "map":
        from . import map_scene as entry
    elif command == "furuno":
        from scripts import run_furuno_conditioned_diagnostic as entry
    elif command == "canoe":
        from scripts import run_canoe_capture_reconstruction as entry
    elif command == "evaluate-map":
        from scripts import run_canoe_legacy_shoreline_diagnostic as entry
    elif command == "audit-geometry":
        from scripts import audit_legacy_random_geometry as entry
    elif command == "calibrate-canoe":
        from scripts import calibrate_canoe as entry
    else:
        from scripts import verify_canoe_profile as entry
    return entry.main


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="tors",
        description="Terrain-Occluded Radar Simulator",
        epilog="\n".join(
            f"{name:<22} {help_text}" for name, help_text in _COMMANDS.items()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=_COMMANDS, nargs="?")
    if not args or args[0] not in _COMMANDS:
        parser.parse_args(args)
        parser.print_help()
        return 0
    command, *command_args = args
    sys.argv[0] = f"tors {command}"
    try:
        return _entrypoint(command)(command_args) or 0
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"tors {command}: {exc}", file=sys.stderr)
        return 2
