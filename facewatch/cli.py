"""Command-line entry point for Facewatch configuration and watchlist operations."""

from __future__ import annotations

import argparse
import logging

from facewatch.config import CameraConfig, load_config


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser and document which arguments belong to each command."""
    p = argparse.ArgumentParser(prog="facewatch")

    # Configuration is global because every command needs watchlist and model settings.
    p.add_argument("--config", help="YAML config file")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Camera arguments override the YAML camera list rather than extending it.
    run = sub.add_parser("run", help="start monitoring")
    run.add_argument("--camera", action="append", default=[],
                     help="camera URI; repeat for several (overrides cameras in the config)")
    run.add_argument("--no-display", action="store_true")

    # Enrollment combines embeddings from every supplied image into one watchlist entry.
    en = sub.add_parser("enroll", help="add a person to the watchlist")
    en.add_argument("--id", required=True)
    en.add_argument("--name", required=True)
    en.add_argument("images", nargs="+")

    # The remaining commands query or mutate the watchlist repository directly.
    sub.add_parser("list", help="list watchlist entries")
    rm = sub.add_parser("remove", help="remove a watchlist entry")
    rm.add_argument("id")
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and dispatch the requested Facewatch operation."""
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Heavy imports stay inside command dispatch so unrelated commands do not initialize
    # the face model, camera stack, or alert workers unnecessarily.
    from facewatch.bootstrap import build_manager, build_repository

    if args.cmd == "run":
        # Ad-hoc camera URIs replace the configured camera list, including RTSP, HTTP,
        # file, and numeric webcam sources accepted by OpenCV.
        if args.camera:
            cfg.cameras = [CameraConfig(f"cam-{i}", uri) for i, uri in enumerate(args.camera)]
        if args.no_display:
            cfg.display = False
        cfg.validate()
        build_manager(cfg).run()
    elif args.cmd == "enroll":
        from facewatch.analysis import InsightFaceAnalyzer
        from facewatch.enrollment import Enroller

        # Enrollment uses the configured analyzer and a stable average of all image embeddings.
        entry = Enroller(InsightFaceAnalyzer(cfg.analyzer), build_repository(cfg)).enroll(
            args.id, args.name, args.images)
        print(f"Enrolled {entry.name} ({entry.id}) from {len(args.images)} image(s)")
    elif args.cmd == "list":
        for e in build_repository(cfg).list_all():
            print(f"{e.id}\t{e.name}")
    elif args.cmd == "remove":
        print("Removed" if build_repository(cfg).remove(args.id) else "Not found")
    return 0
