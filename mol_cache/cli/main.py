"""Command-line entrypoint for the public mol-cache reproduction stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``mol-cache`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mol-cache",
        description="Reproduce SemlaFlow, Tabasco, FLOWR, and FLOWR.root sampling with Pruna mol-caching.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List supported model–dataset routes")
    list_p.set_defaults(func=cmd_list)

    sample_p = sub.add_parser("sample", help="Sample molecules for a model–dataset route")
    sample_p.add_argument("--model", required=True, choices=["semlaflow", "tabasco", "flowr", "flowr_root"])
    sample_p.add_argument("--dataset", required=True, choices=["geom", "spindr", "crossdocked"])
    # Defaults are None so only explicitly supplied flags override Hydra model configs.
    sample_p.add_argument("--n-samples", type=int, default=None)
    sample_p.add_argument("--seed", type=int, default=None)
    sample_p.add_argument("--device", type=int, default=None)
    sample_p.add_argument("--output-dir", type=Path, default=None)
    sample_p.add_argument("--cache-interval", type=int, default=None)
    sample_p.add_argument(
        "--cache-mode",
        default=None,
        choices=["taylor", "ab"],
    )
    sample_p.add_argument("--cache-order", type=int, default=None)
    sample_p.add_argument("--start-step", type=int, default=None)
    sample_p.add_argument("--end-step", type=int, default=None)
    sample_p.add_argument("--max-systems", type=int, default=None, help="SBDD pocket limit")
    sample_p.add_argument("--num-steps", type=int, default=None, help="Tabasco diffusion steps")
    sample_p.add_argument("--integration-steps", type=int, default=None, help="ODE / flow steps")
    sample_p.set_defaults(func=cmd_sample)

    assets_p = sub.add_parser("assets", help="Validate locally placed datasets/checkpoints")
    assets_sub = assets_p.add_subparsers(dest="assets_command", required=True)
    val = assets_sub.add_parser("validate", help="Validate local assets under assets/download/")
    val.add_argument("--model")
    val.add_argument("--dataset")
    val.set_defaults(func=cmd_assets_validate)

    return parser


def cmd_list(_: argparse.Namespace) -> int:
    """Print supported routes."""
    from mol_cache.sample import list_routes

    print("Supported model–dataset routes:")
    for model, dataset in list_routes():
        print(f"  - {model} @ {dataset}")
    print("\nExcluded: flowr @ crossdocked (no pretrained checkpoint)")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Compose Hydra config from CLI flags and run in-process sampling."""
    from mol_cache.config import compose_config, overrides_from_cli
    from mol_cache.sample import sample

    overrides = overrides_from_cli(args)
    cfg = compose_config(args.model, args.dataset, overrides)
    sample(cfg)
    return 0


def cmd_assets_validate(args: argparse.Namespace) -> int:
    """Validate local assets."""
    import json

    from mol_cache.assets import validate_assets

    report = validate_assets(model=args.model, dataset=args.dataset)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    """CLI main; let exceptions raise so researchers see full tracebacks."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
