"""Asset manifest loading, path resolution, and presence validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the public reproduction repository root."""
    return Path(__file__).resolve().parents[2]


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the asset path manifest."""
    manifest_path = path or (repo_root() / "assets" / "manifest.yaml")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Asset manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_asset_dir(manifest: dict[str, Any] | None = None) -> Path:
    """Resolve the local download root for assets."""
    manifest = manifest or load_manifest()
    return repo_root() / manifest.get("root_dir", "assets/download")


def _routes(manifest: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Return ``(model, dataset)`` pairs from the manifest."""
    manifest = manifest or load_manifest()
    return [(r["model"], r["dataset"]) for r in manifest.get("routes", [])]


SUPPORTED_ROUTES = _routes()


def assets_for_route(model: str, dataset: str, manifest: dict[str, Any] | None = None) -> list[str]:
    """Return manifest asset keys required for a model–dataset route."""
    manifest = manifest or load_manifest()
    for route in manifest.get("routes", []):
        if route["model"] == model and route["dataset"] == dataset:
            return list(route["assets"])
    raise ValueError(f"Unsupported route {model}/{dataset}. Supported: {SUPPORTED_ROUTES}")


def validate_route(model: str, dataset: str) -> None:
    """Raise if a model–dataset pair is not in the supported matrix."""
    if (model, dataset) not in SUPPORTED_ROUTES:
        raise ValueError(
            f"Unsupported combination model={model!r} dataset={dataset!r}. "
            f"Supported routes: {SUPPORTED_ROUTES}. "
            "Note: FLOWR × CrossDocked is excluded (no pretrained checkpoint)."
        )


def validate_assets(
    model: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Validate local presence of route assets (no checksums)."""
    manifest = load_manifest()
    root = resolve_asset_dir(manifest)
    if model and dataset:
        validate_route(model, dataset)
        keys = assets_for_route(model, dataset, manifest)
    else:
        keys = list(manifest["assets"].keys())

    missing: list[str] = []
    present: list[str] = []
    for key in keys:
        for file_info in manifest["assets"][key]["files"]:
            local_path = root / file_info["local"]
            label = f"{key}:{file_info['local']}"
            if local_path.exists():
                present.append(label)
            else:
                missing.append(label)
    return {
        "ok": not missing,
        "present": present,
        "missing": missing,
        "root": str(root),
    }


def checkpoint_path(model: str, dataset: str) -> Path:
    """Resolve the primary checkpoint path for a route."""
    manifest = load_manifest()
    root = resolve_asset_dir(manifest)
    for key in assets_for_route(model, dataset, manifest):
        entry = manifest["assets"][key]
        if entry.get("kind") == "checkpoint":
            return root / entry["files"][0]["local"]
    raise FileNotFoundError(f"No checkpoint asset for {model}/{dataset}")


def data_path(model: str, dataset: str) -> Path:
    """Resolve the primary dataset directory for a route."""
    manifest = load_manifest()
    root = resolve_asset_dir(manifest)
    for key in assets_for_route(model, dataset, manifest):
        entry = manifest["assets"][key]
        if entry.get("kind") == "dataset":
            return (root / entry["files"][0]["local"]).parent
    raise FileNotFoundError(f"No dataset asset for {model}/{dataset}")
