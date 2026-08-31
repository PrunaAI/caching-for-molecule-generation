# Third-Party Notices

This repository redistributes and adapts several upstream research codebases.
Original licenses apply to those components. New orchestration, caching
integration, and CLI code under `geo_cache/` is licensed under Apache-2.0
(see `LICENSE`).

## Bundled components

| Component | Path | Upstream | Commit | License |
|-----------|------|----------|--------|---------|
| SemlaFlow | `models/semlaflow/` | https://github.com/rssrwn/semla-flow | `466524581e841abb4271d64f4a256513a040bac8` | MIT (`models/semlaflow/LICENSE`) |
| Tabasco | `models/tabasco/` | https://github.com/carlosinator/tabasco | `097eaae106e90a88294a93d42a3f91f1798f383f` | MIT (`models/tabasco/LICENSE`) |
| FLOWR | `models/flowr/` | https://github.com/jule-c/flowr | `bbb8bfb4e8d8dd16bff1d56a7f96246920e643ba` | MIT (README; license file added for redistribution clarity) |
| FLOWR.root | `models/flowr_root/` | https://github.com/jule-c/flowr_root | `066b3c878b4693d156e5445a0f175e49a5cae3de` | MIT (README; license file added for redistribution clarity) |

See `assets/provenance.yaml` for the machine-readable provenance record.

## External dependencies (not redistributed)

- **Pruna** (`pruna>=0.3.4`): Apache-2.0. Required for caching algorithms.
- **PoseBusters** (`posebusters==0.3.1`) and **PoseCheck** (`posecheck==1.3.1`): installed from PyPI via `environment.yml`. FLOWR vendored copies of both previously shadowed these on `sys.path`
- **GenBench3D / AutoDock Vina / ADFR / xTB**: evaluation-only tooling
