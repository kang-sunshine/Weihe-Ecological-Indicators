# Weihe ecological indicators
Event-scale ecological indicators reveal severity-dependent functional impairment and hydroclimatic constraints on greening in the Weihe River Basin. Ecological Indicators, ECOLIND-64521.

This source collection covers climate extremes, functional impairment, observed legacy, post-event recovery and associations between long-term greening and severity-adjusted CDH damage. The numbered directories follow the analysis stages. The implementation index links scripts to the manuscript.

Use Python 3.10 or later with the scientific and geospatial packages in requirements.txt. Use a separate writable working directory containing user-prepared inputs. Install the requirements, then stage the scripts from the repository root with:

```sh
python config/workspace.py /path/to/analysis-workspace
```

The workspace argument sets the common input/output root. The layout file preserves cross-stage script locations. To execute an individual Python stage, append `--run` and its public script path listed in config/layout.json. Archived programs can execute their analysis and write outputs at module level. Review their input declarations and supply prerequisites before running them.

See [Workflow](docs/WORKFLOW.md), [Input requirements](docs/DATA_REQUIREMENTS.md) and [Manuscript mapping](docs/CODE_MANUSCRIPT_MAP.md). Obtain licensed products through their respective providers and supply the basin geometry and compatible archived cubes or derived inputs. This code release contains no observational or derived datasets. The historical L2 producer and the exact final rendering provenance for some figures remain unavailable. Full end-to-end execution has not been validated.
