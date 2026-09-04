# Third-party notices

## VERL

The training integration is based on VERL (Volcano Engine Reinforcement Learning for LLM), which is Apache-2.0 licensed. This release redistributes the complete modified VERL checkout under `verl/`, including the integrated `nanoclaw_recipe`, agent-loop changes, trainer changes, and model support. Preserve the upstream copyright headers and consult `verl/LICENSE` and `verl/Notice.txt` for the framework's license and notices.

Upstream project: <https://github.com/volcengine/verl>

## AAAI author kit

`paper/aaai2027.bib`, the AAAI style files used to compile the manuscript, and the paper PDF originate from the AAAI 2027 author kit and conference submission materials. They are included for scholarly reproducibility and remain subject to the terms accompanying that kit. The camera-ready manuscript should be checked against the conference's current publication policy before redistribution.

## Source benchmark export

The task records were exported from an internal Nanoclaw/context benchmark pipeline. The release contains only records marked valid by that exporter. No upstream benchmark license file was present in the source directory; maintainers should confirm that the source benchmark permits public redistribution before publishing a Hub dataset. Synthetic names and workplace scenarios should not be interpreted as real personal records.
