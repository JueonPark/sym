# Historical evidence catalog

[historical-manifest.json](historical-manifest.json) inventories original E1–E5
and follow-ups #263–272. Each campaign lists hardware, revisions, build provenance,
settings, dependencies, protocol, raw evidence, summaries, replay commands,
and exclusions. Paths in Wiki campaigns are relative to the named extracted
directory. Paths in repository campaigns are relative to the Sym checkout.
Braces and wildcards identify published sets of files. The repository snapshot
is the commit at which the cataloged files were pinned. Measurement, measured
runtime, build-control, and publication revisions can differ.

| Campaign | Evidence and reproduction |
| --- | --- |
| E1–E5 | [Original versioned package](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #263 critical path | [Profile and matched controls](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-263/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #264 prepared groups | [Fixed-report ablations and model cases](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #265 automatic preparation | [E3 and held-out requests](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-265/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #266 memory and lifetimes | [Physical pressure, placement and burst results](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-266/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #267 kernel specialization | [Materialized, folded, manual and Torch controls](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-267/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #268 CUDA Graph replay | [Unwrapped and matched slot controls](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-268/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #269 consumer layout | [Native, transparent and lowered paths](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-269/ae44b557b836abf02aeca2780ca4b1a4c3561c3b) |
| #270 relocation semantics | [Region rules, published summary and commands](../../docs/relocation-regions.md#same-engine-ablation) |
| #271 joint planner | [Results and raw-data replay](../issue271/README.md#reproduction) |
| #272 automatic specialization and reuse | [E3 controls and captured-artifact results](../issue272/README.md) |

## Verify and download

Run from the checkout root. Python 3.10 or later and its standard library
suffice. The first command verifies local evidence without fetching anything.
The second adds the original archive check once that archive is present.
Omit `--archive` to verify all eight assembled archives in the supplied directory,
or repeat that option to select several by ID.

```sh
python3 -m bench.issue274.verify_catalog
python3 -m bench.issue274.verify_catalog --archive-dir /path/to/downloads --archive original
```

Each `archives` record links the revision-pinned Wiki page, the official raw
download manifest, and the published checksum lists. The pin identifies the
Wiki Git revision. Download manifests and checksum lists are addressed by
versioned paths and checked by their recorded SHA-256. Use the official Wiki
helper for downloads. This example fetches the original archive without
authentication. Use a fresh directory for the downloads.

```sh
downloads=/tmp/sym-historical-downloads
mkdir -p "$downloads"
curl -fL https://raw.githubusercontent.com/wiki/JueonPark/sym/scripts/fetch_artifact.py \
  -o "$downloads/fetch_artifact.py"
printf '%s  %s\n' \
  f8475b9b97513a113037a95a9d508787e74302c93a23fb9489ad735f4c7aabe2 \
  "$downloads/fetch_artifact.py" | sha256sum -c -
python3 "$downloads/fetch_artifact.py" \
  https://raw.githubusercontent.com/wiki/JueonPark/sym/artifacts/experiments-2026-11-04-v1/manifest.json \
  --output "$downloads"
python3 -m bench.issue274.verify_catalog --archive-dir "$downloads" --archive original
```

The offline verifier checks length and SHA-256 for each ordered split at its
original byte boundary, then the assembled archive. It also verifies the
3,113-byte `sym-benefit-assessment.md` member as stored inside the original archive
(SHA-256 `10f2faa59326ad0d4e92fe2a76a3c57c45ba468245a0fc97d18c6a504fe3cf11`).
The published packages provide file inventories and their original
verification and analysis commands. Reanalyze a working copy because some
analyzers write summaries alongside inputs. Use the campaign's stated Python
version for byte-identical outputs. Plotting can require extra dependencies.
For #272, extract the two raw archives into separate directories and pass
each extracted directory to its scorer. The captured-artifact and automatic
endpoint timing envelopes differ.

## Limits of the historical record

- #270 published its legality protocol, fixtures, summary counts and runner.
  Its historical stdout and exact binary/affinity record are absent from the
  public files. A new untimed execution under [issue #274](https://github.com/JueonPark/sym/issues/274)
  has its own provenance under `bench/issue274/region-replay/`.
- #271 publishes the expanded measured sweep and its held-out workload. The
  earlier 49-plan exploratory sweep is described in the protocol. The
  historical E2/E4/E5 comparison uses whole recorded endpoints.
- #263 omitted unrelated workstation/session metadata from pilot Nsight
  records and provides the redaction receipt. Timed accepted records retain
  their original bytes. Other exclusions and OOM outcomes are listed by
  campaign and preserved where distributed.
- Replaying saved data verifies the recorded experiments. Running on a
  different machine or rebuilding a dependency creates a new campaign
  with new provenance.
