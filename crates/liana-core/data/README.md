# Vendored resources

## `omni_resource.csv`

Verbatim copy of `liana/resource/omni_resource.csv` from liana **2.0.0**
(commit `c59472ccc9de8360dbbf5016db75f8abde08dd3e`):

```
sha256  65e46a428818c0600de66c1574c94ff2f5dc80859998321c03a956d330dd67fb
size    2 004 367 bytes · 35 356 data rows + header · 17 resources
```

Regenerate with the oracle venv (must report the pinned version/commit, which
`scripts/dump_resource_ref.py` enforces):

```bash
cp /home/pwwang/p0a/venv/lib/python3.12/site-packages/liana/resource/omni_resource.csv \
   crates/liana-core/data/omni_resource.csv
sha256sum crates/liana-core/data/omni_resource.csv   # must match the hash above
```

`tests/resource_parity.rs::csv_identity` pins the hash, so a drifted or
re-vendored file fails the suite rather than silently changing every selection.

Columns: `""` (a 1-based row index pandas drops into `Unnamed: 0`), `source`,
`target` (uniprot ids), `source_genesymbol`, `target_genesymbol` (what
`select` returns, as `ligand`/`receptor`), `resource`. All fields are quoted;
no field contains a quote, escape, or embedded comma.
