# Resource semantics (W2-T3)

What `liana_core::resource` replicates: how liana 2.0.0 turns
`omni_resource.csv` into ligand–receptor pairs and how it expands protein
complexes. Ground truth: the pinned oracle (liana 2.0.0 @ `c59472cc`), dumped
into `testdata/resource_ref/*.json` by `scripts/dump_resource_ref.py` and
verified by `crates/liana-core/tests/resource_parity.rs`.

| liana | liana-rs | reference |
|---|---|---|
| `select_resource(name)` | `resource::select(name)` | `resources.json` (17 resources), `consensus.json` |
| `show_resources()` | `resource::names()` | `resources.json` keys |
| `_explode_complexes(resource)` | `resource::explode_complexes(pairs)` | `consensus.json` `sha256_exploded` |

The CSV itself is vendored at `crates/liana-core/data/omni_resource.csv`
(sha256 pinned in `data/README.md` and in the test).

## `select`: name → pairs

`select_resource` is four lines of pandas and the port keeps all four
behaviours:

1. the name is **lower-cased** before matching (`select("CONSENSUS")` works);
2. rows are filtered on the `resource` column — a name absent from it raises
   (`ValueError` there, `anyhow` error here);
3. only `source_genesymbol` → `ligand` and `target_genesymbol` → `receptor`
   survive; the uniprot `source`/`target` columns are dropped;
4. **row order is the CSV's order** — no sorting, and no deduplication:
   consensus has 4620 pairs with repeated `(ligand, receptor)` rows, and every
   one is kept (the parity test hashes the stream in order).

Nothing drops NA rows: that only happens in liana's `_handle_resource` for a
resource *DataFrame passed by the user*, which is a different entry point.

## Complexes

A symbol containing `_` is a protein complex whose subunits are the
`_`-separated parts, e.g. `ITGA4_ITGB7`, `FRAS1_FREM1_NPNT`. `select` returns
these symbols verbatim; `explode_complexes` is what turns them into the
per-subunit lookups the expression matrix can be indexed by.

`_explode_complexes` is a pandas one-liner — `set_index("interaction")`
where `interaction = ligand + "&" + receptor`, then
`.explode(TARGET).explode(SOURCE)` — and the port reproduces its row order
exactly:

- one output row per **(receptor subunit, ligand subunit)** combination;
- the **receptor subunit varies slowest, the ligand subunit fastest** — the
  order produced by exploding the target column before the source column.
  For `FRAS1_FREM1_NPNT & ITGA8_ITGB1` that is
  `(FRAS1, ITGA8), (FREM1, ITGA8), (NPNT, ITGA8), (FRAS1, ITGB1), …`;
- a non-complex symbol splits into a single subunit equal to itself, so
  plain pairs pass through with `ligand == ligand_complex`;
- `ligand_complex`/`receptor_complex` carry the original symbols
  (`LrSubunit` mirrors those four column names).

Scale check on consensus: 4620 pairs → **5776** exploded subunit rows, from
**15** distinct ligand complexes and **167** distinct receptor complexes
(`consensus.json` pins the full sorted lists and the exploded stream hash).

## What the resource layer does not cover

`expr_prop`/`min_cells` filtering, complex re-assembly (`_reduce_complexes`,
`_filter_reassemble_complexes`) and the `complex_policy` are method-level
semantics: they need the expression matrix and the labels, and they are
implemented in `liana_core::resource::filter` (W2-T4, with the details of the
min-proportion rule documented there).
