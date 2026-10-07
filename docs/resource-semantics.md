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
| `filter_resource(resource, var_names)` | `resource::filter_resource(subunits, var_names)` | `filter_ref/synthetic.json` `n_exploded` |
| `prep_check_adata` + `_get_props` + `_filter_reassemble_complexes` | `resource::filter_lrs(adata, resource, expr_prop, min_cells)` | `filter_ref/synthetic.json` `sha256_kept` |

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

## Filtering: `expr_prop` and `min_cells` (W2-T4)

`liana_core::resource::filter_lrs(adata, resource, expr_prop, min_cells)` is the
part of `_liana_pipe` between the resource and the matrix: it returns the
`(source, target, ligand_complex, receptor_complex)` keys that pass `expr_prop`,
in liana's row order, with the `prop_min` each passed on.

Ground truth: `testdata/filter_ref/synthetic.json`, dumped by
`scripts/dump_filter_ref.py` (the five cases below), verified by
`crates/liana-core/tests/filter_parity.rs` — each `prop_min` is hashed as the
float64 bit pattern liana compared, so the arithmetic is pinned, not the digits.

**`min_cells`** is a *cell* threshold, not a gene one: clusters with fewer than
`min_cells` cells lose every cell, and the cluster leaves the pair enumeration
with them — liana re-derives the categories after the drop, so `min_cells=2100`
on `synthetic.h5ad` leaves `["B"]` alone and the result covers `(B, B)` only.
`min_cells=0` keeps everything. (liana's `min_cells=None` requires its `groupby`
to be unset and is not reachable here.)

**A pair is "expressed" in a cluster pair when every subunit of both sides is**
— that is what the `min` policy of `_filter_reassemble_complexes` means: the
key's `prop_min` is the minimum over all its exploded subunit rows of

- the ligand subunit's proportion in the *source* cluster, and
- the receptor subunit's proportion in the *target* cluster,

and the key is kept when `prop_min >= expr_prop`. A proportion is
`X.getnnz(axis=0) / n_cells` — the fraction of the cluster's cells with a
*stored* entry for the gene, so an explicitly stored zero counts as expressed,
exactly as in the CSR liana builds. `filter_lrs` returns an error when no pair
passes and the candidate list is non-empty, which is where liana raises a
`ValueError`.

Two `prep_check_adata` rules shape the candidates before the proportions matter,
and both are in `filter_lrs`:

- **features that sum to zero** are dropped from the matrix, so the resource
  rows that need them are dropped too — independently of their stored-entry
  count;
- `filter_resource` (`resource::filter_resource`) drops subunit rows whose gene
  the matrix does not carry, and drops a complex pair *as a whole* when any
  subunit of either of its complexes is missing — so a row whose own two genes
  are present still goes if the complex it belongs to has a missing subunit.

Row order is `_get_lr`'s: the cluster pairs come from `np.meshgrid(labels,
labels)` — source varying fastest, target slowest — crossed with the
`filter_resource`-surviving resource rows, and the kept list keeps the first
occurrence of each key.

## What the resource layer does not cover

The method scores (`ligand_means`, `ligand_pvals`, …) and the per-method columns
`_reduce_complexes` re-assembles complexes over: `filter_lrs` stops at the key
level, which is the granularity `expr_prop` decides on. `assert_covered` (the
98 % resource-coverage guard) and `complex_policy` other than `"min"` are not
implemented.
