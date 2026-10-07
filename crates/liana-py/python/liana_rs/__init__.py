"""liana-rs — LIANA's ligand–receptor methods in Rust, bit-exact.

`run` returns a pandas `DataFrame` when pandas is importable, else the
columnar `dict` `{column: [values]}` the Rust module produced.
"""

from ._core import run as _run

__all__ = ["run"]


def run(h5ad, label_key, resource, method, n_perms=1000, seed=1337, threads=0):
    """Run one method over one `.h5ad`; `resource` is a name (`consensus`) or a
    `ligand,receptor` CSV path, `n_perms`/`seed`/`threads` as in `liana-rs run`."""
    result = _run(h5ad, label_key, resource, method, n_perms, seed, threads)
    try:
        import pandas
    except ImportError:
        return result
    return pandas.DataFrame(result)
