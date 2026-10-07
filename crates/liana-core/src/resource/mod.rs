//! The ligand–receptor resource layer, mirroring `liana.resource`.
//!
//! [`select`] reads the vendored `omni_resource.csv` the way liana's
//! `select_resource` does; [`explode_complexes`] mirrors `_explode_complexes`
//! (protein complexes are `_`-joined subunit symbols). See
//! `docs/resource-semantics.md`.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// The pinned liana resource, vendored verbatim (`data/README.md`).
const OMNI_RESOURCE: &str = include_str!("../../data/omni_resource.csv");

/// A ligand–receptor pair exactly as the resource lists it: either side may be
/// a protein complex (`ITGA4_ITGB7`), not a single gene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrPair {
    pub ligand: String,
    pub receptor: String,
}

/// One subunit pairing of an [`LrPair`], i.e. a row of liana's exploded
/// resource: `ligand`/`receptor` are the individual subunits that get looked up
/// in the expression matrix, `*_complex` the resource's original symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrSubunit {
    pub ligand: String,
    pub receptor: String,
    pub ligand_complex: String,
    pub receptor_complex: String,
}

/// The resource names `select` accepts, in CSV order of first appearance
/// (liana's `show_resources`).
pub fn names() -> Result<Vec<String>> {
    let mut names: Vec<String> = Vec::new();
    for (name, _) in omni_rows()? {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    Ok(names)
}

/// The pairs of one `omni_resource.csv` resource, in file order: liana's
/// `select_resource`. The name is matched case-insensitively against the CSV's
/// `resource` column; there is no deduplication or sorting.
pub fn select(name: &str) -> Result<Vec<LrPair>> {
    let wanted = name.to_lowercase();
    let rows = omni_rows()?;
    let pairs: Vec<LrPair> = rows
        .iter()
        .filter(|(resource, _)| resource == &wanted)
        .map(|(_, pair)| pair.clone())
        .collect();
    if pairs.is_empty() {
        let mut names: Vec<&str> = Vec::new();
        for (resource, _) in &rows {
            if !names.contains(&resource.as_str()) {
                names.push(resource);
            }
        }
        bail!("resource {name:?} not found; choose from {names:?}");
    }
    Ok(pairs)
}

/// Read a `ligand,receptor` CSV (the toy resources in `testdata/`), in file order.
pub fn read_pairs(path: &Path) -> Result<Vec<LrPair>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_pairs(&text)
}

fn parse_pairs(text: &str) -> Result<Vec<LrPair>> {
    let mut lines = text.lines();
    let header = split_csv(lines.next().context("empty resource csv")?);
    let ligand = column_index(&header, "ligand")?;
    let receptor = column_index(&header, "receptor")?;
    lines
        .filter(|line| !line.is_empty())
        .map(|line| pair(&split_csv(line), ligand, receptor))
        .collect()
}

/// Split complexes into their subunits, in liana's `_explode_complexes` order:
/// one row per (receptor subunit, ligand subunit) with the receptor varying
/// slowest — the order its `explode` calls produce. Non-complex symbols pass
/// through as a single subunit equal to the symbol itself.
pub fn explode_complexes(pairs: &[LrPair]) -> Vec<LrSubunit> {
    let mut subunits = Vec::new();
    for pair in pairs {
        let receptors: Vec<&str> = pair.receptor.split('_').collect();
        let ligands: Vec<&str> = pair.ligand.split('_').collect();
        for receptor in &receptors {
            for ligand in &ligands {
                subunits.push(LrSubunit {
                    ligand: (*ligand).to_owned(),
                    receptor: (*receptor).to_owned(),
                    ligand_complex: pair.ligand.clone(),
                    receptor_complex: pair.receptor.clone(),
                });
            }
        }
    }
    subunits
}

/// Every row of the vendored CSV as `(resource, pair)`, in file order.
fn omni_rows() -> Result<Vec<(String, LrPair)>> {
    let mut lines = OMNI_RESOURCE.lines();
    let header = split_csv(lines.next().context("omni_resource.csv is empty")?);
    let source = column_index(&header, "source_genesymbol")?;
    let target = column_index(&header, "target_genesymbol")?;
    let resource = column_index(&header, "resource")?;
    lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields = split_csv(line);
            let name = field(&fields, resource)?.to_owned();
            Ok((name, pair(&fields, source, target)?))
        })
        .collect()
}

fn column_index(header: &[&str], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|column| *column == name)
        .with_context(|| format!("no {name:?} column in {header:?}"))
}

fn pair(fields: &[&str], ligand: usize, receptor: usize) -> Result<LrPair> {
    Ok(LrPair {
        ligand: field(fields, ligand)?.to_owned(),
        receptor: field(fields, receptor)?.to_owned(),
    })
}

fn field<'a>(fields: &[&'a str], index: usize) -> Result<&'a str> {
    fields
        .get(index)
        .copied()
        .context("row is shorter than its header")
}

/// Split one CSV line into fields, unquoting `"`-quoted ones and keeping
/// quoted commas. The shipped resources have no escapes or embedded quotes.
fn split_csv(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(unquote(&line[start..i]));
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(unquote(&line[start..]));
    fields
}

fn unquote(field: &str) -> &str {
    field
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(field)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_csv_unquotes_and_keeps_quoted_commas() {
        assert_eq!(
            split_csv(r#""1","A","B","C","D","consensus""#),
            ["1", "A", "B", "C", "D", "consensus"]
        );
        assert_eq!(split_csv("ligand,receptor"), ["ligand", "receptor"]);
        assert_eq!(split_csv(r#""a,b",c"#), ["a,b", "c"]);
        assert_eq!(split_csv(""), [""]);
    }

    #[test]
    fn select_is_case_insensitive_and_orders_by_file() {
        let pairs = select("CONSENSUS").unwrap();
        assert_eq!(pairs.len(), 4620);
        assert_eq!(pairs[0].ligand, "LGALS9");
        assert_eq!(pairs[0].receptor, "PTPRC");
        assert!(select("nope").is_err());
    }

    #[test]
    fn explode_variants() {
        let pairs = vec![
            LrPair {
                ligand: "A_B".into(),
                receptor: "C_D".into(),
            },
            LrPair {
                ligand: "A".into(),
                receptor: "A_B".into(),
            },
        ];
        let exploded = explode_complexes(&pairs);
        let rows: Vec<(&str, &str)> = exploded
            .iter()
            .map(|s| (s.ligand.as_str(), s.receptor.as_str()))
            .collect();
        // receptors vary slowest, and a non-complex ligand is its own subunit
        assert_eq!(
            rows,
            [
                ("A", "C"),
                ("B", "C"),
                ("A", "D"),
                ("B", "D"),
                ("A", "A"),
                ("A", "B")
            ]
        );
        assert_eq!(exploded[0].ligand_complex, "A_B");
        assert_eq!(exploded[0].receptor_complex, "C_D");
        assert_eq!(exploded[5].ligand_complex, "A");
        assert_eq!(exploded[5].receptor_complex, "A_B");
    }

    #[test]
    fn parse_pairs_reads_a_two_column_resource() {
        let pairs = parse_pairs("ligand,receptor\nECM,ligA\nECM,ligB\n").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[1].ligand, "ECM");
        assert_eq!(pairs[1].receptor, "ligB");
    }
}
