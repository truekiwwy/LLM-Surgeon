//! Read-only cosine comparison of two direction files, before composing them.
//!
//! Two interventions applied on the same layers can cancel or reinforce each other; the
//! radar compares the direction tensors of two `.safetensors` files of `work/` layer by
//! layer and flags the ones that collide. Nothing is written and no model is opened: the
//! files are direction vectors, decoded by the engine (F32, BF16, F16) and compared in f64.

use std::path::{Path, PathBuf};

use weight_surgeon::engine::{load_direction_vector, ReadOnlyModel};

/// Below this absolute cosine, two directions are independent.
pub const SAFE_BELOW: f64 = 0.15;
/// Above this absolute cosine, the two interventions collide on the layer.
pub const STRONG_ABOVE: f64 = 0.35;

/// How much two directions of one layer overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    /// `|cos| < 0.15`: composing both is safe.
    Orthogonal,
    /// `0.15 <= |cos| <= 0.35`: the two interventions partially overlap.
    Moderate,
    /// `|cos| > 0.35`: the two interventions collide.
    Strong,
}

impl Risk {
    pub fn of(cosine: f64) -> Self {
        let overlap = cosine.abs();
        if overlap < SAFE_BELOW {
            Self::Orthogonal
        } else if overlap <= STRONG_ABOVE {
            Self::Moderate
        } else {
            Self::Strong
        }
    }
}

/// One tensor present in both files.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRow {
    pub tensor: String,
    /// The layer named by the tensor (`layers.14` -> `14`), else the `layer` the extraction
    /// wrote in the metadata of file A.
    pub layer: Option<usize>,
    /// `dot(a, b) / (|a| |b|)`, in `[-1, 1]`; `0.0` when a norm is zero.
    pub cosine: f64,
}

impl LayerRow {
    pub fn risk(&self) -> Risk {
        Risk::of(self.cosine)
    }
}

/// The comparison of two files, rows sorted by layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarReport {
    pub rows: Vec<LayerRow>,
    /// The tensor with the largest `|cos|`; `None` when the files share no tensor.
    pub strongest: Option<LayerRow>,
    /// Mean of `|cos|` over the common tensors, `0.0` when there are none.
    pub mean_abs: f64,
    /// At least one layer reaches [`SAFE_BELOW`]: `orthogonal-preserve` is advised.
    pub collision: bool,
}

/// The `.safetensors` files of `work/`, sorted by file name.
pub fn work_files(work: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(work) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        })
        .collect();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    files
}

/// Compares every 1D tensor common to `a` and `b`; a tensor only in one file is ignored.
pub fn compare_direction_files(a: &Path, b: &Path) -> Result<RadarReport, String> {
    let (left, file_layer) = load(a)?;
    let (right, _) = load(b)?;
    let mut rows = Vec::new();
    for (name, values) in &left {
        let Some(other) = right
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, values)| values)
        else {
            continue;
        };
        if values.len() != other.len() {
            return Err(format!(
                "{name}: {} values against {}",
                values.len(),
                other.len()
            ));
        }
        rows.push(LayerRow {
            tensor: name.clone(),
            layer: layer_of(name).or(file_layer),
            cosine: cosine(values, other),
        });
    }
    rows.sort_by(|a, b| {
        a.layer
            .unwrap_or(usize::MAX)
            .cmp(&b.layer.unwrap_or(usize::MAX))
            .then_with(|| a.tensor.cmp(&b.tensor))
    });
    let mean_abs = match rows.is_empty() {
        true => 0.0,
        false => rows.iter().map(|row| row.cosine.abs()).sum::<f64>() / rows.len() as f64,
    };
    let strongest = rows
        .iter()
        .max_by(|a, b| a.cosine.abs().total_cmp(&b.cosine.abs()))
        .cloned();
    let collision = rows.iter().any(|row| row.cosine.abs() >= SAFE_BELOW);
    Ok(RadarReport {
        rows,
        strongest,
        mean_abs,
        collision,
    })
}

/// The 1D tensors of a direction file in f64, in file order, and the `layer` of its
/// metadata. A 2D tensor (a subspace) has no single direction to compare: it is skipped.
type Loaded = (Vec<(String, Vec<f64>)>, Option<usize>);

fn load(path: &Path) -> Result<Loaded, String> {
    let error = |e: weight_surgeon::engine::EngineError| format!("{}: {e}", path.display());
    let model = ReadOnlyModel::open(path).map_err(error)?;
    let mut tensors = Vec::new();
    for info in model.index().tensors.iter().filter(|i| i.shape.len() == 1) {
        let values = load_direction_vector(path, Some(&info.name)).map_err(error)?;
        tensors.push((
            info.name.clone(),
            values.into_iter().map(f64::from).collect(),
        ));
    }
    let layer = model.metadata().get("layer").and_then(|l| l.parse().ok());
    Ok((tensors, layer))
}

/// `dot(a, b) / (|a| |b|)`, `0.0` when either norm is zero.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    match norm_a > 0.0 && norm_b > 0.0 {
        true => dot / (norm_a * norm_b),
        false => 0.0,
    }
}

/// The layer of a tensor name (`model.layers.14.mlp...` -> `14`).
fn layer_of(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    for marker in ["layers.", "layer.", "layers_", "layer_"] {
        let Some((_, rest)) = lower.split_once(marker) else {
            continue;
        };
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(layer) = digits.parse() {
            return Some(layer);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use weight_surgeon::extract::encode_tensors;

    fn write(dir: &Path, name: &str, tensors: &[(&str, &[f32])]) -> PathBuf {
        let bytes = encode_tensors(tensors, &BTreeMap::new()).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn orthogonal_identical_and_opposite_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let x = write(dir.path(), "x.safetensors", &[("layers.0", &[1.0, 0.0])]);
        let y = write(dir.path(), "y.safetensors", &[("layers.0", &[0.0, 1.0])]);
        close(compare_direction_files(&x, &y).unwrap().rows[0].cosine, 0.0);
        close(compare_direction_files(&x, &x).unwrap().rows[0].cosine, 1.0);
        let opposite = write(
            dir.path(),
            "opposite.safetensors",
            &[("layers.0", &[-1.0, 0.0])],
        );
        close(
            compare_direction_files(&x, &opposite).unwrap().rows[0].cosine,
            -1.0,
        );
    }

    #[test]
    fn zero_norms_read_as_orthogonal_and_incompatible_lengths_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let zero = write(dir.path(), "zero.safetensors", &[("layers.0", &[0.0, 0.0])]);
        let x = write(dir.path(), "x.safetensors", &[("layers.0", &[1.0, 0.0])]);
        close(
            compare_direction_files(&zero, &x).unwrap().rows[0].cosine,
            0.0,
        );
        let longer = write(
            dir.path(),
            "longer.safetensors",
            &[("layers.0", &[1.0, 0.0, 0.0])],
        );
        assert!(compare_direction_files(&x, &longer).is_err());
    }

    #[test]
    fn only_common_tensors_are_compared_and_rows_follow_the_layer_order() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(
            dir.path(),
            "a.safetensors",
            &[("layers.10", &[1.0, 0.0]), ("layers.2", &[0.0, 1.0])],
        );
        let b = write(
            dir.path(),
            "b.safetensors",
            &[
                ("layers.10", &[1.0, 0.0]),
                ("layers.2", &[1.0, 0.0]),
                ("layers.3", &[1.0, 0.0]),
            ],
        );
        let report = compare_direction_files(&a, &b).unwrap();
        let layers: Vec<_> = report.rows.iter().map(|row| row.layer).collect();
        assert_eq!(layers, [Some(2), Some(10)]);
        assert!(report.collision);
        assert_eq!(report.strongest.unwrap().tensor, "layers.10");
    }

    #[test]
    fn a_file_without_common_tensor_is_an_empty_report() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.safetensors", &[("layers.0", &[1.0])]);
        let b = write(dir.path(), "b.safetensors", &[("layers.1", &[1.0])]);
        let report = compare_direction_files(&a, &b).unwrap();
        assert!(report.rows.is_empty());
        assert!(report.strongest.is_none());
        close(report.mean_abs, 0.0);
        assert!(!report.collision);
    }

    /// An extraction's file: one `direction` tensor, its layer in the metadata only.
    #[test]
    fn a_bf16_extraction_file_is_read_with_the_layer_of_its_metadata() {
        let dir = tempfile::tempdir().unwrap();
        // BF16 1.0 and -0.5, little endian.
        let data = [0x80, 0x3f, 0x00, 0xbf];
        let mut header = serde_json::json!({
            "__metadata__": {"layer": "14"},
            "direction": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        })
        .to_string();
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        let bf16 = dir.path().join("refusal_L14.safetensors");
        std::fs::write(&bf16, bytes).unwrap();
        let f32 = write(
            dir.path(),
            "f32.safetensors",
            &[("direction", &[2.0, -1.0])],
        );
        let report = compare_direction_files(&bf16, &f32).unwrap();
        assert_eq!(report.rows[0].layer, Some(14));
        close(report.rows[0].cosine, 1.0);
        assert_eq!(report.rows[0].risk(), Risk::Strong);
    }

    #[test]
    fn risks_follow_the_absolute_cosine_bands() {
        assert_eq!(Risk::of(0.14), Risk::Orthogonal);
        assert_eq!(Risk::of(-0.14), Risk::Orthogonal);
        assert_eq!(Risk::of(0.15), Risk::Moderate);
        assert_eq!(Risk::of(0.35), Risk::Moderate);
        assert_eq!(Risk::of(-0.36), Risk::Strong);
    }
}
