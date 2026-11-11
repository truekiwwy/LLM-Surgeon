//! Agreement statistics of the judges: Cohen's kappa, a confusion matrix with per-class
//! precision / recall / F1, and McNemar's exact test for paired binary outcomes. Integer
//! counts in, a fixed order of operations: the results do not depend on threads.

use schemars::JsonSchema;
use serde::Serialize;

/// Cohen's kappa of a square table `table[a][b]` (rater A in rows, rater B in columns).
/// `None` when it is undefined: empty table, or both raters put everything in one class.
pub fn cohen_kappa(table: &[Vec<usize>]) -> Option<f64> {
    let n: usize = table.iter().flatten().sum();
    if n == 0 {
        return None;
    }
    let n = n as f64;
    let observed = (0..table.len()).map(|i| table[i][i]).sum::<usize>() as f64 / n;
    let expected: f64 = (0..table.len())
        .map(|i| {
            let row: usize = table[i].iter().sum();
            let column: usize = table.iter().map(|r| r[i]).sum();
            row as f64 * column as f64 / (n * n)
        })
        .sum();
    (expected < 1.0).then(|| (observed - expected) / (1.0 - expected))
}

/// Two-sided exact McNemar test on the discordant pairs `b` and `c`: the binomial
/// probability, under p = 1/2, of a split at least as uneven. 1 when there is no
/// discordant pair.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    // C(n, i) / 2^n, built term by term in log space: 2^-n underflows past n = 1074.
    let mut log_term = -(n as f64) * std::f64::consts::LN_2;
    let mut tail = 0.0;
    for i in 0..=b.min(c) {
        tail += log_term.exp();
        log_term += ((n - i) as f64 / (i + 1) as f64).ln();
    }
    (2.0 * tail).min(1.0)
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ClassMetrics {
    /// Classe.
    pub label: String,
    /// Éléments de cette classe selon la référence humaine.
    pub support: usize,
    /// Vrais positifs / prédits ; null si la classe n'est jamais prédite.
    pub precision: Option<f64>,
    /// Vrais positifs / support ; null si le support est nul.
    pub recall: Option<f64>,
    /// Moyenne harmonique ; null si précision ou rappel est indéfini.
    pub f1: Option<f64>,
}

/// `matrix[truth][predicted]`, both indexed like `labels`.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct Confusion {
    /// Classes, dans l'ordre des lignes et des colonnes.
    pub labels: Vec<String>,
    /// matrix[référence][prédit].
    pub matrix: Vec<Vec<usize>>,
}

impl Confusion {
    pub fn new(labels: &[&str]) -> Self {
        Self {
            labels: labels.iter().map(|l| (*l).to_owned()).collect(),
            matrix: vec![vec![0; labels.len()]; labels.len()],
        }
    }

    fn at(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }

    /// Labels outside `labels` are a caller bug: caught in debug builds, counted nowhere.
    pub fn add(&mut self, truth: &str, predicted: &str) {
        let cell = self.at(truth).zip(self.at(predicted));
        debug_assert!(
            cell.is_some(),
            "classe hors matrice : {truth} / {predicted}"
        );
        if let Some((t, p)) = cell {
            self.matrix[t][p] += 1;
        }
    }

    pub fn total(&self) -> usize {
        self.matrix.iter().flatten().sum()
    }

    pub fn correct(&self) -> usize {
        (0..self.labels.len()).map(|i| self.matrix[i][i]).sum()
    }

    pub fn kappa(&self) -> Option<f64> {
        cohen_kappa(&self.matrix)
    }

    pub fn class(&self, label: &str) -> ClassMetrics {
        // A label outside the matrix has no row: no support, never predicted.
        let i = self.at(label);
        let hit = i.map_or(0, |i| self.matrix[i][i]) as f64;
        let support: usize = i.map_or(0, |i| self.matrix[i].iter().sum());
        let predicted: usize = i.map_or(0, |i| self.matrix.iter().map(|row| row[i]).sum());
        let precision = (predicted > 0).then(|| hit / predicted as f64);
        let recall = (support > 0).then(|| hit / support as f64);
        let f1 = match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            (Some(_), Some(_)) => Some(0.0),
            _ => None,
        };
        ClassMetrics {
            label: label.to_owned(),
            support,
            precision,
            recall,
            f1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn kappa_matches_hand_computations() {
        // po = 35/50 = 0.7 ; pe = (25*30 + 25*20) / 2500 = 0.5 ; kappa = 0.2 / 0.5 = 0.4.
        assert!(close(
            cohen_kappa(&[vec![20, 5], vec![10, 15]]).unwrap(),
            0.4
        ));
        // Perfect agreement and perfect disagreement on balanced margins.
        assert!(close(cohen_kappa(&[vec![7, 0], vec![0, 3]]).unwrap(), 1.0));
        assert!(close(cohen_kappa(&[vec![0, 5], vec![5, 0]]).unwrap(), -1.0));
        // Three classes: po = 6/10 ; pe = (4*3 + 3*3 + 3*4) / 100 = 0.33 ; kappa = 0.27/0.67.
        let three = [vec![2, 1, 1], vec![0, 2, 1], vec![1, 0, 2]];
        assert!(close(cohen_kappa(&three).unwrap(), 0.27 / 0.67));
        // Undefined: nothing rated, or one class only on both sides.
        assert_eq!(cohen_kappa(&[vec![0, 0], vec![0, 0]]), None);
        assert_eq!(cohen_kappa(&[vec![9, 0], vec![0, 0]]), None);
    }

    #[test]
    fn mcnemar_exact_matches_hand_computations() {
        // n = 6, min = 1: 2 * (C(6,0) + C(6,1)) / 64 = 14/64.
        assert!(close(mcnemar_exact(1, 5), 14.0 / 64.0));
        assert!(close(mcnemar_exact(5, 1), 14.0 / 64.0));
        // n = 10, min = 0: 2 / 1024.
        assert!(close(mcnemar_exact(0, 10), 2.0 / 1024.0));
        // Even split: the two tails overlap, capped at 1.
        assert_eq!(mcnemar_exact(3, 3), 1.0);
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        // n = 9, min = 2: 2 * (1 + 9 + 36) / 512.
        assert!(close(mcnemar_exact(2, 7), 92.0 / 512.0));
        // More discordant pairs than 2^-n can hold: an even split is still p = 1.
        assert_eq!(mcnemar_exact(600, 600), 1.0);
    }

    #[test]
    fn confusion_matrix_precision_recall_f1_by_hand() {
        let mut c = Confusion::new(&["a", "b", "c"]);
        for (truth, predicted, times) in [
            ("a", "a", 4),
            ("a", "b", 1),
            ("b", "b", 2),
            ("b", "a", 2),
            ("b", "c", 1),
        ] {
            for _ in 0..times {
                c.add(truth, predicted);
            }
        }
        assert_eq!(c.matrix, [vec![4, 1, 0], vec![2, 2, 1], vec![0, 0, 0]]);
        assert_eq!((c.total(), c.correct()), (10, 6));
        // a: precision 4/6, recall 4/5, F1 = 2 * (2/3) * (4/5) / (2/3 + 4/5) = 8/11.
        let a = c.class("a");
        assert_eq!(a.support, 5);
        assert!(close(a.precision.unwrap(), 4.0 / 6.0));
        assert!(close(a.recall.unwrap(), 0.8));
        assert!(close(a.f1.unwrap(), 8.0 / 11.0));
        // b: precision 2/3, recall 2/5, F1 = 0.5.
        let b = c.class("b");
        assert!(close(b.precision.unwrap(), 2.0 / 3.0));
        assert!(close(b.recall.unwrap(), 0.4));
        assert!(close(b.f1.unwrap(), 0.5));
        // c: predicted once and always wrong, never in the truth.
        let cc = c.class("c");
        assert_eq!(
            (cc.support, cc.precision, cc.recall, cc.f1),
            (0, Some(0.0), None, None)
        );
        // po = 0.6 ; pe = (5*6 + 5*3 + 0*1) / 100 = 0.45 ; kappa = 0.15 / 0.55.
        assert!(close(c.kappa().unwrap(), 0.15 / 0.55));
    }
}
