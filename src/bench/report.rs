//! The five reports (the `data` of the JSON envelope, CLI and MCP alike) and their
//! terminal rendering. Every float that can be undefined is an `Option` (`null` in JSON):
//! no NaN or infinity ever reaches the JSON. Durations live in `timing_ms` only, so two
//! runs on the same inputs serialize identically once `timing_ms` is removed.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::generation::{GenerationReport, SampledScore};
use super::judge::{self, JudgeReport};
use super::logits_io::{DumpFormat, LogitsDtype};
use crate::ui::{self, Ui};

/// Threshold verdict: `none` when no threshold was given (none is invented), `ok` when
/// every given threshold holds, `warn` when one is strictly exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    None,
    Ok,
    Warn,
}

/// A run's report with its warnings. `threshold_warnings` (also in `warnings`) are the
/// only ones `--fail-on-warn` turns into exit code 9.
#[derive(Debug, Clone)]
pub struct BenchRun<R> {
    pub report: R,
    pub warnings: Vec<String>,
    pub threshold_warnings: Vec<String>,
}

// ------------------------------------------------------------------ logits

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct DumpInfo {
    /// Chemin du dump.
    pub path: PathBuf,
    /// Format lu : safetensors (mmap, streaming), npy ou pt (chargés en mémoire).
    pub format: DumpFormat,
    /// Tenseur comparé.
    pub tensor: String,
    /// Dtype stocké.
    pub dtype: LogitsDtype,
    /// Forme d'origine (les dimensions de tête sont aplaties en positions).
    pub shape: Vec<usize>,
    /// Hash des textes tokenisés, si le dump le porte (`surgeon bench dump`).
    pub prompts_sha256: Option<String>,
    /// Hash de tokenizer.json, si le dump le porte.
    pub tokenizer_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct PromptKl {
    /// Index du prompt (base 0) dans prompt_offsets.
    pub index: usize,
    /// Première position du prompt.
    pub start: usize,
    /// Position de fin (exclue).
    pub end: usize,
    /// KL moyenne sur les positions finies du prompt ; null si aucune.
    pub kl_mean: Option<f64>,
    /// Positions du prompt à KL infinie.
    pub infinite_positions: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct LogitsTiming {
    /// Durée totale de la comparaison.
    pub total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct BenchLogitsReport {
    /// Dump de référence (« avant »).
    pub before: DumpInfo,
    /// Dump comparé (« après »).
    pub after: DumpInfo,
    /// Positions comparées (dimensions de tête aplaties).
    pub positions: usize,
    /// Taille du vocabulaire.
    pub vocab_size: usize,
    /// Direction de la divergence : KL(before‖after), la distribution « avant » sert de référence.
    pub kl_direction: String,
    /// KL moyenne sur les positions à KL finie ; null si aucune.
    pub kl_mean: Option<f64>,
    /// KL maximale finie.
    pub kl_max: Option<f64>,
    /// Première position atteignant kl_max.
    pub kl_max_position: Option<usize>,
    /// 95e centile (rang le plus proche) des KL finies.
    pub kl_p95: Option<f64>,
    /// Positions où un token de probabilité > 0 avant a une probabilité nulle après (logit -inf) : KL infinie, exclue des statistiques finies.
    pub kl_infinite_positions: usize,
    /// Cosinus moyen entre les vecteurs de probabilités p et q.
    pub cosine_probs_mean: f64,
    /// Cosinus moyen entre les logits bruts (entrées finies des deux côtés) ; null si indéfini partout.
    pub cosine_logits_mean: Option<f64>,
    /// Positions où le cosinus des logits est défini (normes non nulles).
    pub cosine_logits_positions: usize,
    /// Part des positions où l'argmax change (égalités : plus petit index).
    pub top1_shift_rate: f64,
    /// Taille des ensembles top-k comparés.
    pub k: usize,
    /// Part des positions où l'ENSEMBLE des k meilleurs tokens diffère.
    pub topk_set_change_rate: f64,
    /// Recouvrement moyen |A∩B| / k des ensembles top-k.
    pub topk_overlap_mean: f64,
    /// Nombre de prompts si prompt_offsets est disponible et identique des deux côtés.
    pub prompts: Option<usize>,
    /// Les prompts à KL moyenne la plus élevée (ceux à KL infinie d'abord).
    pub worst_prompts: Vec<PromptKl>,
    /// KL par prompt, dans l'ordre des prompts (vide sans prompt_offsets).
    pub per_prompt: Vec<PromptKl>,
    /// Seuil d'avertissement sur kl_mean (strictement dépassé), si fourni.
    pub warn_kl: Option<f64>,
    /// Seuil d'avertissement sur top1_shift_rate (strictement dépassé), si fourni.
    pub warn_top1_shift: Option<f64>,
    /// Verdict des seuils.
    pub verdict: Verdict,
    /// Durées en millisecondes (exclues des comparaisons de déterminisme).
    pub timing_ms: LogitsTiming,
}

// ------------------------------------------------------------------ ppl

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct PplBaselineReport {
    /// `model` (second run de llama-perplexity) ou `value` (valeur fournie).
    pub source: String,
    /// Modèle de référence quand source = model.
    pub model: Option<PathBuf>,
    /// Perplexité de référence.
    pub ppl: f64,
    /// Incertitude ± de la référence, si connue.
    pub ppl_err: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct PplDelta {
    /// ppl − ppl_baseline, arrondi à 4 décimales AVANT toute comparaison.
    pub delta: f64,
    /// delta / ppl_baseline × 100, arrondi à 4 décimales.
    pub delta_relative_pct: f64,
    /// |delta| ≤ sqrt(err² + err_baseline²) ; null si une incertitude manque.
    pub within_noise: Option<bool>,
    /// delta < 0 : la perplexité a baissé (jamais un avertissement).
    pub improvement: bool,
    /// Seuil d'avertissement.
    pub warn_delta: f64,
    /// delta > warn_delta (strict).
    pub exceeded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct PplTiming {
    /// Durée du run sur le modèle évalué (conversion GGUF comprise).
    pub patched: u64,
    /// Durée du run de référence, si --baseline-model.
    pub baseline: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct BenchPplReport {
    /// Modèle évalué (.gguf ou dossier SafeTensors converti temporairement).
    pub model: PathBuf,
    /// Fichier texte évalué.
    pub text: PathBuf,
    /// sha256 du fichier texte.
    pub text_sha256: String,
    /// Contexte (-c).
    pub ctx: u32,
    /// --chunks, si fourni.
    pub chunks: Option<u32>,
    /// -t, si fourni.
    pub threads: Option<u32>,
    /// -ngl (0 : CPU seul, reproductible).
    pub n_gpu_layers: u32,
    /// Binaire llama-perplexity utilisé.
    pub binary: PathBuf,
    /// Arguments exacts du run sur le modèle évalué.
    pub args: Vec<String>,
    /// Perplexité finale (« Final estimate »).
    pub ppl: f64,
    /// Incertitude ± imprimée par llama.cpp, si présente.
    pub ppl_err: Option<f64>,
    /// Référence ; null sans --baseline-model ni --baseline-ppl.
    pub baseline: Option<PplBaselineReport>,
    /// Écart à la référence ; null sans référence.
    pub delta: Option<PplDelta>,
    /// Verdict du seuil --warn-delta (none sans référence).
    pub verdict: Verdict,
    /// Durées en millisecondes.
    pub timing_ms: PplTiming,
}

// ------------------------------------------------------------------ eval

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ItemResult {
    /// Identifiant (champ id, sinon item-<n> en base 1).
    pub id: String,
    /// Réponse de référence normalisée.
    pub gold: String,
    /// Réponse extraite de la sortie du modèle ; null si aucune.
    pub extracted: Option<String>,
    /// Règle d'extraction gagnante : hashes, boxed, bold, phrase, last_number.
    pub rule: Option<String>,
    /// Exact match (numérique ou chaîne normalisée).
    pub correct: bool,
    /// correct, wrong, no_answer, loop ou timeout.
    pub reason: String,
    /// Génération interrompue par le détecteur de boucles.
    pub loop_detected: bool,
    /// Génération coupée par max_new_tokens avant toute fin de séquence (backend natif ;
    /// toujours false avec llama.cpp, qui ne le rapporte pas ici).
    pub truncated: bool,
    /// Début de la sortie (200 caractères au plus, coupé proprement en UTF-8).
    pub output_excerpt: String,
    /// Index de l'échantillon (base 0), si --samples > 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct EvalScore {
    /// Questions évaluées.
    pub n: usize,
    /// Réponses exactes.
    pub correct: usize,
    /// Pass@1 = correct / n (un échantillon glouton par question).
    pub accuracy: f64,
    /// Borne basse de l'intervalle de Wilson à 95 %.
    pub ci95_low: f64,
    /// Borne haute de l'intervalle de Wilson à 95 %.
    pub ci95_high: f64,
    /// Générations interrompues par le détecteur de boucles (comptées fausses).
    pub loops: usize,
    /// Sorties sans réponse extractible (comptées fausses).
    pub no_answer: usize,
    /// Générations arrêtées par le délai (comptées fausses).
    pub timeouts: usize,
    /// Générations coupées par max_new_tokens (backend natif ; scorées telles quelles).
    pub truncated: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct EvalBaseline {
    /// Modèle de référence.
    pub model: PathBuf,
    /// Score de la référence.
    #[serde(flatten)]
    pub score: EvalScore,
    /// Résultat par question de la référence.
    pub items: Vec<ItemResult>,
    /// Cache des références : hit (lue, pas régénérée) ou miss (générée puis enregistrée) ;
    /// absent sans cache (--no-ref-cache, aucun dossier).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct PairedComparison {
    /// Questions justes des deux côtés.
    pub both_correct: usize,
    /// Questions fausses des deux côtés.
    pub both_wrong: usize,
    /// Ids justes pour la référence et faux après chirurgie.
    pub lost: Vec<String>,
    /// Ids faux pour la référence et justes après chirurgie.
    pub gained: Vec<String>,
    /// accuracy − accuracy de la référence.
    pub delta_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct EvalTiming {
    /// Durée du modèle évalué (conversion GGUF comprise).
    pub total: u64,
    /// Durée de la référence, si --baseline-model.
    pub baseline: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct BenchEvalReport {
    /// Modèle évalué.
    pub model: PathBuf,
    /// Dataset.
    pub dataset: PathBuf,
    /// sha256 du fichier dataset.
    pub dataset_sha256: String,
    /// Format demandé : gsm8k, math ou auto.
    pub format: String,
    /// sha256 du gabarit de prompt utilisé.
    pub prompt_template_sha256: String,
    /// Gabarit de chat appliqué au prompt rendu ; absent : texte brut.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    /// Message système par défaut du gabarit du modèle, s'il en déclare un.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_system: Option<String>,
    /// Tokens générés au plus par question.
    pub max_new_tokens: u32,
    /// Graine de l'échantillonnage glouton.
    pub seed: u32,
    /// Score du modèle évalué.
    #[serde(flatten)]
    pub score: EvalScore,
    /// Résultat par question, dans l'ordre du fichier.
    pub items: Vec<ItemResult>,
    /// Référence, si --baseline-model.
    pub baseline: Option<EvalBaseline>,
    /// Comparaison appariée question par question, si --baseline-model.
    pub paired: Option<PairedComparison>,
    /// Seuil --min-accuracy, si fourni.
    pub min_accuracy: Option<f64>,
    /// Verdict du seuil --min-accuracy.
    pub verdict: Verdict,
    /// Durées en millisecondes.
    pub timing_ms: EvalTiming,
    /// Génération autre que l'historique (sampling, échantillons, lots, backend) ; absent
    /// sinon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<super::generation::GenerationReport>,
}

// ------------------------------------------------------------------ refusal

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RefusalItem {
    /// Index du prompt (base 0) dans le fichier.
    pub index: usize,
    /// La réponse contient un marqueur de refus.
    pub refused: bool,
    /// Premier marqueur trouvé ; null si aucun.
    pub marker: Option<String>,
    /// Réponse générée, entière.
    pub output: String,
    /// Index de l'échantillon (base 0), si --samples > 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RefusalScore {
    /// Prompts évalués.
    pub n: usize,
    /// Réponses comptées comme refus.
    pub refused: usize,
    /// refused / n.
    pub refusal_rate: f64,
    /// Borne basse de l'intervalle de Wilson à 95 %.
    pub ci95_low: f64,
    /// Borne haute de l'intervalle de Wilson à 95 %.
    pub ci95_high: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RefusalBaseline {
    /// Modèle de référence.
    pub model: PathBuf,
    /// Score de la référence.
    #[serde(flatten)]
    pub score: RefusalScore,
    /// Résultat par prompt de la référence.
    pub items: Vec<RefusalItem>,
    /// Cache des références : hit (lue, pas régénérée) ou miss (générée puis enregistrée) ;
    /// absent sans cache (--no-ref-cache, aucun dossier).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RefusalPaired {
    /// Prompts refusés des deux côtés.
    pub both_refused: usize,
    /// Prompts acceptés des deux côtés.
    pub both_complied: usize,
    /// Index refusés par la référence et acceptés après chirurgie.
    pub unblocked: Vec<usize>,
    /// Index acceptés par la référence et refusés après chirurgie.
    pub newly_refused: Vec<usize>,
    /// refusal_rate − refusal_rate de la référence (négatif : moins de refus).
    pub delta_rate: f64,
    /// Borne basse à 95 % de delta_rate (Newcombe apparié, méthode 10).
    pub delta_ci95_low: f64,
    /// Borne haute à 95 % de delta_rate (Newcombe apparié, méthode 10).
    pub delta_ci95_high: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RefusalTiming {
    /// Durée du modèle évalué (chargement compris).
    pub total: u64,
    /// Durée de la référence, si --baseline-model.
    pub baseline: Option<u64>,
    /// Durée du juge (--judge llm|both) ; absent sinon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct BenchRefusalReport {
    /// Modèle évalué.
    pub model: PathBuf,
    /// sha256 des prompts (tableau JSON des textes, avant gabarit).
    pub prompts_sha256: String,
    /// Gabarit appliqué à chaque prompt : none, chatml, llama3, inst ou gemma.
    pub chat_template: String,
    /// Message système par défaut du gabarit du modèle, s'il en déclare un.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_system: Option<String>,
    /// Tokens générés au plus par prompt.
    pub max_new_tokens: u32,
    /// Marqueurs de refus cherchés (sans casse, aux frontières de mots).
    pub markers: Vec<String>,
    /// Score du modèle évalué.
    #[serde(flatten)]
    pub score: RefusalScore,
    /// Résultat par prompt, dans l'ordre du fichier.
    pub items: Vec<RefusalItem>,
    /// Référence, si --baseline-model.
    pub baseline: Option<RefusalBaseline>,
    /// Comparaison appariée prompt par prompt, si --baseline-model.
    pub paired: Option<RefusalPaired>,
    /// Verdicts du juge LLM (--judge llm|both) ; absent avec --judge rules, dont le rapport
    /// est celui d'avant les juges. Les champs ci-dessus restent ceux des marqueurs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeReport>,
    /// Durées en millisecondes.
    pub timing_ms: RefusalTiming,
    /// Génération autre que l'historique (sampling, échantillons, lots, backend) ; absent
    /// sinon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<super::generation::GenerationReport>,
}

// ------------------------------------------------------------------ refusal probe

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeItem {
    /// Index du prompt (base 0) dans le fichier.
    pub index: usize,
    /// first_token_prob : probabilité que le premier token de réponse ouvre un refus ;
    /// direction_projection : projection du flux résiduel sur la direction.
    pub score: f64,
    /// Score au-dessus du seuil.
    pub refused: bool,
    /// first_token_prob : le premier token le plus probable (diagnostic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeToken {
    /// Mot de la liste des préfixes de refus.
    pub word: String,
    /// Son premier token pour le tokenizer du modèle.
    pub id: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeDirectionInfo {
    /// Fichier de `surgeon extract`.
    pub path: PathBuf,
    pub layer: usize,
    pub site: String,
    pub pooling: String,
    /// Séparation des prompts d'extraction le long de la direction ; null si non écrite.
    pub d_prime: Option<f64>,
    /// Seuil calibré à l'extraction (positifs au-dessus).
    pub calibrated_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeComparison {
    /// Budget de la génération complète, en tokens.
    pub max_new_tokens: u32,
    /// Refus selon la génération complète (marqueurs).
    pub generated_refused: usize,
    /// Prompts où la sonde et la génération disent la même chose.
    pub agree: usize,
    /// agree / n.
    pub agreement: f64,
    pub both_refused: usize,
    /// Refus selon la sonde seulement.
    pub probe_only: usize,
    /// Refus selon la génération seulement.
    pub generation_only: usize,
    pub neither: usize,
    /// Corrélation de Pearson entre le score et le verdict généré (0/1, point-bisériale) ;
    /// null quand l'un des deux ne varie pas.
    pub point_biserial: Option<f64>,
    /// Durée de la génération / durée de la sonde.
    pub speedup: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeTiming {
    /// Sonde seule.
    pub probe: u64,
    /// Génération complète, avec --compare-generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ProbeReport {
    /// Toujours true : un score de sonde n'est jamais une preuve.
    pub is_heuristic: bool,
    /// Toujours heuristic_screening : à valider par une génération complète.
    pub evidence_level: String,
    /// Toujours le même rappel : une heuristique d'exploration, pas une preuve.
    pub caveat: String,
    pub model: PathBuf,
    pub mode: super::refusal::RefusalProbeMode,
    /// sha256 des prompts (tableau JSON des textes, avant gabarit).
    pub prompts_sha256: String,
    pub chat_template: String,
    pub chat_system: Option<String>,
    /// Seuil de décision du score.
    pub threshold: f64,
    pub prompt_count: usize,
    /// Prompts dont le score dépasse le seuil.
    pub refused: usize,
    /// refused / prompt_count.
    pub probe_refusal_rate: f64,
    /// Moyenne des scores.
    pub mean_score: f64,
    /// first-token-prob : probabilité moyenne que la réponse s'ouvre sur un refus.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_refusal_prob: Option<f64>,
    /// Détail par prompt.
    pub items: Vec<ProbeItem>,
    /// first_token_prob : les préfixes de refus et leurs tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_tokens: Option<Vec<ProbeToken>>,
    /// direction_projection : la direction utilisée.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<ProbeDirectionInfo>,
    /// --compare-generation : accord avec `bench refusal` sur les mêmes prompts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<ProbeComparison>,
    /// Commande entière, chargement compris, en millisecondes.
    pub duration_ms: u64,
    /// Durées des étapes en millisecondes.
    pub timing_ms: ProbeTiming,
}

// ------------------------------------------------------------------ dump

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct DumpTiming {
    /// Durée totale (chargement, forwards, écriture).
    pub total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct BenchDumpReport {
    /// Fichier .safetensors écrit (tenseurs logits et prompt_offsets).
    pub out: PathBuf,
    /// Dossier du modèle.
    pub model: PathBuf,
    /// Dtype de calcul : f32 ou bf16.
    pub dtype_used: String,
    /// Dtype des logits écrits.
    pub out_dtype: LogitsDtype,
    /// Prompts.
    pub n_prompts: usize,
    /// Positions écrites (somme des longueurs tokenisées).
    pub n_positions: usize,
    /// Taille du vocabulaire.
    pub vocab_size: usize,
    /// Troncature appliquée à chaque prompt.
    pub max_tokens: usize,
    /// Gabarit de chat appliqué aux prompts ; absent : texte brut.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    /// Prompts tronqués à max_tokens.
    pub truncated_prompts: usize,
    /// Hash des séquences de tokens (identique avant/après une chirurgie si mêmes prompts et tokenizer).
    pub prompts_sha256: String,
    /// sha256 de tokenizer.json.
    pub tokenizer_sha256: String,
    /// Taille du fichier écrit.
    pub file_bytes: u64,
    /// Durées en millisecondes.
    pub timing_ms: DumpTiming,
}

// ------------------------------------------------------------------ terminal

fn opt(value: Option<f64>, decimals: usize) -> String {
    value.map_or("—".to_owned(), |v| format!("{v:.decimals$}"))
}

pub(super) fn pct(rate: f64) -> String {
    format!("{:.1} %", rate * 100.0)
}

fn verdict_line(ui: &Ui, verdict: Verdict, threshold_warnings: &[String]) -> String {
    let text = match (verdict, threshold_warnings.first()) {
        (Verdict::Warn, Some(w)) => format!("avertissement : {w}"),
        (Verdict::Warn, None) => "avertissement".to_owned(),
        (Verdict::Ok, _) => "ok · seuils respectés".to_owned(),
        (Verdict::None, _) => "aucun seuil fourni".to_owned(),
    };
    format!("{} {}", ui::badge(ui, "VERDICT"), ui::metric(ui, text))
}

fn finish(
    ui: &Ui,
    mut lines: Vec<String>,
    run_warnings: &[String],
    threshold: &[String],
) -> String {
    lines.extend(
        run_warnings
            .iter()
            .filter(|w| !threshold.contains(w))
            .map(|w| format!("  {}", ui::dim(ui, &format!("! {w}")))),
    );
    lines.join("\n") + "\n"
}

pub fn logits_text(ui: &Ui, run: &BenchRun<BenchLogitsReport>) -> String {
    let r = &run.report;
    let infinite = match r.kl_infinite_positions {
        0 => String::new(),
        n => format!(" · ∞ sur {n} positions"),
    };
    let max_at = r
        .kl_max_position
        .map_or(String::new(), |p| format!(" @ {p}"));
    let mut lines = vec![
        format!(
            "{} logits · {} positions · vocab {}",
            ui::badge(ui, "BENCH"),
            ui::metric(ui, r.positions),
            ui::metric(ui, r.vocab_size)
        ),
        format!(
            "{} moyenne {} · max {}{max_at} · p95 {}{infinite} · cos(p,q) {}",
            ui::badge(ui, "KL"),
            ui::metric(ui, opt(r.kl_mean, 4)),
            ui::metric(ui, opt(r.kl_max, 4)),
            ui::metric(ui, opt(r.kl_p95, 4)),
            ui::metric(ui, format!("{:.4}", r.cosine_probs_mean))
        ),
        format!(
            "{} top-1 changé {} · top-{} (ensemble) {} · recouvrement {}",
            ui::badge(ui, "TOP-K"),
            ui::metric(ui, pct(r.top1_shift_rate)),
            r.k,
            ui::metric(ui, pct(r.topk_set_change_rate)),
            ui::metric(ui, format!("{:.2}", r.topk_overlap_mean))
        ),
    ];
    if !r.worst_prompts.is_empty() {
        let worst: Vec<String> = r
            .worst_prompts
            .iter()
            .map(|p| format!("#{} {}", p.index, opt(p.kl_mean, 4)))
            .collect();
        lines.push(format!(
            "  {}",
            ui::dim(ui, &format!("pires prompts : {}", worst.join(" · ")))
        ));
    }
    lines.push(verdict_line(ui, r.verdict, &run.threshold_warnings));
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}

fn ppl_value(ppl: f64, err: Option<f64>) -> String {
    match err {
        Some(e) => format!("{ppl:.4} ± {e:.2}"),
        None => format!("{ppl:.4}"),
    }
}

pub fn ppl_text(ui: &Ui, run: &BenchRun<BenchPplReport>) -> String {
    let r = &run.report;
    let value = match (&r.baseline, &r.delta) {
        (Some(b), Some(d)) => {
            let noise = match (d.within_noise, d.improvement) {
                (Some(true), _) => " (dans le bruit)",
                (_, true) => " (amélioration)",
                (Some(false), false) => " (hors du bruit)",
                (None, false) => "",
            };
            format!(
                "{} → {} · Δ {}{noise}",
                ui::metric(ui, ppl_value(b.ppl, b.ppl_err)),
                ui::metric(ui, ppl_value(r.ppl, r.ppl_err)),
                ui::metric(ui, format!("{:+.4}", d.delta))
            )
        }
        _ => ui::metric(ui, ppl_value(r.ppl, r.ppl_err)),
    };
    let lines = vec![
        format!(
            "{} ppl · ctx {}  {}",
            ui::badge(ui, "BENCH"),
            ui::metric(ui, r.ctx),
            ui::path(ui, &r.model)
        ),
        format!("{} {value}", ui::badge(ui, "PPL")),
        verdict_line(ui, r.verdict, &run.threshold_warnings),
    ];
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}

fn score_line(ui: &Ui, label: &str, s: &EvalScore) -> String {
    format!(
        "{} {label}{}/{} · {} · IC95 [{} ; {}] · boucles {}",
        ui::badge(ui, "PASS@1"),
        ui::metric(ui, s.correct),
        ui::metric(ui, s.n),
        ui::metric(ui, pct(s.accuracy)),
        ui::metric(ui, format!("{:.1}", s.ci95_low * 100.0)),
        ui::metric(ui, format!("{:.1}", s.ci95_high * 100.0)),
        ui::metric(ui, s.loops)
    )
}

/// "6 prompts", or "6 prompts × 3" when each one was answered several times.
fn count(ui: &Ui, n: usize, noun: &str, generation: Option<&GenerationReport>) -> String {
    match generation.filter(|g| g.samples > 1) {
        Some(g) => format!(
            "{} {noun} × {}",
            ui::metric(ui, n / g.samples),
            ui::metric(ui, g.samples)
        ),
        None => format!("{} {noun}", ui::metric(ui, n)),
    }
}

/// The label of the per-answer line: "réponses " when a prompt has several answers.
fn answers(generation: Option<&GenerationReport>) -> &'static str {
    match generation.is_some_and(|g| g.samples > 1) {
        true => "réponses ",
        false => "",
    }
}

/// The per-prompt statistics of `--samples`: mean rate, bootstrap interval over prompts,
/// dispersions; with a baseline, its line and the paired difference.
fn sampled_lines(ui: &Ui, generation: Option<&GenerationReport>) -> Vec<String> {
    let Some(g) = generation else {
        return Vec::new();
    };
    let line = |label: &str, s: &SampledScore| {
        format!(
            "{} {label}par prompt {} · IC95 bootstrap [{} ; {}] · dispersion intra {} · inter {}",
            ui::badge(ui, "ÉCHANT."),
            ui::metric(ui, pct(s.mean)),
            ui::metric(ui, format!("{:.1}", s.ci95_low * 100.0)),
            ui::metric(ui, format!("{:.1}", s.ci95_high * 100.0)),
            ui::metric(ui, format!("{:.4}", s.within_prompt_variance)),
            ui::metric(ui, format!("{:.4}", s.between_prompt_variance))
        )
    };
    let mut lines: Vec<String> = g.sampled.iter().map(|s| line("", s)).collect();
    lines.extend(g.baseline_sampled.iter().map(|s| line("référence ", s)));
    lines.extend(g.delta.iter().map(|d| {
        format!(
            "  {}",
            ui::dim(
                ui,
                &format!(
                    "par prompt : Δ {:+.1} points, IC95 bootstrap [{:+.1} ; {:+.1}]",
                    d.delta * 100.0,
                    d.ci95_low * 100.0,
                    d.ci95_high * 100.0
                )
            )
        )
    }));
    lines
}

/// Where the baseline came from, when a reference cache was in use.
fn cache_line(ui: &Ui, cache: Option<&str>) -> Option<String> {
    let what = match cache? {
        "hit" => "[cache hit] référence relue du cache, non régénérée",
        _ => "[cache miss] référence générée, puis enregistrée dans le cache",
    };
    Some(format!("  {}", ui::dim(ui, what)))
}

pub fn eval_text(ui: &Ui, run: &BenchRun<BenchEvalReport>) -> String {
    let r = &run.report;
    let mut lines = vec![
        format!(
            "{} eval · {} · {}  {}",
            ui::badge(ui, "BENCH"),
            count(ui, r.score.n, "questions", r.generation.as_ref()),
            ui::metric(ui, &r.format),
            ui::path(ui, &r.dataset)
        ),
        score_line(ui, answers(r.generation.as_ref()), &r.score),
    ];
    lines.extend(sampled_lines(ui, r.generation.as_ref()));
    if let (Some(b), Some(p)) = (&r.baseline, &r.paired) {
        lines.push(score_line(ui, "référence ", &b.score));
        lines.extend(cache_line(ui, b.cache.as_deref()));
        lines.push(format!(
            "  {}",
            ui::dim(
                ui,
                &format!(
                    "appariement : perdues {} · gagnées {} · Δ {:+.1} points",
                    p.lost.len(),
                    p.gained.len(),
                    p.delta_accuracy * 100.0
                )
            )
        ));
    }
    lines.push(verdict_line(ui, r.verdict, &run.threshold_warnings));
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}

fn refusal_line(ui: &Ui, label: &str, s: &RefusalScore) -> String {
    format!(
        "{} {label}{}/{} · {} · IC95 [{} ; {}]",
        ui::badge(ui, "REFUS"),
        ui::metric(ui, s.refused),
        ui::metric(ui, s.n),
        ui::metric(ui, pct(s.refusal_rate)),
        ui::metric(ui, format!("{:.1}", s.ci95_low * 100.0)),
        ui::metric(ui, format!("{:.1}", s.ci95_high * 100.0))
    )
}

pub fn refusal_text(ui: &Ui, run: &BenchRun<BenchRefusalReport>) -> String {
    let r = &run.report;
    let mut lines = vec![
        format!(
            "{} refusal · {} · {} · {} tokens max  {}",
            ui::badge(ui, "BENCH"),
            count(ui, r.score.n, "prompts", r.generation.as_ref()),
            ui::metric(ui, &r.chat_template),
            ui::metric(ui, r.max_new_tokens),
            ui::path(ui, &r.model)
        ),
        refusal_line(ui, answers(r.generation.as_ref()), &r.score),
    ];
    lines.extend(sampled_lines(ui, r.generation.as_ref()));
    if let (Some(b), Some(p)) = (&r.baseline, &r.paired) {
        lines.push(refusal_line(ui, "référence ", &b.score));
        lines.extend(cache_line(ui, b.cache.as_deref()));
        lines.push(format!(
            "  {}",
            ui::dim(
                ui,
                &format!(
                    "appariement : débloqués {} · nouveaux refus {} · Δ {:+.1} points, IC95 [{:+.1} ; {:+.1}]",
                    p.unblocked.len(),
                    p.newly_refused.len(),
                    p.delta_rate * 100.0,
                    p.delta_ci95_low * 100.0,
                    p.delta_ci95_high * 100.0
                )
            )
        ));
    }
    if let Some(j) = &r.judge {
        lines.extend(judge::text_lines(ui, j));
        if let Some(exceeded) = run.threshold_warnings.first() {
            lines.push(format!(
                "{} {}",
                ui::badge(ui, "VERDICT"),
                ui::metric(ui, format!("avertissement : {exceeded}"))
            ));
        }
    }
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}

pub fn probe_text(ui: &Ui, run: &BenchRun<ProbeReport>) -> String {
    let r = &run.report;
    let mode = match r.mode {
        super::refusal::RefusalProbeMode::FirstTokenProb => "first-token-prob",
        super::refusal::RefusalProbeMode::DirectionProjection => "direction-projection",
    };
    let mut lines = vec![
        format!(
            "{} refusal-probe · {} · {} prompts · seuil {}  {}",
            ui::badge(ui, "PROBE"),
            ui::metric(ui, mode),
            ui::metric(ui, r.prompt_count),
            ui::metric(ui, format!("{:.3}", r.threshold)),
            ui::path(ui, &r.model)
        ),
        format!(
            "{} {}/{} · {} (sonde, en {} ms)",
            ui::badge(ui, "REFUS"),
            ui::metric(ui, r.refused),
            ui::metric(ui, r.prompt_count),
            ui::metric(ui, pct(r.probe_refusal_rate)),
            ui::metric(ui, r.timing_ms.probe)
        ),
    ];
    if let Some(c) = &r.comparison {
        let correlation = c
            .point_biserial
            .map_or("indéfinie".to_owned(), |x| format!("{x:.2}"));
        lines.push(format!(
            "  {}",
            ui::dim(
                ui,
                &format!(
                    "génération ({} tokens) : {} refus · accord {}/{} ({}) · corrélation {} · \
                     sonde {}x plus rapide",
                    c.max_new_tokens,
                    c.generated_refused,
                    c.agree,
                    r.prompt_count,
                    pct(c.agreement),
                    correlation,
                    format_args!("{:.1}", c.speedup)
                )
            )
        ));
    }
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}

pub fn dump_text(ui: &Ui, run: &BenchRun<BenchDumpReport>) -> String {
    let r = &run.report;
    let lines = vec![
        format!(
            "{} dump · {} prompts · {} positions · vocab {} · {}",
            ui::badge(ui, "BENCH"),
            ui::metric(ui, r.n_prompts),
            ui::metric(ui, r.n_positions),
            ui::metric(ui, r.vocab_size),
            ui::metric(ui, &r.dtype_used)
        ),
        format!("{} {}", ui::badge(ui, "WRITE"), ui::path(ui, &r.out)),
    ];
    finish(ui, lines, &run.warnings, &run.threshold_warnings)
}
