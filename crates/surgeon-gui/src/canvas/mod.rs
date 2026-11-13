//! The studio workspace: typed cards on an infinite plane, wired output to input, saved as
//! `*.surgeon-studio.json`. Running a card runs its ancestors first, in wiring order; each
//! card is one `surgeon` command line (see [`crate::runner`]), so a card does exactly what
//! the CLI would do with the same flags.
//!
//! - Model: `inspect` (architecture, patchable tensors, layers; size on disk).
//! - Extract: `extract` (direction from two prompt sets).
//! - Patch: `apply --dry-run`, or `apply` with its verified backup.
//! - Graph Diff: `graph --baseline` (spectral delta, opened in the 3D view).
//! - Eval / Gate: `bench refusal --fail-on-warn` on a backend, GO / NO-GO.

pub mod view;

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use weight_surgeon::bench::score::Backend;
use weight_surgeon::patch::{
    Axis, LayerSelection, LayersField, Operation, OperationKind, PatchFile, Projection,
};

use crate::runner::{Finished, Step, Task};

pub const WORKSPACE_VERSION: u32 = 1;
/// Suffix of a saved workspace.
pub const EXTENSION: &str = ".surgeon-studio.json";
/// Where a forked or duplicated card lands relative to its source.
const DUPLICATE_OFFSET: [f32; 2] = [32.0, 32.0];
const FORK_GAP: f32 = 260.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workspace {
    pub version: u32,
    pub cards: Vec<Card>,
    pub wires: Vec<Wire>,
    /// Canvas pan (points) and zoom, restored with the file.
    #[serde(default)]
    pub pan: [f32; 2],
    #[serde(default = "one")]
    pub zoom: f32,
}

fn one() -> f32 {
    1.0
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            version: WORKSPACE_VERSION,
            cards: Vec::new(),
            wires: Vec::new(),
            pan: [0.0; 2],
            zoom: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub id: u32,
    /// Top-left corner, canvas coordinates.
    pub pos: [f32; 2],
    pub kind: CardKind,
}

/// An output of `from` feeding the input of `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wire {
    pub from: u32,
    pub to: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CardKind {
    Model(ModelCard),
    Extract(ExtractCard),
    Patch(PatchCard),
    GraphDiff(GraphDiffCard),
    Gate(GateCard),
}

/// The five kinds, in the order of the "add" menu.
pub const KINDS: [&str; 5] = ["Model", "Extract", "Patch", "Graph Diff", "Eval / Gate"];

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// SafeTensors file or folder, or GGUF.
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractCard {
    pub pos_prompts: String,
    pub neg_prompts: String,
    pub layer: usize,
    /// An identifier of the extraction registry (`surgeon plugins list`).
    pub method: String,
    /// Directions kept (`--rank`, svd rank k / leace); 1 is the plain direction.
    pub rank: u32,
    pub out: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchCard {
    pub operation: OperationKind,
    /// `"all"`, `"10-20"` or `"3,5,7"`.
    pub layers: String,
    pub projections: Vec<Projection>,
    pub alpha: f32,
    pub axis: Axis,
    /// Empty: the output of the nearest upstream Extract card.
    pub vector_file: String,
    /// Empty: the only tensor of the file (`direction` for an extract output).
    pub vector_tensor: String,
    pub backup: bool,
    /// `false`: Run is a dry run, nothing is written.
    pub apply: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphDiffCard {
    /// Reference model folder or file, or a `graph.json` exported before the patch.
    pub baseline: String,
    pub out_dir: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateCard {
    pub prompts: String,
    pub backend: Backend,
    /// Optional model compared against (`--baseline-model`).
    pub baseline: String,
    /// GO needs a refusal rate at most this (0..1).
    pub max_refusal: f64,
}

impl CardKind {
    /// A fresh card of [`KINDS`]`[index]`.
    pub fn new(index: usize) -> Self {
        match index {
            0 => Self::Model(ModelCard::default()),
            1 => Self::Extract(ExtractCard {
                pos_prompts: String::new(),
                neg_prompts: String::new(),
                layer: 14,
                method: weight_surgeon::plugin::DEFAULT_METHOD.into(),
                rank: 1,
                out: "direction.safetensors".into(),
            }),
            2 => Self::Patch(PatchCard {
                operation: OperationKind::Subtract,
                layers: "all".into(),
                projections: vec![Projection::Down],
                alpha: 1.0,
                axis: Axis::Cols,
                vector_file: String::new(),
                vector_tensor: String::new(),
                backup: true,
                apply: false,
            }),
            3 => Self::GraphDiff(GraphDiffCard {
                baseline: String::new(),
                out_dir: "graph-out".into(),
            }),
            _ => Self::Gate(GateCard {
                prompts: String::new(),
                backend: Backend::Native,
                baseline: String::new(),
                max_refusal: 0.1,
            }),
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::Model(_) => KINDS[0],
            Self::Extract(_) => KINDS[1],
            Self::Patch(_) => KINDS[2],
            Self::GraphDiff(_) => KINDS[3],
            Self::Gate(_) => KINDS[4],
        }
    }

    /// Whether an output of `from` can feed this card.
    pub fn accepts(&self, from: &CardKind) -> bool {
        use CardKind::*;
        match self {
            Model(_) => false,
            Extract(_) => matches!(from, Model(_)),
            Patch(_) => matches!(from, Model(_) | Extract(_) | Patch(_)),
            GraphDiff(_) => matches!(from, Model(_) | Patch(_)),
            Gate(_) => matches!(from, Model(_) | Patch(_) | GraphDiff(_)),
        }
    }
}

impl PatchCard {
    /// The selection this card targets, when its layer spec parses.
    pub fn selection(&self) -> Option<(LayerSelection, Vec<Projection>)> {
        let layers = self.layers.parse().ok()?;
        (!self.projections.is_empty()).then(|| (layers, self.projections.clone()))
    }

    /// The one-operation `patch.json` of this card.
    pub fn patch_file(&self, id: u32, vector_file: PathBuf) -> PatchFile {
        let tensor = (!self.vector_tensor.trim().is_empty()).then(|| self.vector_tensor.clone());
        let project_out = self.operation == OperationKind::ProjectOut;
        PatchFile {
            schema_version: 1,
            description: Some(format!("surgeon-gui, carte Patch #{id}")),
            backup: self.backup,
            operations: vec![Operation {
                id: format!("card-{id}"),
                layers: LayersField::Spec(self.layers.clone()),
                projections: self.projections.clone(),
                operation: self.operation,
                vector_file,
                // project_out reads b = a = v (directional ablation).
                vector_tensor_b: tensor.clone().filter(|_| project_out),
                vector_tensor: tensor,
                rank: None,
                allow_non_projector: false,
                alpha: self.alpha,
                axis: self.axis,
            }],
        }
    }
}

impl Workspace {
    pub fn card(&self, id: u32) -> Option<&Card> {
        self.cards.iter().find(|c| c.id == id)
    }

    pub fn card_mut(&mut self, id: u32) -> Option<&mut Card> {
        self.cards.iter_mut().find(|c| c.id == id)
    }

    fn next_id(&self) -> u32 {
        self.cards.iter().map(|c| c.id + 1).max().unwrap_or(1)
    }

    pub fn add(&mut self, kind: CardKind, pos: [f32; 2]) -> u32 {
        let id = self.next_id();
        self.cards.push(Card { id, pos, kind });
        id
    }

    /// The card and its wires.
    pub fn remove(&mut self, id: u32) {
        self.cards.retain(|c| c.id != id);
        self.wires.retain(|w| w.from != id && w.to != id);
    }

    pub fn parents(&self, id: u32) -> impl Iterator<Item = u32> + '_ {
        self.wires
            .iter()
            .filter(move |w| w.to == id)
            .map(|w| w.from)
    }

    /// `id` and every card downstream of it.
    fn descendants(&self, id: u32) -> BTreeSet<u32> {
        let mut seen = BTreeSet::from([id]);
        let mut stack = vec![id];
        while let Some(at) = stack.pop() {
            for w in self.wires.iter().filter(|w| w.from == at) {
                if seen.insert(w.to) {
                    stack.push(w.to);
                }
            }
        }
        seen
    }

    /// Wires `from -> to` when the kinds fit, the wire is new and no cycle appears.
    pub fn connect(&mut self, from: u32, to: u32) -> Result<(), String> {
        let (Some(a), Some(b)) = (self.card(from), self.card(to)) else {
            return Err("carte inconnue".into());
        };
        if !b.kind.accepts(&a.kind) {
            return Err(format!(
                "{} ne peut pas alimenter {}",
                a.kind.title(),
                b.kind.title()
            ));
        }
        if self.wires.contains(&Wire { from, to }) {
            return Err("câble déjà présent".into());
        }
        if self.descendants(to).contains(&from) {
            return Err("ce câble fermerait une boucle".into());
        }
        self.wires.push(Wire { from, to });
        Ok(())
    }

    /// A copy of one card, unwired, slightly offset.
    pub fn duplicate(&mut self, id: u32) -> Option<u32> {
        let card = self.card(id)?.clone();
        let pos = [
            card.pos[0] + DUPLICATE_OFFSET[0],
            card.pos[1] + DUPLICATE_OFFSET[1],
        ];
        Some(self.add(card.kind, pos))
    }

    /// Forks the experiment branch starting at `id`: the card and everything downstream are
    /// copied below the original, wired the same way, and the copy of `id` is fed by the same
    /// parents. Returns the copy of `id`.
    pub fn fork(&mut self, id: u32) -> Option<u32> {
        self.card(id)?;
        let branch = self.descendants(id);
        let (top, bottom) = self
            .cards
            .iter()
            .filter(|c| branch.contains(&c.id))
            .fold((f32::MAX, f32::MIN), |(lo, hi), c| {
                (lo.min(c.pos[1]), hi.max(c.pos[1]))
            });
        let shift = bottom - top + FORK_GAP;
        let mut copies = HashMap::new();
        for old in &branch {
            let card = self.card(*old)?.clone();
            let new = self.add(card.kind, [card.pos[0], card.pos[1] + shift]);
            copies.insert(*old, new);
        }
        let wires: Vec<Wire> = self
            .wires
            .iter()
            .filter_map(|w| match (copies.get(&w.from), copies.get(&w.to)) {
                (Some(&from), Some(&to)) => Some(Wire { from, to }),
                (None, Some(&to)) if w.to == id => Some(Wire { from: w.from, to }),
                _ => None,
            })
            .collect();
        self.wires.extend(wires);
        copies.get(&id).copied()
    }

    /// The ancestors of `id` then `id`, each after the cards feeding it.
    pub fn run_order(&self, id: u32) -> Vec<u32> {
        fn visit(ws: &Workspace, id: u32, out: &mut Vec<u32>) {
            if out.contains(&id) {
                return;
            }
            let mut parents: Vec<u32> = ws.parents(id).collect();
            parents.sort_unstable();
            for p in parents {
                visit(ws, p, out);
            }
            out.push(id);
        }
        let mut out = Vec::new();
        if self.card(id).is_some() {
            visit(self, id, &mut out);
        }
        out
    }

    /// The nearest upstream card (breadth first, `id` excluded) that `pick` accepts.
    fn upstream<'a, T>(&'a self, id: u32, pick: impl Fn(&'a CardKind) -> Option<T>) -> Option<T> {
        let mut seen = BTreeSet::from([id]);
        let mut queue: std::collections::VecDeque<u32> = self.parents(id).collect();
        while let Some(at) = queue.pop_front() {
            if !seen.insert(at) {
                continue;
            }
            if let Some(found) = self.card(at).and_then(|c| pick(&c.kind)) {
                return Some(found);
            }
            queue.extend(self.parents(at));
        }
        None
    }

    /// The model a card works on: its own path for a Model card, else the nearest upstream
    /// Model card.
    pub fn model_of(&self, id: u32) -> Option<&str> {
        fn own(k: &CardKind) -> Option<&str> {
            match k {
                CardKind::Model(m) if !m.path.trim().is_empty() => Some(m.path.as_str()),
                _ => None,
            }
        }
        self.card(id)
            .and_then(|c| own(&c.kind))
            .or_else(|| self.upstream(id, own))
    }

    /// The `patch.json` text of a Patch card, its vector made absolute (the file is written
    /// elsewhere than the workspace). The vector is the card's own, else the output of the
    /// nearest upstream Extract card.
    pub fn patch_json(&self, id: u32) -> Result<String, String> {
        let Some(CardKind::Patch(p)) = self.card(id).map(|c| &c.kind) else {
            return Err(format!("#{id} n'est pas une carte Patch"));
        };
        let vector = match p.vector_file.trim() {
            "" => self
                .upstream(id, |k| match k {
                    CardKind::Extract(e) => Some(e.out.clone()),
                    _ => None,
                })
                .ok_or_else(|| format!("#{id} Patch : ni vecteur, ni carte Extract en amont"))?,
            v => v.to_owned(),
        };
        let vector = std::path::absolute(&vector).map_err(|e| format!("{vector} : {e}"))?;
        serde_json::to_string_pretty(&p.patch_file(id, vector)).map_err(|e| e.to_string())
    }

    /// The command lines that run `id` and its ancestors. Patch cards write their
    /// `patch.json` into `scratch` here, so the steps can run as they are.
    pub fn plan(&self, id: u32, scratch: &Path) -> Result<Vec<Step>, String> {
        self.run_order(id)
            .into_iter()
            .map(|card| self.step(card, scratch))
            .collect()
    }

    fn step(&self, id: u32, scratch: &Path) -> Result<Step, String> {
        let card = self.card(id).ok_or("carte inconnue")?;
        let label = format!("#{id} {}", card.kind.title());
        let needs = |what: &str, value: &str| match value.trim() {
            "" => Err(format!("{label} : {what} manquant")),
            v => Ok(v.to_owned()),
        };
        let model = || {
            self.model_of(id)
                .map(str::to_owned)
                .ok_or_else(|| format!("{label} : aucune carte Model en amont"))
        };
        let args: Vec<String> = match &card.kind {
            CardKind::Model(m) => {
                let mut args = vec![
                    "inspect".into(),
                    "--model".into(),
                    needs("modèle", &m.path)?,
                ];
                for proj in ["gate", "up", "down", "o_proj", "embed_tokens"] {
                    args.extend(["--proj".into(), proj.into()]);
                }
                args
            }
            CardKind::Extract(e) => {
                let mut args = vec![
                    "extract".into(),
                    "--model".into(),
                    model()?,
                    "--pos-prompts".into(),
                    needs("prompts positifs", &e.pos_prompts)?,
                    "--neg-prompts".into(),
                    needs("prompts négatifs", &e.neg_prompts)?,
                    "--layer".into(),
                    e.layer.to_string(),
                    "--method".into(),
                    e.method.clone(),
                    "--out".into(),
                    needs("sortie", &e.out)?,
                    "--overwrite".into(),
                ];
                if e.rank > 1 {
                    args.extend(["--rank".into(), e.rank.to_string()]);
                }
                args
            }
            CardKind::Patch(p) => {
                let text = self.patch_json(id)?;
                std::fs::create_dir_all(scratch).map_err(|e| e.to_string())?;
                let file = scratch.join(format!("card-{id}.patch.json"));
                std::fs::write(&file, text).map_err(|e| format!("{} : {e}", file.display()))?;
                let mut args = vec![
                    "apply".into(),
                    "--model".into(),
                    model()?,
                    "--patch".into(),
                    file.display().to_string(),
                ];
                if !p.apply {
                    args.push("--dry-run".into());
                }
                args
            }
            CardKind::GraphDiff(g) => vec![
                "graph".into(),
                "--model".into(),
                model()?,
                "--baseline".into(),
                needs("baseline", &g.baseline)?,
                "--out-dir".into(),
                needs("dossier de sortie", &g.out_dir)?,
                "--overwrite".into(),
            ],
            CardKind::Gate(g) => {
                let mut args = vec![
                    "bench".into(),
                    "refusal".into(),
                    "--model".into(),
                    model()?,
                    "--prompts".into(),
                    needs("prompts", &g.prompts)?,
                    "--backend".into(),
                    g.backend.label().into(),
                    "--fail-on-warn".into(),
                ];
                if !g.baseline.trim().is_empty() {
                    args.extend(["--baseline-model".into(), g.baseline.trim().into()]);
                }
                args
            }
        };
        Ok(Step {
            card: Some(id),
            label,
            task: Task::Cli(args),
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("{} : {e}", path.display()))?;
        let ws: Self =
            serde_json::from_str(&text).map_err(|e| format!("{} : {e}", path.display()))?;
        match ws.version == WORKSPACE_VERSION {
            true => Ok(ws),
            false => Err(format!(
                "{} : version {} (attendu {WORKSPACE_VERSION})",
                path.display(),
                ws.version
            )),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{} : {e}", path.display()))
    }
}

/// The first number found under `key`, at any depth (report layouts differ per command).
pub fn find_number(value: &Value, key: &str) -> Option<f64> {
    match value {
        Value::Object(map) => map
            .get(key)
            .and_then(Value::as_f64)
            .or_else(|| map.values().find_map(|v| find_number(v, key))),
        Value::Array(items) => items.iter().find_map(|v| find_number(v, key)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Go,
    NoGo,
    /// The bench itself failed: no verdict.
    Error,
}

/// GO: the bench passed its own thresholds (`--fail-on-warn`, exit 0) and the refusal
/// rate is at most `max_refusal`. A threshold exceeded (exit 9) is NO-GO, any other
/// failure is no verdict at all.
pub fn verdict(run: &Finished, max_refusal: f64) -> Verdict {
    const THRESHOLD_EXCEEDED: i32 = 9;
    let rate = find_number(&run.envelope["data"], "refusal_rate");
    match run.code {
        0 if rate.is_some_and(|r| r <= max_refusal) => Verdict::Go,
        0 | THRESHOLD_EXCEEDED => Verdict::NoGo,
        _ => Verdict::Error,
    }
}

/// `layers.N.` in a tensor name.
fn layer_of(name: &str) -> Option<usize> {
    let (_, rest) = name.split_once("layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// The lines a card shows under its fields after a run.
pub fn summary(kind: &CardKind, run: &Finished) -> Vec<(String, String)> {
    let data = &run.envelope["data"];
    if run.code != 0 && !matches!(kind, CardKind::Gate(_)) {
        let message = run.envelope["error"]["message"].as_str().unwrap_or("échec");
        return vec![("erreur".into(), message.lines().next().unwrap_or("").into())];
    }
    let num = |key: &str| find_number(data, key).map_or("-".into(), |v| format!("{v:.4}"));
    match kind {
        CardKind::Model(_) => {
            let tensors = data["tensors"].as_array().map_or(&[][..], Vec::as_slice);
            let layers: BTreeSet<usize> = tensors
                .iter()
                .filter_map(|t| layer_of(t["name"].as_str()?))
                .collect();
            vec![
                (
                    "architecture".into(),
                    data["arch"]["id"].as_str().unwrap_or("-").into(),
                ),
                ("couches".into(), layers.len().to_string()),
                ("tenseurs ciblables".into(), tensors.len().to_string()),
            ]
        }
        CardKind::Extract(_) => vec![
            (
                "sortie".into(),
                data["out_path"].as_str().unwrap_or("-").into(),
            ),
            ("couche".into(), num("layer")),
        ],
        CardKind::Patch(_) => {
            let ops = data["report"]["operations"].as_array();
            let tensors: usize = ops.map_or(0, |ops| {
                ops.iter()
                    .filter_map(|o| o["tensors"].as_array())
                    .map(Vec::len)
                    .sum()
            });
            let mut lines = vec![
                (
                    "mode".into(),
                    match data["report"]["dry_run"].as_bool() {
                        Some(true) => "dry-run (rien écrit)".into(),
                        _ => "appliqué".into(),
                    },
                ),
                ("tenseurs".into(), tensors.to_string()),
            ];
            let backups = data["backups"].as_array().map_or(0, Vec::len);
            if backups > 0 {
                lines.push(("backups".into(), backups.to_string()));
            }
            lines
        }
        CardKind::GraphDiff(_) => vec![
            ("nœuds modifiés".into(), num("nodes_changed")),
            ("Δ relatif max".into(), num("max_relative_change")),
        ],
        CardKind::Gate(_) => vec![
            ("taux de refus".into(), num("refusal_rate")),
            ("Δ vs baseline".into(), num("delta_rate")),
        ],
    }
}

/// The backups an apply left (`data.backups` of its envelope).
pub fn backups_of(run: &Finished) -> Vec<PathBuf> {
    run.envelope["data"]["backups"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(Value::as_str)
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Model -> Extract -> Patch -> Gate, plus a Graph Diff on the Patch.
    fn pipeline() -> (Workspace, [u32; 5]) {
        let mut ws = Workspace::default();
        let ids = [0, 1, 2, 3, 4].map(|k| ws.add(CardKind::new(k), [k as f32 * 300.0, 0.0]));
        let [model, extract, patch, diff, gate] = ids;
        if let CardKind::Model(m) = &mut ws.card_mut(model).unwrap().kind {
            m.path = "models/qwen".into();
        }
        for (from, to) in [
            (model, extract),
            (extract, patch),
            (patch, diff),
            (patch, gate),
        ] {
            ws.connect(from, to).unwrap();
        }
        (ws, ids)
    }

    fn finished(code: i32, envelope: Value) -> Finished {
        Finished {
            card: None,
            label: String::new(),
            code,
            envelope,
            seconds: 0.0,
        }
    }

    #[test]
    fn a_workspace_round_trips_through_its_json_file() {
        let (mut ws, _) = pipeline();
        ws.pan = [12.0, -40.0];
        ws.zoom = 0.75;
        let dir = std::env::temp_dir().join(format!("surgeon-gui-ws-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("exp{EXTENSION}"));
        ws.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#""type": "graph_diff""#), "{text}");
        assert!(text.contains(r#""operation": "subtract""#));
        assert!(text.contains(
            r#""projections": [
          "down_proj""#
        ));
        assert_eq!(Workspace::load(&path).unwrap(), ws);

        let mut old: Value = serde_json::from_str(&text).unwrap();
        old["version"] = json!(0);
        std::fs::write(&path, old.to_string()).unwrap();
        assert!(Workspace::load(&path).unwrap_err().contains("version 0"));
        // A file written without the view fields still opens.
        let bare = json!({"version": 1, "cards": [], "wires": []});
        std::fs::write(&path, bare.to_string()).unwrap();
        assert_eq!(Workspace::load(&path).unwrap().zoom, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wires_respect_the_card_types_and_never_loop() {
        let (mut ws, [model, extract, patch, diff, gate]) = pipeline();
        assert!(ws.connect(gate, model).unwrap_err().contains("ne peut pas"));
        assert!(ws.connect(model, extract).unwrap_err().contains("déjà"));
        let second = ws.add(CardKind::new(2), [0.0, 0.0]);
        ws.connect(patch, second).unwrap();
        assert!(ws.connect(second, patch).unwrap_err().contains("boucle"));
        assert!(ws.connect(patch, patch).unwrap_err().contains("boucle"));
        ws.connect(diff, gate).unwrap();
        ws.remove(patch);
        assert!(ws.wires.iter().all(|w| w.from != patch && w.to != patch));
        assert_eq!(ws.wires.len(), 2, "{:?}", ws.wires);
    }

    #[test]
    fn a_fork_copies_the_branch_and_keeps_its_upstream() {
        let (mut ws, [model, extract, patch, diff, gate]) = pipeline();
        let before = ws.cards.len();
        let copy = ws.fork(patch).unwrap();
        assert_eq!(
            ws.cards.len(),
            before + 3,
            "patch, diff and gate are copied"
        );
        assert!(ws.wires.contains(&Wire {
            from: extract,
            to: copy
        }));
        let children: Vec<u32> = ws
            .wires
            .iter()
            .filter(|w| w.from == copy)
            .map(|w| w.to)
            .collect();
        assert_eq!(children.len(), 2);
        assert!(!children.contains(&diff) && !children.contains(&gate));
        assert!(ws.card(copy).unwrap().pos[1] > ws.card(patch).unwrap().pos[1]);
        // The copy is independent: tuning it leaves the original alone.
        if let CardKind::Patch(p) = &mut ws.card_mut(copy).unwrap().kind {
            p.alpha = 1.2;
        }
        let alpha = |ws: &Workspace, id| match &ws.card(id).unwrap().kind {
            CardKind::Patch(p) => p.alpha,
            _ => unreachable!(),
        };
        assert_eq!((alpha(&ws, patch), alpha(&ws, copy)), (1.0, 1.2));

        let dup = ws.duplicate(model).unwrap();
        assert_eq!(ws.card(dup).unwrap().kind, ws.card(model).unwrap().kind);
        assert_eq!(
            ws.parents(dup).count() + ws.wires.iter().filter(|w| w.from == dup).count(),
            0
        );
    }

    #[test]
    fn running_a_card_runs_its_ancestors_first_with_cli_flags() {
        let (mut ws, [model, extract, patch, _, gate]) = pipeline();
        if let CardKind::Extract(e) = &mut ws.card_mut(extract).unwrap().kind {
            e.pos_prompts = "pos.json".into();
            e.neg_prompts = "neg.json".into();
            e.rank = 4;
        }
        if let CardKind::Gate(g) = &mut ws.card_mut(gate).unwrap().kind {
            g.prompts = "harmful.json".into();
        }
        let scratch = std::env::temp_dir().join(format!("surgeon-gui-plan-{}", std::process::id()));
        let steps = ws.plan(gate, &scratch).unwrap();
        let cards: Vec<_> = steps.iter().map(|s| s.card.unwrap()).collect();
        assert_eq!(cards, [model, extract, patch, gate]);
        // Every command line is one the real CLI parser accepts.
        for step in &steps {
            let Task::Cli(args) = &step.task else {
                panic!()
            };
            let argv = std::iter::once("surgeon").chain(args.iter().map(String::as_str));
            let parsed = weight_surgeon::app::parse_cli(argv, &mut Vec::new(), &mut Vec::new());
            assert!(parsed.is_ok(), "{args:?}");
        }
        let Task::Cli(apply) = &steps[2].task else {
            panic!()
        };
        assert!(
            apply.contains(&"--dry-run".to_owned()),
            "Run is a dry run by default"
        );
        let written: PatchFile = weight_surgeon::patch::load_patch(Path::new(&apply[4])).unwrap();
        assert!(written.operations[0].vector_file.is_absolute());
        assert!(written.operations[0]
            .vector_file
            .ends_with("direction.safetensors"));
        let _ = std::fs::remove_dir_all(&scratch);

        if let CardKind::Model(m) = &mut ws.card_mut(model).unwrap().kind {
            m.path.clear();
        }
        assert!(ws
            .plan(gate, &scratch)
            .unwrap_err()
            .contains("modèle manquant"));
    }

    #[test]
    fn the_gate_verdict_needs_a_pass_and_a_low_refusal_rate() {
        let data = |rate: f64| json!({"data": {"model": {"refusal_rate": rate}}});
        assert_eq!(verdict(&finished(0, data(0.05)), 0.1), Verdict::Go);
        assert_eq!(verdict(&finished(0, data(0.30)), 0.1), Verdict::NoGo);
        assert_eq!(verdict(&finished(9, data(0.05)), 0.1), Verdict::NoGo);
        assert_eq!(verdict(&finished(8, json!({})), 0.1), Verdict::Error);
        let lines = summary(&CardKind::new(4), &finished(0, data(0.05)));
        assert_eq!(lines[0].1, "0.0500");
    }

    #[test]
    fn summaries_read_the_envelopes() {
        let inspect = json!({"data": {"arch": {"id": "qwen2"}, "tensors": [
            {"name": "model.layers.0.mlp.down_proj.weight"},
            {"name": "model.layers.1.mlp.down_proj.weight"},
            {"name": "model.layers.1.mlp.up_proj.weight"},
            {"name": "model.embed_tokens.weight"}]}});
        let lines = summary(&CardKind::new(0), &finished(0, inspect));
        assert_eq!(lines[0].1, "qwen2");
        assert_eq!((lines[1].1.as_str(), lines[2].1.as_str()), ("2", "4"));
        let failed = json!({"error": {"message": "patch invalide :\n  - operations"}});
        assert_eq!(
            summary(&CardKind::new(2), &finished(2, failed))[0].1,
            "patch invalide :"
        );
        let apply = finished(0, json!({"data": {"backups": ["m.safetensors.bak-1"]}}));
        assert_eq!(backups_of(&apply), [PathBuf::from("m.safetensors.bak-1")]);
    }
}
