use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;

use super::command::{last_lines, CommandRunner};
use super::error::{EvalError, Result};
use super::generator::find_binary;
use super::runner::{EvalReport, EvalVerdict};
use super::safetensors_gguf::{convert_safetensors, is_convertible_dir, WeightType};

/// Types accepted by `llama-quantize`. Anything else is refused before any command runs.
pub const ALLOWED_QUANTS: [&str; 17] = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
];
const OUTTYPES: [&str; 3] = ["f16", "bf16", "f32"];
const CONVERTERS: [&str; 2] = ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"];
const QUANTIZERS: [&str; 2] = ["llama-quantize", "quantize"];
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const STDERR_TAIL_LINES: usize = 20;

#[derive(Debug, Clone, Serialize)]
pub struct GgufExportConfig {
    /// Root of the llama.cpp checkout (holds the converter script and `build/`).
    pub llama_cpp_dir: PathBuf,
    /// Hugging Face model directory (config.json, tokenizer, *.safetensors).
    pub model_dir: PathBuf,
    pub output_gguf: PathBuf,
    /// `None`: the converted file is the final output.
    pub quant: Option<String>,
    /// "f16", "bf16" or "f32".
    pub intermediate_type: String,
    /// Usually "python3"; on Windows often "python" or "py".
    pub python_bin: String,
    pub keep_intermediate: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufExportReport {
    /// No longer on disk unless `keep_intermediate` (or moved to `final_gguf` without quant).
    pub intermediate_gguf: PathBuf,
    pub final_gguf: PathBuf,
    pub steps: Vec<StepReport>,
    pub final_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    pub step: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub duration_ms: u128,
}

/// Convert, optionally quantize, then verify the GGUF magic. Every input is validated and
/// every tool located before the first command, so a typo never wastes a long conversion.
pub fn export_gguf(cfg: &GgufExportConfig, runner: &dyn CommandRunner) -> Result<GgufExportReport> {
    export_with_path(cfg, runner, std::env::var_os("PATH"))
}

/// Exports only a `Healthy` model, or a `Degraded` one with `allow_degraded`. A `Broken`
/// model, or a report with no result, returns `Ok(None)` without running anything.
pub fn export_if_healthy(
    report: &EvalReport,
    cfg: &GgufExportConfig,
    runner: &dyn CommandRunner,
    allow_degraded: bool,
) -> Result<Option<GgufExportReport>> {
    let exportable = match report.verdict {
        EvalVerdict::Healthy => true,
        EvalVerdict::Degraded => allow_degraded,
        EvalVerdict::Broken => false,
    };
    if !exportable || report.results.is_empty() {
        return Ok(None);
    }
    export_gguf(cfg, runner).map(Some)
}

fn export_with_path(
    cfg: &GgufExportConfig,
    runner: &dyn CommandRunner,
    path_var: Option<OsString>,
) -> Result<GgufExportReport> {
    let quant = cfg.quant.as_deref().map(normalize_quant).transpose()?;
    let outtype = cfg.intermediate_type.to_ascii_lowercase();
    if !OUTTYPES.contains(&outtype.as_str()) {
        return Err(EvalError::InvalidConfig(format!(
            "intermediate_type « {} » (attendu f16, bf16 ou f32)",
            cfg.intermediate_type
        )));
    }
    if cfg.python_bin.trim().is_empty() {
        return Err(EvalError::InvalidConfig("python_bin vide".into()));
    }
    if !cfg.model_dir.is_dir() {
        return Err(EvalError::InvalidConfig(format!(
            "dossier du modèle introuvable : {}",
            cfg.model_dir.display()
        )));
    }
    let out_dir = match cfg.output_gguf.parent() {
        Some(p) if p.as_os_str().is_empty() => Path::new("."),
        Some(p) => p,
        None => Path::new("."),
    };
    if !out_dir.is_dir() || cfg.output_gguf.file_name().is_none() {
        return Err(EvalError::InvalidConfig(format!(
            "sortie invalide : {} (le dossier parent doit exister)",
            cfg.output_gguf.display()
        )));
    }
    let script = CONVERTERS
        .iter()
        .map(|name| cfg.llama_cpp_dir.join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| EvalError::BinaryNotFound {
            name: CONVERTERS[0].into(),
            hint: format!(
                "ni {} ni {} dans {} : --llama-cpp-dir doit pointer vers la racine de llama.cpp",
                CONVERTERS[0],
                CONVERTERS[1],
                cfg.llama_cpp_dir.display()
            ),
        })?;
    let quantizer = match quant {
        Some(_) => Some(
            find_binary(&QUANTIZERS, std::slice::from_ref(&cfg.llama_cpp_dir), path_var).ok_or_else(
                || EvalError::BinaryNotFound {
                    name: QUANTIZERS[0].into(),
                    hint: format!(
                        "compilez llama.cpp (cmake --build build) dans {} ou ajoutez llama-quantize au PATH",
                        cfg.llama_cpp_dir.display()
                    ),
                },
            )?,
        ),
        None => None,
    };
    let intermediate = cfg.output_gguf.with_extension(format!("{outtype}.gguf"));

    let mut steps = Vec::new();
    let convert_args = vec![
        utf8(&script)?,
        utf8(&cfg.model_dir)?,
        "--outfile".into(),
        utf8(&intermediate)?,
        "--outtype".into(),
        outtype,
    ];
    let cleanup = |paths: &[&Path]| {
        for p in paths {
            // Best effort: a leftover partial file is junk, never the user's data.
            let _ = std::fs::remove_file(p);
        }
    };
    if let Err(e) = run_step(
        runner,
        "convert",
        Path::new(&cfg.python_bin),
        convert_args,
        &mut steps,
    ) {
        if !cfg.keep_intermediate {
            cleanup(&[&intermediate]);
        }
        return Err(e);
    }

    match (quant, quantizer) {
        (Some(quant), Some(quantizer)) => {
            let args = vec![utf8(&intermediate)?, utf8(&cfg.output_gguf)?, quant];
            let result = run_step(runner, "quantize", &quantizer, args, &mut steps);
            if !cfg.keep_intermediate {
                cleanup(&[&intermediate]);
            }
            if let Err(e) = result {
                cleanup(&[&cfg.output_gguf]);
                return Err(e);
            }
        }
        _ => std::fs::rename(&intermediate, &cfg.output_gguf).map_err(|e| EvalError::Io {
            path: intermediate.clone(),
            message: format!("déplacement vers {} : {e}", cfg.output_gguf.display()),
        })?,
    }

    let final_size_bytes = verify_gguf(&cfg.output_gguf)?;
    Ok(GgufExportReport {
        intermediate_gguf: intermediate,
        final_gguf: cfg.output_gguf.clone(),
        steps,
        final_size_bytes,
    })
}

/// Whitelist match after uppercasing: "q4_k_m" is accepted as "Q4_K_M".
fn normalize_quant(quant: &str) -> Result<String> {
    let upper = quant.trim().to_ascii_uppercase();
    match ALLOWED_QUANTS.contains(&upper.as_str()) {
        true => Ok(upper),
        false => Err(EvalError::InvalidQuant(quant.to_owned())),
    }
}

/// `CommandRunner` takes strings: refuse non-UTF-8 paths rather than mangle them.
fn utf8(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| EvalError::InvalidConfig(format!("chemin non UTF-8 : {}", path.display())))
}

fn run_step(
    runner: &dyn CommandRunner,
    step: &str,
    program: &Path,
    args: Vec<String>,
    steps: &mut Vec<StepReport>,
) -> Result<()> {
    let started = Instant::now();
    let out = runner.run(program, &args, None)?;
    if out.status_code != 0 {
        return Err(EvalError::CommandFailed {
            step: step.into(),
            code: out.status_code,
            stderr_tail: last_lines(&out.stderr, STDERR_TAIL_LINES),
        });
    }
    steps.push(StepReport {
        step: step.into(),
        program: program.to_path_buf(),
        args,
        duration_ms: started.elapsed().as_millis(),
    });
    Ok(())
}

/// Message of the error returned when a conversion is needed without llama.cpp.
pub const LLAMA_DIR_REQUIRED: &str =
    "--llama-cpp-dir (ou LLAMA_CPP_DIR) requis pour convertir en GGUF";

/// A GGUF llama.cpp can read: the model itself, or a temporary f16 conversion that is
/// deleted when this value drops (errors and panics included).
#[derive(Debug)]
pub struct GgufModel {
    path: PathBuf,
    temporary: bool,
}

impl GgufModel {
    /// A not yet written temporary file in the temp dir; the unique name keeps concurrent
    /// runs (and a leftover of a killed one) apart.
    fn temporary(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!("weight-surgeon-{tag}-{}-{nanos}.gguf", std::process::id());
        Self {
            path: std::env::temp_dir().join(name),
            temporary: true,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_temporary(&self) -> bool {
        self.temporary
    }
}

impl Drop for GgufModel {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// The SafeTensors directory of a model: itself, or the parent of a single file.
pub fn model_dir(model: &Path) -> PathBuf {
    match model.is_dir() {
        true => model.to_path_buf(),
        false => model
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
    }
}

/// `.gguf` as is; otherwise one unquantized f16 conversion (see [`ensure_gguf_as`] for f32)
/// to `<temp dir>/weight-surgeon-<tag>-<pid>-<nanos>.gguf`, deleted when the returned value
/// drops. Point `TMPDIR` / `TEMP` at a big disk for a large model. With a llama.cpp checkout
/// holding `convert_hf_to_gguf.py` that script converts; otherwise a Qwen2, Llama, Mistral,
/// Gemma, Gemma 2 or Phi-3 model goes through the pure Rust converter (a binaries-only build
/// such as the Vulkan release, no python), and any other model is refused with
/// [`LLAMA_DIR_REQUIRED`].
pub fn ensure_gguf(
    model: &Path,
    llama_cpp_dir: Option<&Path>,
    tag: &str,
    runner: &dyn CommandRunner,
) -> Result<GgufModel> {
    ensure_gguf_as(model, llama_cpp_dir, tag, runner, WeightType::F16)
}

/// [`ensure_gguf`] with the matrix type of the conversion (f32 is lossless from bf16).
pub fn ensure_gguf_as(
    model: &Path,
    llama_cpp_dir: Option<&Path>,
    tag: &str,
    runner: &dyn CommandRunner,
    dtype: WeightType,
) -> Result<GgufModel> {
    if is_gguf(model) {
        return Ok(GgufModel {
            path: model.to_path_buf(),
            temporary: false,
        });
    }
    let model_dir = model_dir(model);
    let file = GgufModel::temporary(tag);
    let has_script = llama_cpp_dir.is_some_and(|d| CONVERTERS.iter().any(|c| d.join(c).is_file()));
    if !has_script && is_convertible_dir(&model_dir) {
        convert_safetensors(&model_dir, &file.path, dtype)?;
        return Ok(file);
    }
    let llama_cpp_dir =
        llama_cpp_dir.ok_or_else(|| EvalError::InvalidConfig(LLAMA_DIR_REQUIRED.into()))?;
    let cfg = GgufExportConfig {
        llama_cpp_dir: llama_cpp_dir.to_path_buf(),
        model_dir,
        output_gguf: file.path.clone(),
        quant: None,
        intermediate_type: match dtype {
            WeightType::F16 => "f16",
            WeightType::F32 => "f32",
        }
        .into(),
        python_bin: if cfg!(windows) { "python" } else { "python3" }.into(),
        keep_intermediate: false,
    };
    export_gguf(&cfg, runner)?;
    Ok(file)
}

fn verify_gguf(path: &Path) -> Result<u64> {
    let invalid = |reason: &str| EvalError::InvalidGguf {
        path: path.to_path_buf(),
        reason: reason.into(),
    };
    let mut file = std::fs::File::open(path).map_err(|e| invalid(&format!("absent ({e})")))?;
    let size = file.metadata().map_err(|e| invalid(&e.to_string()))?.len();
    if size == 0 {
        return Err(invalid("fichier vide"));
    }
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) if &magic == GGUF_MAGIC => Ok(size),
        _ => Err(invalid("signature GGUF absente")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::command::CommandOutput;
    use crate::eval::runner::PromptResult;
    use crate::eval::PromptStatus;
    use std::cell::RefCell;

    /// Records calls and writes the file each tool would produce.
    struct MockRunner {
        calls: RefCell<Vec<(PathBuf, Vec<String>)>>,
        fail_call: Option<usize>,
        content: Vec<u8>,
    }

    impl MockRunner {
        fn new(content: &[u8]) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                fail_call: None,
                content: content.to_vec(),
            }
        }
        fn calls(&self) -> Vec<(PathBuf, Vec<String>)> {
            self.calls.borrow().clone()
        }
    }

    impl CommandRunner for MockRunner {
        fn run(
            &self,
            program: &Path,
            args: &[String],
            _cwd: Option<&Path>,
        ) -> Result<CommandOutput> {
            let index = self.calls.borrow().len();
            self.calls
                .borrow_mut()
                .push((program.to_path_buf(), args.to_vec()));
            if self.fail_call == Some(index) {
                let stderr: String = (1..=50).map(|i| format!("erreur {i}\n")).collect();
                return Ok(CommandOutput {
                    status_code: 1,
                    stdout: String::new(),
                    stderr,
                });
            }
            let out = match args.iter().position(|a| a == "--outfile") {
                Some(i) => &args[i + 1],
                None => &args[1],
            };
            std::fs::write(out, &self.content).unwrap();
            Ok(CommandOutput {
                status_code: 0,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    struct Fixture {
        _root: tempfile::TempDir,
        cfg: GgufExportConfig,
        quantize: PathBuf,
    }

    fn fixture(quant: Option<&str>) -> Fixture {
        let root = tempfile::tempdir().unwrap();
        let llama = root.path().join("llama.cpp");
        let bin = llama.join("build/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(llama.join("convert_hf_to_gguf.py"), "").unwrap();
        let quantize = bin.join(format!("llama-quantize{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&quantize, "").unwrap();
        let model = root.path().join("model");
        std::fs::create_dir(&model).unwrap();
        let cfg = GgufExportConfig {
            llama_cpp_dir: llama,
            model_dir: model,
            output_gguf: root.path().join("out.gguf"),
            quant: quant.map(String::from),
            intermediate_type: "f16".into(),
            python_bin: "python3".into(),
            keep_intermediate: false,
        };
        Fixture {
            _root: root,
            cfg,
            quantize,
        }
    }

    const VALID: &[u8] = b"GGUF\x03\x00\x00\x00payload";

    fn s(p: &Path) -> String {
        p.to_str().unwrap().to_owned()
    }

    #[test]
    fn convert_then_quantize_in_order_with_exact_args() {
        let f = fixture(Some("q4_k_m"));
        let runner = MockRunner::new(VALID);
        let report = export_with_path(&f.cfg, &runner, None).unwrap();
        let inter = f.cfg.output_gguf.with_extension("f16.gguf");
        let script = f.cfg.llama_cpp_dir.join("convert_hf_to_gguf.py");
        assert_eq!(
            runner.calls(),
            [
                (
                    PathBuf::from("python3"),
                    vec![
                        s(&script),
                        s(&f.cfg.model_dir),
                        "--outfile".into(),
                        s(&inter),
                        "--outtype".into(),
                        "f16".into()
                    ]
                ),
                (
                    f.quantize.clone(),
                    vec![s(&inter), s(&f.cfg.output_gguf), "Q4_K_M".into()]
                ),
            ]
        );
        assert!(!inter.exists(), "intermediate removed");
        assert_eq!(report.final_size_bytes, VALID.len() as u64);
        assert_eq!(report.steps.len(), 2);
        assert_eq!(report.intermediate_gguf, inter);
    }

    #[test]
    fn without_quant_one_command_then_move() {
        let f = fixture(None);
        let runner = MockRunner::new(VALID);
        let report = export_with_path(&f.cfg, &runner, None).unwrap();
        assert_eq!(runner.calls().len(), 1);
        assert!(!report.intermediate_gguf.exists());
        assert_eq!(std::fs::read(&f.cfg.output_gguf).unwrap(), VALID);
    }

    #[test]
    fn keep_intermediate_and_converter_fallback() {
        let mut f = fixture(Some("Q8_0"));
        f.cfg.keep_intermediate = true;
        let legacy = f.cfg.llama_cpp_dir.join("convert-hf-to-gguf.py");
        std::fs::rename(f.cfg.llama_cpp_dir.join("convert_hf_to_gguf.py"), &legacy).unwrap();
        let runner = MockRunner::new(VALID);
        let report = export_with_path(&f.cfg, &runner, None).unwrap();
        assert_eq!(runner.calls()[0].1[0], s(&legacy));
        assert!(report.intermediate_gguf.exists());
    }

    #[test]
    fn first_failure_stops_the_pipeline_with_a_short_tail() {
        let f = fixture(Some("Q4_K_M"));
        let mut runner = MockRunner::new(VALID);
        runner.fail_call = Some(0);
        let err = export_with_path(&f.cfg, &runner, None).unwrap_err();
        let EvalError::CommandFailed {
            step,
            code,
            stderr_tail,
        } = &err
        else {
            panic!("{err:?}")
        };
        assert_eq!((step.as_str(), *code), ("convert", 1));
        assert_eq!(stderr_tail.lines().count(), 20);
        assert!(stderr_tail.ends_with("erreur 50"));
        assert_eq!(runner.calls().len(), 1, "quantize never launched");
    }

    #[test]
    fn quantize_failure_removes_partial_files() {
        let f = fixture(Some("Q4_K_M"));
        let mut runner = MockRunner::new(VALID);
        runner.fail_call = Some(1);
        let err = export_with_path(&f.cfg, &runner, None).unwrap_err();
        assert!(matches!(err, EvalError::CommandFailed { ref step, .. } if step == "quantize"));
        assert!(!f.cfg.output_gguf.exists());
        assert!(!f.cfg.output_gguf.with_extension("f16.gguf").exists());
    }

    #[test]
    fn final_file_must_be_non_empty_gguf() {
        for (content, reason) in [
            (&b""[..], "vide"),
            (b"PK\x03\x04zip", "signature"),
            (b"GG", "signature"),
        ] {
            let f = fixture(Some("Q4_K_M"));
            let err = export_with_path(&f.cfg, &MockRunner::new(content), None).unwrap_err();
            assert!(
                matches!(&err, EvalError::InvalidGguf { reason: r, .. } if r.contains(reason)),
                "{err:?}"
            );
        }
    }

    #[test]
    fn invalid_inputs_are_refused_before_any_command() {
        type Mutation = fn(&mut GgufExportConfig);
        let cases: [(&str, Mutation); 6] = [
            ("injection", |c| c.quant = Some("Q4_K_M; rm -rf /".into())),
            ("unknown quant", |c| c.quant = Some("Q9_X".into())),
            ("outtype", |c| c.intermediate_type = "q8_0".into()),
            ("python", |c| c.python_bin = " ".into()),
            ("model dir", |c| c.model_dir.push("absent")),
            ("output dir", |c| {
                c.output_gguf = c.model_dir.join("absent/out.gguf")
            }),
        ];
        for (name, mutate) in cases {
            let mut f = fixture(Some("Q4_K_M"));
            mutate(&mut f.cfg);
            let runner = MockRunner::new(VALID);
            assert!(export_with_path(&f.cfg, &runner, None).is_err(), "{name}");
            assert!(runner.calls().is_empty(), "{name}: no command may run");
        }
        let mut f = fixture(Some("Q4_K_M; rm -rf /"));
        assert!(matches!(
            export_with_path(&f.cfg, &MockRunner::new(VALID), None),
            Err(EvalError::InvalidQuant(_))
        ));

        f.cfg.quant = Some("Q4_K_M".into());
        std::fs::remove_file(&f.quantize).unwrap();
        let runner = MockRunner::new(VALID);
        let err = export_with_path(&f.cfg, &runner, None).unwrap_err();
        assert!(
            matches!(err, EvalError::BinaryNotFound { ref name, .. } if name == "llama-quantize")
        );
        assert!(
            runner.calls().is_empty(),
            "missing quantize detected before converting"
        );

        std::fs::remove_file(f.cfg.llama_cpp_dir.join("convert_hf_to_gguf.py")).unwrap();
        f.cfg.quant = None;
        let err = export_with_path(&f.cfg, &runner, None).unwrap_err();
        assert!(
            matches!(err, EvalError::BinaryNotFound { ref name, .. } if name == "convert_hf_to_gguf.py")
        );
        assert!(runner.calls().is_empty());
    }

    fn report(verdict: EvalVerdict) -> EvalReport {
        EvalReport {
            results: vec![PromptResult {
                id: "p".into(),
                status: PromptStatus::Pass,
                output_excerpt: String::new(),
                chars_generated: 0,
                duration_ms: 0,
            }],
            passed: 1,
            failed: 0,
            verdict,
        }
    }

    #[test]
    fn export_if_healthy_gates_on_verdict() {
        let f = fixture(Some("Q4_K_M"));
        let runner = MockRunner::new(VALID);
        for (verdict, allow) in [
            (EvalVerdict::Broken, true),
            (EvalVerdict::Broken, false),
            (EvalVerdict::Degraded, false),
        ] {
            assert!(export_if_healthy(&report(verdict), &f.cfg, &runner, allow)
                .unwrap()
                .is_none());
        }
        let mut empty = report(EvalVerdict::Healthy);
        empty.results.clear();
        assert!(export_if_healthy(&empty, &f.cfg, &runner, true)
            .unwrap()
            .is_none());
        assert!(runner.calls().is_empty());

        assert!(
            export_if_healthy(&report(EvalVerdict::Degraded), &f.cfg, &runner, true)
                .unwrap()
                .is_some()
        );
        assert!(
            export_if_healthy(&report(EvalVerdict::Healthy), &f.cfg, &runner, false)
                .unwrap()
                .is_some()
        );
        assert_eq!(runner.calls().len(), 4);
    }

    /// The temporary GGUF is a guard: unique name in the temp dir (two runs never share a
    /// file), deleted on drop, and deleted too when the conversion fails halfway.
    #[test]
    fn the_temporary_gguf_is_unique_in_the_temp_dir_and_removed_on_drop_and_error() {
        let f = fixture(None);
        let runner = MockRunner::new(VALID);
        let (llama, model) = (&f.cfg.llama_cpp_dir, &f.cfg.model_dir);
        let first = ensure_gguf(model, Some(llama), "guard", &runner).unwrap();
        let second = ensure_gguf(model, Some(llama), "guard", &runner).unwrap();
        for g in [&first, &second] {
            assert!(g.is_temporary() && g.path().is_file());
            assert_eq!(g.path().parent(), Some(std::env::temp_dir().as_path()));
            let name = g.path().file_name().unwrap().to_string_lossy().into_owned();
            let own = format!("weight-surgeon-guard-{}-", std::process::id());
            assert!(name.starts_with(&own) && name.ends_with(".gguf"), "{name}");
        }
        assert_ne!(first.path(), second.path());
        let (a, b) = (first.path().to_path_buf(), second.path().to_path_buf());
        drop(first);
        assert!(!a.exists() && b.exists(), "only the dropped one goes");
        drop(second);
        assert!(!b.exists());

        // A failing conversion leaves nothing behind either (the guard exists before it).
        let failing = MockRunner {
            fail_call: Some(0),
            ..MockRunner::new(VALID)
        };
        assert!(ensure_gguf(model, Some(llama), "guard", &failing).is_err());
        let outfile = failing.calls()[0]
            .1
            .iter()
            .position(|a| a == "--outfile")
            .unwrap();
        assert!(!Path::new(&failing.calls()[0].1[outfile + 1]).exists());

        // A .gguf given as is is never a temporary file, so never deleted.
        let own = f.cfg.model_dir.join("m.gguf");
        std::fs::write(&own, VALID).unwrap();
        drop(ensure_gguf(&own, None, "guard", &runner).unwrap());
        assert!(own.exists());
    }
}
