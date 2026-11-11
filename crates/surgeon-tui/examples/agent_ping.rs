//! Example: an agent client for the bridge. Send one command to a running
//! surgeon-tui from any script:
//!
//! ```text
//! cargo run -p surgeon-tui --example agent_ping -- highlight 14
//! cargo run -p surgeon-tui --example agent_ping -- preview blk.14.ffn_down.weight 0.5
//! cargo run -p surgeon-tui --example agent_ping -- snapshot
//! ```

use surgeon_tui::ipc::IpcTarget;

#[tokio::main]
async fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = raw.iter().map(String::as_str).collect();
    // An optional leading argument is the endpoint; the rest is the command.
    let (endpoint, command_args): (Option<&str>, &[&str]) = match args.as_slice() {
        [] => (None, &[]),
        [only] if !matches!(*only, "highlight" | "select" | "preview" | "filter" | "clear" | "snapshot") => {
            (Some(only), &[])
        }
        _ => (None, &args[..]),
    };
    let target = endpoint.map_or_else(IpcTarget::default_target, IpcTarget::parse);
    let Some(command) = build(command_args) else {
        eprintln!(
            "usage: agent_ping [ENDPOINT] (highlight N | select NAME | preview TENSOR ALPHA | filter ROLE | clear | snapshot)"
        );
        return;
    };
    match surgeon_tui::ipc::send_commands(&target, &[command]).await {
        Ok(replies) => {
            for reply in replies {
                println!("{reply}");
            }
        }
        Err(e) => eprintln!("bridge {} injoignable : {e}", target.display()),
    }
}

/// Builds one bridge command from the command-line words; `None` on bad usage.
fn build(args: &[&str]) -> Option<serde_json::Value> {
    match args {
        ["highlight", layer] => Some(serde_json::json!({
            "cmd": "highlight_layer",
            "layer": layer.parse::<u64>().ok()?
        })),
        ["select", name] => Some(serde_json::json!({"cmd": "select_tensor", "name": name})),
        ["preview", tensor, alpha] => Some(serde_json::json!({
            "cmd": "preview_patch",
            "tensor": tensor,
            "alpha": alpha.parse::<f64>().ok()?
        })),
        ["filter", role] => Some(serde_json::json!({"cmd": "set_filter", "role": role})),
        ["clear"] => Some(serde_json::json!({"cmd": "clear_preview"})),
        ["snapshot"] => Some(serde_json::json!({"cmd": "snapshot"})),
        ["ping"] => Some(serde_json::json!({"cmd": "ping"})),
        _ => None,
    }
}
