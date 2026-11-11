//! End-to-end bridge tests: a real server on a real endpoint (Unix socket or
//! Windows named pipe), JSON-lines commands from a client, one reply per
//! command, snapshot answered by the app side of the channel.

use serde_json::{json, Value};
use surgeon_tui::ipc::{IpcCommand, IpcRequest, IpcTarget};

/// A unique endpoint for this test run. The timestamp alone is not enough on
/// Windows (clock resolution ~15 ms can collide across parallel tests), so a
/// per-call counter disambiguates.
fn test_target() -> IpcTarget {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let serial = CALLS.fetch_add(1, Ordering::Relaxed);
    if cfg!(windows) {
        IpcTarget::WindowsPipe(format!(
            "surgeon-tui-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
            serial
        ))
    } else {
        let dir = tempfile::tempdir().expect("tempdir");
        // Leak the tempdir: the socket file must outlive the test's server.
        IpcTarget::Unix(dir.keep().join("t.sock"))
    }
}

/// Runs the server task and returns the receiving end of its command channel.
async fn spawn_server(target: IpcTarget) -> tokio::sync::mpsc::UnboundedReceiver<IpcRequest> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<IpcRequest>();
    tokio::spawn(async move {
        let _ = surgeon_tui::ipc::serve(target, tx).await;
    });
    rx
}

/// Retries until the server accepts connections, then consumes the ping the
/// connect probe forwarded to `rx` (deterministic drain).
async fn connect(target: &IpcTarget, rx: &mut tokio::sync::mpsc::UnboundedReceiver<IpcRequest>) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if surgeon_tui::ipc::send_commands(target, &[json!({"cmd": "ping"})])
            .await
            .is_ok()
        {
            break;
        }
        if std::time::Instant::now() > deadline {
            panic!("le serveur n'écoute pas sur {}", target.display());
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let request = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("event")
        .expect("channel");
    assert_eq!(request.command.name(), "ping");
    assert!(request.reply.is_none());
}

#[tokio::test]
async fn commands_get_one_ack_each_and_reach_the_app() {
    let target = test_target();
    let mut rx = spawn_server(target.clone()).await;
    connect(&target, &mut rx).await;

    let replies = surgeon_tui::ipc::send_commands(
        &target,
        &[
            json!({"cmd": "ping"}),
            json!({"cmd": "highlight_layer", "layer": 14}),
            json!({"cmd": "preview_patch", "tensor": "blk.14.ffn_down.weight", "alpha": 0.5}),
            json!({"cmd": "set_filter", "role": "gate"}),
            json!({"cmd": "clear_preview"}),
            json!({"cmd": "set_filter"}),
        ],
    )
    .await
    .expect("exchange");
    assert_eq!(replies.len(), 6);
    assert_eq!(replies[0], json!({"ok": true, "pong": true}));
    for reply in &replies[1..] {
        assert_eq!(reply["ok"], json!(true), "{reply}");
    }

    // The app side received every forwarded command, in order.
    let mut seen = Vec::new();
    for _ in 0..6 {
        let request = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("event")
            .expect("channel");
        seen.push(request.command.name().to_owned());
        assert!(request.reply.is_none());
    }
    assert_eq!(
        seen,
        vec![
            "ping",
            "highlight_layer",
            "preview_patch",
            "set_filter",
            "clear_preview",
            "set_filter"
        ]
    );
}

#[tokio::test]
async fn malformed_and_unknown_commands_are_errors_not_crashes() {
    let target = test_target();
    let mut rx = spawn_server(target.clone()).await;
    connect(&target, &mut rx).await;

    let replies = surgeon_tui::ipc::send_commands(
        &target,
        &[
            json!({"nope": true}),
            json!({"cmd": "select_tensor"}),
            json!({"cmd": "preview_patch", "tensor": "t", "alpha": "∞"}),
            json!({"cmd": "set_filter", "role": "quatrigeux"}),
            Value::Null,
        ],
    )
    .await
    .expect("exchange");
    assert_eq!(replies.len(), 5);
    // Every malformed command is a typed error...
    for reply in [&replies[0], &replies[1], &replies[2], &replies[4]] {
        assert_eq!(reply["ok"], json!(false), "{reply}");
        assert!(reply["error"].as_str().is_some(), "{reply}");
    }
    // ...but `set_filter` with an unknown role is protocol-valid: it is
    // forwarded and the app side logs the unknown role.
    assert_eq!(replies[3]["ok"], json!(true), "{}", replies[3]);
    let request = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("event")
        .expect("channel");
    assert_eq!(request.command.name(), "set_filter");
}

#[tokio::test]
async fn snapshot_goes_through_the_oneshot_reply() {
    let target = test_target();
    let mut rx = spawn_server(target.clone()).await;
    connect(&target, &mut rx).await;

    let commands = [json!({"cmd": "snapshot"})];
    let client = surgeon_tui::ipc::send_commands(&target, &commands);
    let (client_replies, command) = tokio::join!(client, async {
        let request = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("event")
            .expect("channel");
        // The app answers the snapshot right away; the client's reply line
        // only then completes.
        if let Some(sender) = request.reply {
            let _ = sender.send(json!({"tensor": "token_embd.weight", "role": "embed"}));
        }
        request.command
    });
    let replies = client_replies.expect("replies");
    assert_eq!(replies.len(), 1);
    assert_eq!(replies[0]["ok"], json!(true));
    assert_eq!(replies[0]["snapshot"]["tensor"], json!("token_embd.weight"));
    assert_eq!(command, IpcCommand::Snapshot);
}

#[tokio::test]
async fn multiple_commands_on_one_connection() {
    let target = test_target();
    let mut rx = spawn_server(target.clone()).await;
    connect(&target, &mut rx).await;

    let replies = surgeon_tui::ipc::send_commands(
        &target,
        &[
            json!({"cmd": "ping"}),
            json!({"cmd": "ping"}),
            json!({"cmd": "ping"}),
        ],
    )
    .await
    .expect("exchange");
    assert_eq!(replies.len(), 3);
    let mut pings = 0;
    while let Ok(request) = rx.try_recv() {
        assert_eq!(request.command.name(), "ping");
        pings += 1;
    }
    assert_eq!(pings, 3);
}

#[test]
fn the_protocol_parses_typical_agent_payloads() {
    let parsed =
        IpcCommand::from_value(&json!({"cmd": "highlight_layer", "layer": 3})).expect("highlight");
    assert_eq!(parsed, IpcCommand::HighlightLayer { layer: 3 });
    let parsed =
        IpcCommand::from_value(&json!({"cmd": "select_tensor", "name": "token_embd.weight"}))
            .expect("select");
    assert_eq!(parsed.name(), "select_tensor");
    // alpha must be finite; a float is accepted whatever its sign.
    assert!(
        IpcCommand::from_value(&json!({"cmd": "preview_patch", "tensor": "t", "alpha": -1.5}))
            .is_ok()
    );
    assert!(
        IpcCommand::from_value(&json!({"cmd": "preview_patch", "tensor": "t", "alpha": "x"}))
            .is_err()
    );
    // The role filter stays optional.
    assert!(matches!(
        IpcCommand::from_value(&json!({"cmd": "set_filter"})),
        Ok(IpcCommand::SetFilter { role: None })
    ));
}

#[test]
fn targets_resolve_their_display_form() {
    assert_eq!(
        IpcTarget::parse(r"\\.\pipe\surgeon-tui").display(),
        r"\\.\pipe\surgeon-tui"
    );
    assert_eq!(
        IpcTarget::parse("/tmp/other.sock").display(),
        "/tmp/other.sock"
    );
    assert_eq!(
        IpcTarget::default_target().display(),
        if cfg!(windows) {
            r"\\.\pipe\surgeon-tui".to_owned()
        } else {
            format!("{}/surgeon-tui.sock", std::env::temp_dir().display())
        }
    );
}
