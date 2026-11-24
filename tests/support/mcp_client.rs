//! A `surgeon mcp` client for the tests. Closing stdin is the client's shutdown: the server
//! cancels the running call and answers nothing more, so stdin closes only once every request
//! has its reply.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ExitStatus};

use serde_json::Value;

/// Sends `lines` to `child` (stdin and stdout piped), reads stdout until every request among
/// them is answered, then closes stdin and waits: what stdout held (replies and
/// notifications, in order) and the exit status.
pub fn exchange(child: &mut Child, lines: &[Value]) -> (Vec<Value>, ExitStatus) {
    let mut stdin = child.stdin.take().expect("piped stdin");
    for line in lines {
        writeln!(stdin, "{line}").unwrap();
    }
    stdin.flush().unwrap();
    // A message with an id gets a reply, unless it is itself one; notifications get none.
    let mut pending: Vec<Value> = lines
        .iter()
        .filter(|l| l.get("result").is_none() && l.get("error").is_none())
        .filter_map(|l| l.get("id").cloned())
        .collect();
    let parse = |line: std::io::Result<String>| -> Value {
        let line = line.unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("not JSON-RPC ({e}): {line:?}"))
    };
    let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout")).lines();
    let mut read = Vec::new();
    while !pending.is_empty() {
        let Some(line) = stdout.next() else {
            break;
        };
        let value = parse(line);
        pending.retain(|id| value.get("id") != Some(id));
        read.push(value);
    }
    drop(stdin);
    let status = child.wait().unwrap();
    read.extend(stdout.map(parse));
    (read, status)
}
