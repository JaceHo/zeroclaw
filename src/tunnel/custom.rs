use super::{kill_shared, new_shared_process, SharedProcess, Tunnel, TunnelProcess};
use anyhow::{bail, Result};
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

/// Shell metacharacters that could enable command injection.
const DANGEROUS_CHARS: &[char] = &['|', ';', '&', '$', '`', '(', ')', '<', '>'];

/// Allowed tunnel binary base names. Commands must start with one of these.
const ALLOWED_TUNNEL_BINARIES: &[&str] = &[
    "cloudflared",
    "ngrok",
    "bore",
    "ssh",
    "frp",
    "frpc",
    "tailscale",
    "localtunnel",
    "lt",
    "zrok",
    "rathole",
    "chisel",
    "pagekite",
];

/// Validate a custom tunnel `start_command` for safety.
///
/// Rejects commands that:
/// - Are empty or whitespace-only
/// - Contain shell metacharacters (`|;&$`()<>`)
/// - Start with a binary not in the known tunnel allowlist
///
/// This is called both at config validation time and at `CustomTunnel`
/// construction as defense-in-depth.
pub fn validate_start_command(start_command: &str) -> Result<()> {
    let cmd = start_command.trim();

    if cmd.is_empty() {
        bail!("tunnel.custom.start_command must not be empty");
    }

    // Block shell metacharacters
    if cmd.contains(DANGEROUS_CHARS) {
        bail!(
            "tunnel.custom.start_command contains shell metacharacters (|;&$`()<>); \
             use a wrapper script if complex shell logic is needed"
        );
    }

    // Extract the binary name (first whitespace-delimited token, basename only)
    let binary_token = cmd.split_whitespace().next().unwrap_or("");
    let binary_name = std::path::Path::new(binary_token)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(binary_token);

    if !ALLOWED_TUNNEL_BINARIES.contains(&binary_name) {
        bail!(
            "tunnel.custom.start_command must start with a known tunnel binary \
             ({}), got \"{binary_name}\". \
             If you need a different binary, wrap it in an allowed tunnel script.",
            ALLOWED_TUNNEL_BINARIES.join(", ")
        );
    }

    Ok(())
}

/// Custom Tunnel — bring your own tunnel binary.
///
/// Provide a `start_command` with `{port}` and `{host}` placeholders.
/// Optionally provide a `url_pattern` regex to extract the public URL
/// from stdout, and a `health_url` to poll for liveness.
///
/// Only known tunnel binaries are permitted as the command's first token
/// (see [`ALLOWED_TUNNEL_BINARIES`]). Shell metacharacters are rejected.
///
/// Examples:
/// - `bore local {port} --to bore.pub`
/// - `frp -c /etc/frp/frpc.ini`
/// - `ssh -R 80:localhost:{port} serveo.net`
pub struct CustomTunnel {
    start_command: String,
    health_url: Option<String>,
    url_pattern: Option<String>,
    proc: SharedProcess,
}

impl CustomTunnel {
    /// Create a new `CustomTunnel`, validating the `start_command` at
    /// construction time. Returns an error if the command is empty,
    /// contains shell metacharacters, or uses a disallowed binary.
    pub fn new(
        start_command: String,
        health_url: Option<String>,
        url_pattern: Option<String>,
    ) -> Result<Self> {
        validate_start_command(&start_command)?;
        Ok(Self {
            start_command,
            health_url,
            url_pattern,
            proc: new_shared_process(),
        })
    }
}

#[async_trait::async_trait]
impl Tunnel for CustomTunnel {
    fn name(&self) -> &str {
        "custom"
    }

    async fn start(&self, local_host: &str, local_port: u16) -> Result<String> {
        // Defense-in-depth: re-validate even though `new()` already checked.
        // Placeholder substitution could theoretically introduce metacharacters
        // if local_host were attacker-controlled, so validate the final command.
        let cmd = self
            .start_command
            .replace("{port}", &local_port.to_string())
            .replace("{host}", local_host);

        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            bail!("Custom tunnel start_command is empty");
        }

        // Re-check the expanded command for metacharacters
        if cmd.contains(DANGEROUS_CHARS) {
            bail!(
                "Custom tunnel start_command contains shell metacharacters after \
                 placeholder expansion — this is not allowed"
            );
        }

        let mut child = Command::new(parts[0])
            .args(&parts[1..])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        let mut public_url = format!("http://{local_host}:{local_port}");

        // If a URL pattern is provided, try to extract the public URL from stdout
        if let Some(ref pattern) = self.url_pattern {
            if let Some(stdout) = child.stdout.take() {
                let mut reader = tokio::io::BufReader::new(stdout).lines();
                let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(15);

                while tokio::time::Instant::now() < deadline {
                    let line = tokio::time::timeout(
                        tokio::time::Duration::from_secs(3),
                        reader.next_line(),
                    )
                    .await;

                    match line {
                        Ok(Ok(Some(l))) => {
                            tracing::debug!("custom-tunnel: {l}");
                            // Simple substring match on the pattern
                            if l.contains(pattern)
                                || l.contains("https://")
                                || l.contains("http://")
                            {
                                // Extract URL from the line
                                if let Some(idx) = l.find("https://") {
                                    let url_part = &l[idx..];
                                    let end = url_part
                                        .find(|c: char| c.is_whitespace())
                                        .unwrap_or(url_part.len());
                                    public_url = url_part[..end].to_string();
                                    break;
                                } else if let Some(idx) = l.find("http://") {
                                    let url_part = &l[idx..];
                                    let end = url_part
                                        .find(|c: char| c.is_whitespace())
                                        .unwrap_or(url_part.len());
                                    public_url = url_part[..end].to_string();
                                    break;
                                }
                            }
                        }
                        Ok(Ok(None) | Err(_)) => break,
                        Err(_) => {}
                    }
                }
            }
        }

        let mut guard = self.proc.lock().await;
        *guard = Some(TunnelProcess {
            child,
            public_url: public_url.clone(),
        });

        Ok(public_url)
    }

    async fn stop(&self) -> Result<()> {
        kill_shared(&self.proc).await
    }

    async fn health_check(&self) -> bool {
        // If a health URL is configured, try to reach it
        if let Some(ref url) = self.health_url {
            return crate::config::build_runtime_proxy_client("tunnel.custom")
                .get(url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .is_ok();
        }

        // Otherwise check if the process is still alive
        let guard = self.proc.lock().await;
        guard.as_ref().is_some_and(|tp| tp.child.id().is_some())
    }

    fn public_url(&self) -> Option<String> {
        self.proc
            .try_lock()
            .ok()
            .and_then(|g| g.as_ref().map(|tp| tp.public_url.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_start_command unit tests ─────────────────────

    #[test]
    fn validate_rejects_empty_command() {
        let err = validate_start_command("   ").unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn validate_rejects_metacharacters() {
        for bad in &[
            "bore local 8080 ; rm -rf /",
            "ngrok http 8080 | tee log",
            "bore local 8080 & bg",
            "bore local $(id)",
            "bore local `id`",
            "bore local 8080 > /tmp/x",
            "bore local 8080 < /dev/null",
            "ngrok http (8080)",
        ] {
            let err = validate_start_command(bad).unwrap_err();
            assert!(
                err.to_string().contains("shell metacharacters"),
                "expected metachar error for \"{bad}\", got: {err}"
            );
        }
    }

    #[test]
    fn validate_rejects_disallowed_binary() {
        let err = validate_start_command("curl http://evil.com").unwrap_err();
        assert!(
            err.to_string().contains("known tunnel binary"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn validate_rejects_arbitrary_path_binary() {
        let err = validate_start_command("/usr/bin/python3 -c 'import os'").unwrap_err();
        assert!(
            err.to_string().contains("known tunnel binary"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn validate_accepts_allowed_binaries() {
        for cmd in &[
            "bore local 8080 --to bore.pub",
            "ngrok http 8080",
            "cloudflared tunnel run",
            "ssh -R 80:localhost:8080 serveo.net",
            "frp -c /etc/frp/frpc.ini",
            "frpc -c /etc/frp/frpc.ini",
            "tailscale funnel 8080",
            "lt --port 8080",
            "localtunnel --port 8080",
            "zrok share public localhost:8080",
            "rathole client.toml",
            "chisel client https://example.com 8080",
            "pagekite 8080 mysite.pagekite.me",
        ] {
            assert!(
                validate_start_command(cmd).is_ok(),
                "expected Ok for \"{cmd}\""
            );
        }
    }

    #[test]
    fn validate_accepts_full_path_to_allowed_binary() {
        assert!(validate_start_command("/usr/local/bin/bore local 8080 --to bore.pub").is_ok());
        assert!(validate_start_command("/snap/bin/ngrok http 8080").is_ok());
    }

    #[test]
    fn new_rejects_disallowed_binary() {
        let result = CustomTunnel::new("curl http://evil.com".into(), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_metacharacters() {
        let result = CustomTunnel::new("bore local 8080 ; rm -rf /".into(), None, None);
        assert!(result.is_err());
    }

    // ── Integration tests (using allowed binaries via system utils) ──

    #[tokio::test]
    async fn health_check_with_unreachable_health_url_returns_false() {
        let tunnel = CustomTunnel::new(
            "bore local 8080".into(),
            Some("http://127.0.0.1:9/healthz".into()),
            None,
        )
        .unwrap();

        assert!(!tunnel.health_check().await);
    }
}
