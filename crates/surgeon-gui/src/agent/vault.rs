//! API keys: in the system vault (Windows Credential Manager), else for this session only
//! (memory, or an environment variable), never in a file of the app. A [`Secret`] cannot be
//! printed, serialized or compared by accident: `Debug` shows `***` and there is no
//! `Display` nor `Serialize`.

use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The value, for the HTTP header only.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl Clone for Secret {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    /// Windows Credential Manager: survives restarts, encrypted for the user account.
    System,
    /// Memory or an environment variable: gone when the app quits.
    Session,
}

pub trait SecretStore: Send + Sync {
    fn kind(&self) -> StoreKind;
    fn get(&self, id: &str) -> Result<Option<Secret>, String>;
    fn set(&self, id: &str, secret: &Secret) -> Result<(), String>;
    fn delete(&self, id: &str) -> Result<(), String>;
}

/// The environment variable a session key may come from: `SURGEON_AGENT_KEY_<ID>`, the id
/// upper-cased with `-` and `.` as `_`.
pub fn env_name(id: &str) -> String {
    let id: String = id
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() {
            true => c.to_ascii_uppercase(),
            false => '_',
        })
        .collect();
    format!("SURGEON_AGENT_KEY_{id}")
}

/// Keys typed during this session, then the environment variable.
#[derive(Default)]
pub struct SessionStore {
    keys: Mutex<BTreeMap<String, Secret>>,
}

impl SecretStore for SessionStore {
    fn kind(&self) -> StoreKind {
        StoreKind::Session
    }

    fn get(&self, id: &str) -> Result<Option<Secret>, String> {
        let typed = self
            .keys
            .lock()
            .map_err(|e| e.to_string())?
            .get(id)
            .cloned();
        Ok(typed.or_else(|| {
            std::env::var(env_name(id))
                .ok()
                .filter(|v| !v.is_empty())
                .map(Secret::new)
        }))
    }

    fn set(&self, id: &str, secret: &Secret) -> Result<(), String> {
        self.keys
            .lock()
            .map_err(|e| e.to_string())?
            .insert(id.into(), secret.clone());
        Ok(())
    }

    fn delete(&self, id: &str) -> Result<(), String> {
        self.keys.lock().map_err(|e| e.to_string())?.remove(id);
        Ok(())
    }
}

/// The system vault when there is one, else the session store.
pub fn system() -> Box<dyn SecretStore> {
    #[cfg(windows)]
    {
        Box::new(windows::CredentialStore)
    }
    #[cfg(not(windows))]
    {
        // shortcut: no Keychain / Secret Service binding yet, keys live for the session;
        // add one (keyring crate) when a non-Windows build ships.
        Box::new(SessionStore::default())
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    //! `CredWriteW` / `CredReadW` / `CredDeleteW` on generic credentials named
    //! `LLM-Surgeon/agent/<id>`, persisted for the local machine and user.
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_NOT_FOUND, FILETIME};
    use windows_sys::Win32::Security::Credentials::{
        CredDeleteW, CredFree, CredReadW, CredWriteW, CREDENTIALW, CRED_PERSIST_LOCAL_MACHINE,
        CRED_TYPE_GENERIC,
    };

    use super::{Secret, SecretStore, StoreKind};

    pub struct CredentialStore;

    fn target(id: &str) -> Vec<u16> {
        format!("LLM-Surgeon/agent/{id}")
            .encode_utf16()
            .chain(Some(0))
            .collect()
    }

    impl SecretStore for CredentialStore {
        fn kind(&self) -> StoreKind {
            StoreKind::System
        }

        fn get(&self, id: &str) -> Result<Option<Secret>, String> {
            let name = target(id);
            let mut cred: *mut CREDENTIALW = std::ptr::null_mut();
            // SAFETY: `name` is NUL-terminated and outlives the call; on success `cred`
            // points to a block owned by the system, read once below and freed by CredFree.
            let ok = unsafe { CredReadW(name.as_ptr(), CRED_TYPE_GENERIC, 0, &mut cred) };
            if ok == 0 {
                // SAFETY: no argument; reads the calling thread's last error.
                let err = unsafe { GetLastError() };
                return match err {
                    ERROR_NOT_FOUND => Ok(None),
                    e => Err(format!("Gestionnaire d'identifiants : erreur {e}")),
                };
            }
            // SAFETY: CredReadW succeeded, so `cred` is valid and its blob holds
            // `CredentialBlobSize` bytes; both are released by CredFree right after.
            let bytes = unsafe {
                let c = &*cred;
                let blob =
                    std::slice::from_raw_parts(c.CredentialBlob, c.CredentialBlobSize as usize)
                        .to_vec();
                CredFree(cred as *const _);
                blob
            };
            String::from_utf8(bytes)
                .map(|s| Some(Secret::new(s)))
                .map_err(|_| "Gestionnaire d'identifiants : clé illisible".into())
        }

        fn set(&self, id: &str, secret: &Secret) -> Result<(), String> {
            let mut name = target(id);
            let mut user: Vec<u16> = "surgeon-gui".encode_utf16().chain(Some(0)).collect();
            let mut blob = secret.expose().as_bytes().to_vec();
            let cred = CREDENTIALW {
                Flags: 0,
                Type: CRED_TYPE_GENERIC,
                TargetName: name.as_mut_ptr(),
                Comment: std::ptr::null_mut(),
                LastWritten: FILETIME {
                    dwLowDateTime: 0,
                    dwHighDateTime: 0,
                },
                CredentialBlobSize: blob.len() as u32,
                CredentialBlob: blob.as_mut_ptr(),
                Persist: CRED_PERSIST_LOCAL_MACHINE,
                AttributeCount: 0,
                Attributes: std::ptr::null_mut(),
                TargetAlias: std::ptr::null_mut(),
                UserName: user.as_mut_ptr(),
            };
            // SAFETY: every pointer in `cred` points into a local buffer alive for the call.
            let ok = unsafe { CredWriteW(&cred, 0) };
            blob.fill(0);
            match ok {
                0 => Err(format!(
                    "Gestionnaire d'identifiants : écriture refusée (erreur {})",
                    // SAFETY: no argument.
                    unsafe { GetLastError() }
                )),
                _ => Ok(()),
            }
        }

        fn delete(&self, id: &str) -> Result<(), String> {
            let name = target(id);
            // SAFETY: `name` is NUL-terminated and outlives the call.
            let ok = unsafe { CredDeleteW(name.as_ptr(), CRED_TYPE_GENERIC, 0) };
            if ok == 0 {
                // SAFETY: no argument.
                let err = unsafe { GetLastError() };
                if err != ERROR_NOT_FOUND {
                    return Err(format!("Gestionnaire d'identifiants : erreur {err}"));
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_never_prints() {
        let s = Secret::new("sk-ant-api-very-secret");
        assert_eq!(format!("{s:?}"), "Secret(***)");
        assert_eq!(env_name("open-ai.2"), "SURGEON_AGENT_KEY_OPEN_AI_2");
    }

    #[test]
    fn the_session_store_forgets_on_delete() {
        let store = SessionStore::default();
        store.set("t", &Secret::new("k")).unwrap();
        assert_eq!(store.get("t").unwrap().unwrap().expose(), "k");
        store.delete("t").unwrap();
        assert!(store.get("t").unwrap().is_none());
    }

    /// Writes, reads and deletes a real credential (a throwaway name).
    #[cfg(windows)]
    #[test]
    fn the_windows_vault_round_trips() {
        let store = system();
        let id = format!("selftest-{}", std::process::id());
        store.set(&id, &Secret::new("valeur-de-test")).unwrap();
        assert_eq!(store.get(&id).unwrap().unwrap().expose(), "valeur-de-test");
        store.delete(&id).unwrap();
        assert!(store.get(&id).unwrap().is_none());
        store.delete(&id).unwrap();
    }
}
