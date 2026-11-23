use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileIdentity {
    pub volume: u64,
    pub index: u64,
    pub links: u64,
}

impl FileIdentity {
    pub fn key(&self) -> (u64, u64) {
        (self.volume, self.index)
    }
}

#[cfg(unix)]
pub fn file_identity(path: &Path) -> io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path)?;
    Ok(FileIdentity {
        volume: meta.dev(),
        index: meta.ino(),
        links: meta.nlink(),
    })
}

#[cfg(windows)]
pub fn file_identity(path: &Path) -> io::Result<FileIdentity> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let file = std::fs::File::open(path)?;
    let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
    // SAFETY: `file` keeps the handle open for the whole call and `info` is a writable,
    // correctly sized `BY_HANDLE_FILE_INFORMATION`.
    let ok = unsafe { GetFileInformationByHandle(file.as_raw_handle(), info.as_mut_ptr()) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call succeeded, so it filled the structure (zeroed beforehand).
    let info = unsafe { info.assume_init() };
    Ok(FileIdentity {
        volume: u64::from(info.dwVolumeSerialNumber),
        index: (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        links: u64::from(info.nNumberOfLinks),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hard_link_shares_the_identity_and_counts_two_links() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let (a, b, c) = (
            dir.path().join("a"),
            dir.path().join("b"),
            dir.path().join("c"),
        );
        std::fs::write(&a, b"weights")?;
        std::fs::hard_link(&a, &b)?;
        std::fs::write(&c, b"weights")?;
        let (ia, ib, ic) = (file_identity(&a)?, file_identity(&b)?, file_identity(&c)?);
        assert_eq!(ia.key(), ib.key());
        assert_ne!(ia.key(), ic.key());
        assert_eq!((ia.links, ic.links), (2, 1));
        Ok(())
    }
}
