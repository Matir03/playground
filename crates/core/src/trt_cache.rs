//! Content-keyed cache paths for TensorRT engines.
//!
//! TensorRT names a cached engine after the *graph* it compiled, not the weights
//! in it. Every checkpoint of a given architecture therefore collides: the first
//! engine built for a shape is silently reused for every later model loaded from
//! the same cache directory.
//!
//! That turns a hot-reload into a no-op, which is very hard to see from the
//! outside — the process loads the new file, reports no error, and keeps playing
//! the old net. It cost a 15-hour Hex run: self-play generated from the net
//! current at startup while the champion advanced through five promotions, and
//! gate matches "compared" two checkpoints that were both executing a third
//! one's weights.
//!
//! Keying the directory by the model's bytes fixes it: identical models share an
//! engine, different ones each build their own.
//!
//! To check whether some other backend has the same problem, fingerprint it —
//! swap a different net into the same path and compare the outputs. Identical
//! numbers mean you are running a stale engine.

use std::hash::{Hash, Hasher};
use std::path::Path;

/// `<cache_dir>/<hash of the model file>`, ready to hand to an engine cache.
pub fn content_keyed(cache_dir: &str, model_path: impl AsRef<Path>) -> anyhow::Result<String> {
    let path = model_path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("reading {} to key its engine cache: {e}", path.display()))?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    Ok(format!("{cache_dir}/{:016x}", h.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_share_a_directory_and_different_bytes_do_not() {
        let dir = std::env::temp_dir().join(format!("trtkey-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, c) = (dir.join("a"), dir.join("b"), dir.join("c"));
        std::fs::write(&a, b"weights-one").unwrap();
        std::fs::write(&b, b"weights-one").unwrap();
        std::fs::write(&c, b"weights-two").unwrap();

        let ka = content_keyed("/cache", &a).unwrap();
        let kb = content_keyed("/cache", &b).unwrap();
        let kc = content_keyed("/cache", &c).unwrap();
        assert_eq!(ka, kb, "identical models must reuse one engine");
        assert_ne!(ka, kc, "different models must not share an engine");

        // And the same path with new contents keys differently — the hot-reload
        // case that made this necessary.
        std::fs::write(&a, b"weights-two").unwrap();
        assert_eq!(content_keyed("/cache", &a).unwrap(), kc);
        std::fs::remove_dir_all(&dir).ok();
    }
}
