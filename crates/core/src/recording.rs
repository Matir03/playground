//! Conversion from in-memory search trees to on-disk `GameRecord`s.
//!
//! The on-disk schema lives in `schemas/proto/playground/record`.
//! Actual proto↔in-memory conversion is implemented by `playground-protocol`,
//! which owns the generated types; this module keeps transport-independent
//! helpers (e.g. length-delimited writers, shard path layout).

use std::path::PathBuf;

/// Directory layout for recorded games.
/// `<records_dir>/<run_id>/<YYYY-MM-DD>/<record_id>.pb`
pub fn record_path(
    records_dir: &std::path::Path,
    run_id: &str,
    date: &str,
    record_id: &str,
) -> PathBuf {
    records_dir
        .join(run_id)
        .join(date)
        .join(format!("{record_id}.pb"))
}
