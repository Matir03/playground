//! Guards the one constant that is hand-duplicated across languages.
//!
//! The example record's byte layout is written out independently in
//! `crates/games/minions/src/selfplay.rs` and `python/minions_az_learner.py`,
//! and a mismatch does not fail loudly: the learner reshapes the body by its
//! own stride, so a wrong size either raises a confusing byte-count error or,
//! worse, silently reads every field at the wrong offset and trains on noise.
//!
//! `scripts/check-record-layout.sh` compares the two sides directly. This test
//! is the cheap half of that: it pins the Rust value so a change to the layout
//! has to be deliberate, and names the files that must change with it.

#[test]
fn example_bytes_and_version_are_pinned() {
    // Update these together with, and only with:
    //   * python/minions_az_learner.py  (EXAMPLE_BYTES, VERSION, the O_* offsets)
    //   * python/minions_watchdog.py    (if it reads the new field)
    // then run scripts/check-record-layout.sh.
    assert_eq!(
        minions_engine::selfplay::EXAMPLE_BYTES,
        2660,
        "record size changed; update the Python mirror and the offsets with it"
    );
    assert_eq!(
        minions_engine::selfplay::VERSION,
        8,
        "record version changed; the learner's COMPAT_VERSIONS must decide what \
         to do with chunks still in the sliding window"
    );
}
