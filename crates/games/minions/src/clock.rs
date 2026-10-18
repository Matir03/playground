//! A cheap clock for profiling counters. `Instant::now` (a vDSO
//! `clock_gettime`) is ~5% of tier-2 search time when every witness
//! candidate is timed (2026-10-17 profile); the TSC is a few cycles. This box's TSC
//! is constant and invariant (`constant_tsc`, `nonstop_tsc`), and the
//! counters only feed time shares, so ticks scaled by a one-off calibration
//! are plenty.

/// A point in time, for [`Tick::ns`].
#[derive(Clone, Copy, Debug)]
pub struct Tick(u64);

impl Tick {
    #[inline]
    pub fn now() -> Tick {
        Tick(raw())
    }

    /// Nanoseconds since `self`.
    #[inline]
    pub fn ns(self) -> u64 {
        (raw().wrapping_sub(self.0) as f64 * ns_per_tick()) as u64
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn raw() -> u64 {
    // SAFETY: RDTSC has no preconditions.
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn raw() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

fn ns_per_tick() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if cfg!(not(target_arch = "x86_64")) {
            return 1.0;
        }
        let (t0, r0) = (std::time::Instant::now(), raw());
        while t0.elapsed() < std::time::Duration::from_millis(20) {
            std::hint::spin_loop();
        }
        let (dt, dr) = (t0.elapsed().as_nanos() as f64, raw().wrapping_sub(r0) as f64);
        if dr > 0.0 {
            dt / dr
        } else {
            1.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticks scale to wall time within a few percent.
    #[test]
    fn tick_tracks_wall_time() {
        let (t, i) = (Tick::now(), std::time::Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (a, b) = (t.ns() as f64, i.elapsed().as_nanos() as f64);
        assert!((a / b - 1.0).abs() < 0.05, "tick {a} ns vs wall {b} ns");
    }
}
