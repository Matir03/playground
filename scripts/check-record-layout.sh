#!/usr/bin/env bash
# Compare the Rust and Python views of the self-play record layout.
#
# The layout is hand-duplicated, and a mismatch fails silently rather than
# loudly: the learner reshapes chunk bodies by its own stride, so a disagreement
# either raises an opaque byte-count error or reads every field at the wrong
# offset and trains on noise. Run this after touching either side.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

py=$("$repo/python/.venv/bin/python" - <<'PY'
import importlib.util, sys
sys.path.insert(0, "python/src")
spec = importlib.util.spec_from_file_location("L", "python/minions_az_learner.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
print(L.EXAMPLE_BYTES, L.VERSION, L.TOP_K, L.CAND_BYTES)
PY
)
py_bytes=$(echo "$py" | cut -d' ' -f1)
py_ver=$(echo "$py" | cut -d' ' -f2)
py_topk=$(echo "$py" | cut -d' ' -f3)
py_cand=$(echo "$py" | cut -d' ' -f4)

# The pinned values live in the Rust test, which asserts them against the real
# constants -- so running the test is what makes reading them here trustworthy.
cargo test -q -p minions-engine --test record_layout >/dev/null 2>&1 || {
  echo "rust record_layout test FAILED: the pinned constants no longer match" >&2
  exit 1
}
read -r rs_bytes rs_ver rs_topk rs_cand < <(python3 - <<'PY'
import re
s = open("crates/games/minions/tests/record_layout.rs").read()
b = re.search(r"EXAMPLE_BYTES,\s*(\d+)", s).group(1)
v = re.search(r"VERSION,\s*(\d+)", s).group(1)
k = re.search(r"TOP_K,\s*(\d+)", s).group(1)
c = re.search(r"CAND_BYTES,\s*(\d+)", s).group(1)
print(b, v, k, c)
PY
)

echo "rust:   EXAMPLE_BYTES=$rs_bytes VERSION=$rs_ver TOP_K=$rs_topk CAND_BYTES=$rs_cand"
echo "python: EXAMPLE_BYTES=$py_bytes VERSION=$py_ver TOP_K=$py_topk CAND_BYTES=$py_cand"

fail=0
[[ "$rs_bytes" == "$py_bytes" ]] || { echo "MISMATCH: record size" >&2; fail=1; }
[[ "$rs_ver"   == "$py_ver"   ]] || { echo "MISMATCH: record version" >&2; fail=1; }
[[ "$rs_topk"  == "$py_topk"  ]] || { echo "MISMATCH: record capacity (TOP_K)" >&2; fail=1; }
[[ "$rs_cand"  == "$py_cand"  ]] || { echo "MISMATCH: candidate size" >&2; fail=1; }
[[ $fail -eq 0 ]] && echo "record layout agrees"
exit $fail
