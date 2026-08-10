#!/usr/bin/env bash
# Build MoHex 2.0 (benzene) — the reference strong Hex engine we play against.
#
# MoHex is the long-standing Computer-Olympiad-winning Hex program (MCTS +
# virtual connections + learned patterns). We use it the way go9 uses GNU Go:
# an external GTP opponent for absolute-strength eval, and a rules oracle to
# cross-check our own implementation.
#
# There are no prebuilt binaries, so this clones, patches (the 2015 source
# doesn't survive a modern toolchain — see scripts/patches/) and compiles.
# Boost and Berkeley DB are the dependencies; when the system lacks them we
# pull them into a micromamba env and bake its lib path into the binary's
# RPATH, so the built engine needs no environment setup to run. The default
# build supports up to 13×13, which is exactly the size we play.
#
# Source: https://github.com/cgao3/benzene-vanilla-cmake (LGPL-3.0).
#
# Installs to  ~/.local/src/benzene  (the tree must stay: the binary reads its
# pattern/book data from the source-relative share/ dir) and symlinks the
# `mohex` and `wolve` binaries into ~/.local/bin.
set -euo pipefail

src_dir="${BENZENE_SRC:-$HOME/.local/src/benzene}"
bin_dir="${BENZENE_BIN:-$HOME/.local/bin}"
repo="https://github.com/cgao3/benzene-vanilla-cmake.git"
jobs="${JOBS:-$(nproc)}"

if [[ -x "$src_dir/build/src/mohex/mohex" ]]; then
  echo "MoHex already built at $src_dir/build/src/mohex/mohex"
  exit 0
fi

# ── Source ───────────────────────────────────────────────────────────────────
if [[ -d "$src_dir/.git" ]]; then
  echo "==> Using existing checkout at $src_dir"
else
  echo "==> Cloning benzene to $src_dir"
  mkdir -p "$(dirname "$src_dir")"
  git clone --depth 1 "$repo" "$src_dir"
fi

# ── Dependencies: Boost + Berkeley DB ────────────────────────────────────────
# Boost is found by CMake (BOOST_ROOT); Berkeley DB is not — benzene just
# `#include <db.h>` and links `-ldb`, so the conda prefix goes on the
# compiler's CPATH/LIBRARY_PATH instead. Note CMakeLists.txt *overwrites*
# CMAKE_CXX_FLAGS, which is why the include path can't be passed as a define.
cmake_extra=()
prefix="${BENZENE_DEPS_PREFIX:-$HOME/micromamba/envs/benzene}"
need=()
[[ -f /usr/include/boost/version.hpp || -f "$prefix/include/boost/version.hpp" ]] || need+=(libboost-devel)
[[ -f /usr/include/db.h || -f "$prefix/include/db.h" ]] || need+=(libdb)

if ((${#need[@]})); then
  command -v micromamba >/dev/null 2>&1 \
    || { echo "missing ${need[*]} and no micromamba to install them" >&2; exit 1; }
  echo "==> Installing ${need[*]} into micromamba env $prefix"
  action=create
  [[ -d "$prefix/conda-meta" ]] && action=install
  micromamba "$action" -y -p "$prefix" -c conda-forge "${need[@]}"
fi

if [[ -d "$prefix" ]]; then
  echo "==> Using dependencies from $prefix"
  export CPATH="$prefix/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  cmake_extra+=(
    "-DCMAKE_EXE_LINKER_FLAGS=-L$prefix/lib -Wl,-rpath,$prefix/lib"
  )
  [[ -f "$prefix/include/boost/version.hpp" ]] \
    && cmake_extra+=("-DBOOST_ROOT=$prefix" "-DBoost_NO_SYSTEM_PATHS=ON")
fi

# ── Patch ────────────────────────────────────────────────────────────────────
# benzene stopped being maintained in 2015. Two fixes (see the patch header):
# Boost.Filesystem v2 calls that no longer exist, and a segfault in the
# resistance evaluation that MoHex hits several times per 13×13 game.
patch_file="$(cd "$(dirname "$0")" && pwd)/patches/benzene-modern-toolchain.patch"
if git -C "$src_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "==> Patch already applied"
else
  echo "==> Applying $patch_file"
  git -C "$src_dir" apply "$patch_file"
fi

# ── Build ────────────────────────────────────────────────────────────────────
mkdir -p "$src_dir/build"
cd "$src_dir/build"
cmake .. -DCMAKE_BUILD_TYPE=Release "${cmake_extra[@]}"
make "-j$jobs"

mkdir -p "$bin_dir"
for b in mohex wolve; do
  ln -sf "$src_dir/build/src/$b/$b" "$bin_dir/$b"
done

echo
echo "==> Built:"
echo "    $src_dir/build/src/mohex/mohex  (symlinked to $bin_dir/mohex)"
echo "Smoke test:  echo -e 'boardsize 13\\ngenmove b\\nquit' | mohex --quiet"
