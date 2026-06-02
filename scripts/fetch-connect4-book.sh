#!/usr/bin/env bash
# Download Pascal Pons' Connect 4 opening book used by the perfect engine.
#
# The book (~32 MB, depth 14) lets the engine play the opening instantly and
# perfectly; without it the solver still plays perfectly but is very slow in
# the first ~14 plies. It is gitignored (large binary asset), so fetch it once.
#
# Source: https://github.com/PascalPons/connect4 (AGPL-3.0), release "book".
set -euo pipefail

dest="$(dirname "$0")/../crates/games/connect4/data/7x6.book"
url="https://github.com/PascalPons/connect4/releases/download/book/7x6.book"

mkdir -p "$(dirname "$dest")"
if [[ -f "$dest" ]]; then
  echo "Opening book already present at $dest"
  exit 0
fi

echo "Downloading opening book to $dest ..."
curl -fL -o "$dest" "$url"
echo "Done ($(du -h "$dest" | cut -f1))."
