#!/usr/bin/env bash
# Re-download the vendored CodeMirror 5 files (see web/vendor/codemirror/README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/web/vendor/codemirror"
VERSION="${CODEMIRROR_VERSION:-5.65.16}"
BASE="https://cdn.jsdelivr.net/npm/codemirror@${VERSION}"

FILES=(
  lib/codemirror.js
  lib/codemirror.css
  keymap/vim.js
  mode/clike/clike.js
  mode/python/python.js
  addon/selection/active-line.js
  addon/edit/matchbrackets.js
  addon/edit/closebrackets.js
  addon/comment/comment.js
  addon/search/searchcursor.js
  addon/dialog/dialog.js
  addon/dialog/dialog.css
  LICENSE
)

for file in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$file")"
  curl -fsSL -o "$DEST/$file" "$BASE/$file"
  echo "fetched $file"
done

echo "CodeMirror $VERSION vendored into web/vendor/codemirror"
