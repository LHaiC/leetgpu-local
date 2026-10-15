# Vendored CodeMirror 5

The code editor is [CodeMirror 5](https://codemirror.net/5/) (MIT), vendored here
so the app needs no package manager, no build step and no network access at
runtime. `scripts/update_codemirror.sh` re-downloads exactly these files.

- Version: **5.65.16**
- Source: `https://cdn.jsdelivr.net/npm/codemirror@5.65.16/`
- License: see `LICENSE` (MIT, © Marijn Haverbeke and others)

## Why vendored instead of a git submodule

The app is served straight from `web/` with zero setup (`clone → run`). A
submodule would need `git submodule update --init` before the editor works, and
the upstream repository ships every mode/theme/addon (tens of MB) where we need
twelve files. Vendoring keeps the zero-config flow and is still reproducible
via the pinned version plus the update script.

## Files in use

| File | Purpose |
| --- | --- |
| `lib/codemirror.js`, `lib/codemirror.css` | core editor |
| `keymap/vim.js` | real Vim keybindings (visual/visual-line/visual-block, registers, macros, text objects, `:` ex commands, `/` search) |
| `mode/clike/clike.js` | C/C++ base mode; CUDA keywords are injected in `web/editor-setup.js` |
| `mode/python/python.js` | Triton / PyTorch solutions |
| `addon/selection/active-line.js` | highlight the caret's line |
| `addon/edit/matchbrackets.js` | bracket matching |
| `addon/edit/closebrackets.js` | auto-close brackets |
| `addon/comment/comment.js` | toggle line comment |
| `addon/search/searchcursor.js`, `addon/dialog/dialog.js`, `dialog.css` | `/` and `:` search prompts used by the Vim keymap |

Indentation guides are a small local overlay in `web/editor-setup.js` (CodeMirror
5 has no built-in guides).
