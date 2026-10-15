/* Minimal Vim keybinding for the <textarea> editor.
 * Modes: normal / insert / visual. Covers:
 *   motions: h j k l w b e 0 $ ^ gg G { } %
 *   operators: d y c (with motions), dd yy cc, D C Y, x, p P
 *   insert entry: i a o O I A, visual: v V, undo: u, redo: Ctrl-r
 *   counts: 3w 5j 2dd etc., r<char>, ESC to normal
 * Uses the host's execCommand-free text splicing; syncs editor history
 * hooks via input events.
 */
(function () {
  "use strict";

  const textarea = () => document.querySelector("#code-editor");
  const indicator = () => document.querySelector("#vim-mode");

  const state = {
    enabled: localStorage.getItem("leetgpu.vim") === "1",
    mode: "normal",
    count: "",
    op: null, // pending operator: 'd' | 'y' | 'c'
    opCount: 0,
    visualAnchor: 0,
    register: "",
    registerLinewise: false,
    awaitingG: false,
    lastFind: null,
  };

  function el() {
    return textarea();
  }

  function setMode(mode) {
    state.mode = mode;
    state.count = "";
    state.op = null;
    state.awaitingG = false;
    const ind = indicator();
    if (ind) {
      ind.textContent = state.enabled ? `-- ${mode.toUpperCase()} --` : "";
      ind.dataset.mode = mode;
      ind.classList.toggle("on", state.enabled);
    }
    const t = el();
    if (t) t.classList.toggle("vim-normal", state.enabled && mode !== "insert");
  }

  function pos() {
    return el().selectionStart;
  }

  function text() {
    return el().value;
  }

  function setCaret(i, extend = false) {
    const t = el();
    const n = t.value.length;
    i = Math.max(0, Math.min(i, n));
    if (extend) t.setSelectionRange(state.visualAnchor, i);
    else t.setSelectionRange(i, i);
  }

  function lineInfo(i) {
    const v = text();
    const start = v.lastIndexOf("\n", i - 1) + 1;
    const end = v.indexOf("\n", i);
    return { start, end: end === -1 ? v.length : end };
  }

  function col(i) {
    return i - lineInfo(i).start;
  }

  function clampCol(i, c) {
    const { start, end } = lineInfo(i);
    return Math.min(start + c, Math.max(start, end - 1));
  }

  // ---------- motions: return new cursor index ----------

  function motionH(i) { return Math.max(lineInfo(i).start, i - 1); }
  function motionL(i) { const { end } = lineInfo(i); return Math.min(i + 1, Math.max(lineInfo(i).start, end - 1)); }
  function motionJ(i) {
    const { end } = lineInfo(i);
    if (end >= text().length) return i;
    return clampCol(end + 1, col(i));
  }
  function motionK(i) {
    const { start } = lineInfo(i);
    if (start === 0) return i;
    return clampCol(start - 1, col(i));
  }

  function wordBoundaryForward(i) {
    const v = text();
    const cls = (ch) => (/\w/.test(ch) ? "w" : /\s/.test(ch) ? "s" : "p");
    while (i < v.length && /\s/.test(v[i])) i++;
    if (i >= v.length) return v.length;
    const c = cls(v[i]);
    while (i < v.length && cls(v[i]) === c) i++;
    while (i < v.length && /\s/.test(v[i])) i++;
    return i;
  }

  function wordForward(i) { // w
    const v = text();
    const cls = (ch) => (/\w/.test(ch) ? "w" : /\s/.test(ch) ? "s" : "p");
    if (i >= v.length) return i;
    const c = cls(v[i]);
    if (c !== "s") { while (i < v.length && cls(v[i]) === c) i++; }
    while (i < v.length && /\s/.test(v[i])) i++;
    return i;
  }

  function wordEnd(i) { // e
    const v = text();
    const cls = (ch) => (/\w/.test(ch) ? "w" : /\s/.test(ch) ? "s" : "p");
    i++;
    while (i < v.length && /\s/.test(v[i])) i++;
    if (i >= v.length) return v.length;
    const c = cls(v[i]);
    while (i < v.length && cls(v[i]) === c) i++;
    return i - 1;
  }

  function wordBack(i) { // b
    const v = text();
    const cls = (ch) => (/\w/.test(ch) ? "w" : /\s/.test(ch) ? "s" : "p");
    i--;
    while (i > 0 && /\s/.test(v[i])) i--;
    const c = cls(v[i]);
    while (i > 0 && cls(v[i - 1]) === c) i--;
    return i;
  }

  function lineStart(i) { return lineInfo(i).start; }
  function firstNonBlank(i) {
    const { start, end } = lineInfo(i);
    const v = text();
    for (let k = start; k < end; k++) if (!/\s/.test(v[k])) return k;
    return start;
  }
  function lineEnd(i) { const { end, start } = lineInfo(i); return Math.max(start, end - 1); }

  function gotoLine(n) {
    const lines = text().split("\n");
    n = Math.max(1, Math.min(n, lines.length));
    let i = 0;
    for (let k = 1; k < n; k++) i += lines[k - 1].length + 1;
    return i;
  }

  function findChar(i, ch, forward = true) {
    const { start, end } = lineInfo(i);
    const v = text();
    if (forward) {
      for (let k = i + 1; k <= end; k++) if (v[k] === ch) return k;
    } else {
      for (let k = i - 1; k >= start; k--) if (v[k] === ch) return k;
    }
    return i;
  }

  const MOTIONS = {
    h: motionH, l: motionL, j: motionJ, k: motionK,
    w: wordForward, b: wordBack, e: wordEnd,
    "0": lineStart, "^": firstNonBlank, $: lineEnd,
  };

  // ---------- editing ----------

  function splice(start, end, insert, caret) {
    const t = el();
    t.value = t.value.slice(0, start) + insert + t.value.slice(end);
    setCaret(caret === undefined ? start : caret);
    t.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function yankRange(start, end, linewise) {
    state.register = text().slice(start, end);
    state.registerLinewise = !!linewise;
    toast_(`已复制 ${end - start} 字符`);
  }

  function toast_(msg) {
    if (window.toast) window.toast(msg, "info");
  }

  function applyOperator(op, start, end, linewise) {
    if (end <= start) return;
    if (op === "y") { yankRange(start, end, linewise); return; }
    if (op === "d" || op === "c") {
      state.register = text().slice(start, end);
      state.registerLinewise = !!linewise;
      splice(start, end, "", start);
      if (op === "c") setMode("insert");
    }
  }

  function linewiseRange(i, count) {
    const { start } = lineInfo(i);
    let end = i;
    for (let k = 0; k < count - 1; k++) {
      const info = lineInfo(end);
      end = info.end < text().length ? info.end + 1 : info.end;
    }
    const last = lineInfo(end);
    return [start, last.end < text().length ? last.end + 1 : last.end];
  }

  function paste(after) {
    const t = el();
    const v = t.value;
    let at;
    if (state.registerLinewise) {
      const info = lineInfo(pos());
      at = after ? (info.end < v.length ? info.end + 1 : v.length) : info.start;
      let content = state.register;
      if (!content.endsWith("\n")) content += "\n";
      splice(at, at, content, at);
    } else {
      at = after ? Math.min(pos() + 1, v.length) : pos();
      splice(at, at, state.register, at + state.register.length - 1);
    }
  }

  function openLine(below) {
    const info = lineInfo(pos());
    const v = text();
    if (below) {
      splice(info.end, info.end, "\n", info.end + 1);
    } else {
      splice(info.start, info.start, "\n", info.start);
      setCaret(info.start);
    }
    setMode("insert");
  }

  // ---------- key handling ----------

  function takeCount() {
    const n = parseInt(state.count || "0", 10);
    state.count = "";
    return Math.max(1, n || 1);
  }

  function doMotion(key, extend) {
    const count = takeCount();
    let i = pos();
    if (key === "g") { i = gotoLine(count === 1 && !state.awaitingG ? 1 : count); }
    else if (key === "G") { i = gotoLine(count > 1 ? count : Number.MAX_SAFE_INTEGER); }
    else {
      const fn = MOTIONS[key];
      if (!fn) return;
      for (let k = 0; k < count; k++) i = fn(i);
    }
    setCaret(i, extend);
  }

  function doOperator(op, count) {
    const i = pos();
    if (["d", "y", "c"].includes(op)) {
      if (state.op === op) { // dd yy cc
        const [s, e] = linewiseRange(i, count * (state.opCount || 1));
        applyOperator(op, s, e, true);
        state.op = null; state.opCount = 0;
        return;
      }
      state.op = op;
      state.opCount = count;
      return;
    }
  }

  function handleNormal(e) {
    const key = e.key;
    if (/^[0-9]$/.test(key) && !(key === "0" && !state.count && !state.op)) {
      state.count += key;
      e.preventDefault();
      return;
    }
    const count = parseInt(state.count || "0", 10) || 1;

    if (state.awaitingG) {
      e.preventDefault();
      state.awaitingG = false;
      if (key === "g") {
        if (state.op) {
          const [s, e] = linewiseRange(0, count);
          applyOperator(state.op, s, e, true);
          state.op = null; state.opCount = 0;
        } else setCaret(gotoLine(count === 1 && state.count === "" ? 1 : count), state.mode === "visual");
      }
      state.count = "";
      return;
    }

    if (state.op) {
      const fn = MOTIONS[key];
      if (fn) {
        e.preventDefault();
        const target = (() => { let i = pos(); for (let k = 0; k < count; k++) i = fn(i); return i; })();
        const inclusive = ["e", "$"].includes(key) ? 1 : 0;
        const [s, e2] = target >= pos()
          ? [pos(), Math.min(target + inclusive, text().length)]
          : [target, pos() + inclusive];
        applyOperator(state.op, s, e2, false);
        state.op = null; state.opCount = 0; state.count = "";
      } else if (key === "g") {
        state.awaitingG = true;
      } else if (["d", "y", "c"].includes(key) && key === state.op) {
        const [s, e2] = linewiseRange(pos(), count * state.opCount);
        applyOperator(state.op, s, e2, true);
        state.op = null; state.opCount = 0; state.count = "";
      } else {
        state.op = null; state.opCount = 0; state.count = "";
      }
      e.preventDefault();
      return;
    }

    switch (key) {
      case "i": setMode("insert"); break;
      case "a": setCaret(Math.min(pos() + 1, text().length)); setMode("insert"); break;
      case "I": setCaret(firstNonBlank(pos())); setMode("insert"); break;
      case "A": setCaret(lineInfo(pos()).end); setMode("insert"); break;
      case "o": openLine(true); break;
      case "O": openLine(false); break;
      case "v": state.mode = "visual"; state.visualAnchor = pos(); setMode("visual"); return;
      case "x": {
        const c = takeCount();
        const { end } = lineInfo(pos());
        applyOperator("d", pos(), Math.min(pos() + c, end), false);
        return;
      }
      case "r": {
        e.preventDefault();
        const onChar = (ev) => {
          el().removeEventListener("keydown", onChar, true);
          ev.preventDefault(); ev.stopPropagation();
          if (ev.key.length === 1) splice(pos(), pos() + 1, ev.key, pos());
        };
        el().addEventListener("keydown", onChar, { once: true, capture: true });
        return;
      }
      case "p": case "P": paste(key === "p"); return;
      case "u": window.undoEditor && window.undoEditor(); return;
      case "D": applyOperator("d", pos(), lineEnd(pos()) + 1, false); return;
      case "C": applyOperator("c", pos(), lineEnd(pos()) + 1, false); return;
      case "Y": applyOperator("y", pos(), lineEnd(pos()) + 1, false); return;
      case "G": setCaret(gotoLine(state.count ? count : Number.MAX_SAFE_INTEGER), state.mode === "visual"); state.count = ""; return;
      case "g": state.awaitingG = true; state.opCount = count; return;
      case "f": case "F": {
        e.preventDefault();
        const forward = key === "f";
        const onChar = (ev) => {
          ev.preventDefault(); ev.stopPropagation();
          if (ev.key.length === 1) {
            state.lastFind = { ch: ev.key, forward };
            setCaret(findChar(pos(), ev.key, forward), state.mode === "visual");
          }
        };
        el().addEventListener("keydown", onChar, { once: true, capture: true });
        return;
      }
      case "d": case "y": case "c":
        state.op = key; state.opCount = count; state.count = "";
        e.preventDefault();
        return;
      default:
        if (MOTIONS[key]) { doMotion(key, state.mode === "visual"); }
        else if (key === ";") {
          if (state.lastFind) setCaret(findChar(pos(), state.lastFind.ch, state.lastFind.forward), state.mode === "visual");
        } else if (key === ",") {
          if (state.lastFind) setCaret(findChar(pos(), state.lastFind.ch, !state.lastFind.forward), state.mode === "visual");
        } else {
          state.count = "";
          return;
        }
        e.preventDefault();
        return;
    }
    state.count = "";
    e.preventDefault();
  }

  function handleVisual(e) {
    const key = e.key;
    if (key === "Escape") {
      setMode("normal");
      setCaret(pos());
      e.preventDefault();
      return;
    }
    if (["d", "y", "c"].includes(key)) {
      const [s, e2] = [Math.min(state.visualAnchor, pos()), Math.max(state.visualAnchor, pos()) + 1];
      applyOperator(key, s, e2, false);
      setMode("normal");
      e.preventDefault();
      return;
    }
    if (key === "v") { setMode("normal"); setCaret(pos()); e.preventDefault(); return; }
    handleNormal(e);
  }

  function handleInsert(e) {
    if (e.key === "Escape") {
      setMode("normal");
      setCaret(motionH(pos() === 0 ? 0 : pos()));
      e.preventDefault();
    }
  }

  function onKeydown(e) {
    if (!state.enabled) return;
    const t = el();
    if (e.target !== t) return;
    if (e.ctrlKey && e.key === "r" && state.mode === "normal") {
      window.redoEditor && window.redoEditor();
      e.preventDefault();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (state.mode === "insert") handleInsert(e);
    else if (state.mode === "visual") handleVisual(e);
    else handleNormal(e);
  }

  function attach() {
    const t = el();
    if (!t) return;
    t.addEventListener("keydown", onKeydown, true);
    t.addEventListener("focus", () => setMode(state.mode));
    if (state.enabled) setMode("normal");
  }

  window.LeetGPUVim = {
    get enabled() { return state.enabled; },
    toggle() {
      state.enabled = !state.enabled;
      localStorage.setItem("leetgpu.vim", state.enabled ? "1" : "0");
      setMode(state.enabled ? "normal" : "insert");
      return state.enabled;
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attach);
  } else {
    attach();
  }
})();
