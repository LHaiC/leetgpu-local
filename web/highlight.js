/* Dependency-free syntax highlighting + keyword hints for the <textarea>
 * editor.
 *
 * A <pre> layer sits exactly behind a transparent textarea and is repainted
 * on every edit. Both layers share identical font metrics, padding and
 * scroll offsets, so the caret always lines up with the highlighted glyphs.
 */
(function () {
  "use strict";

  const CUDA_KEYWORDS = [
    "__global__", "__device__", "__host__", "__shared__", "__constant__",
    "__restrict__", "__inline__", "__forceinline__", "__launch_bounds__",
    "__syncthreads", "__syncwarp", "__threadfence",
  ];

  const CPP_KEYWORDS = [
    "alignas", "alignof", "asm", "auto", "break", "case", "catch", "class",
    "const", "constexpr", "continue", "default", "delete", "do", "else",
    "enum", "explicit", "extern", "false", "for", "friend", "goto", "if",
    "inline", "mutable", "namespace", "new", "noexcept", "nullptr", "operator",
    "private", "protected", "public", "register", "return", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this",
    "throw", "true", "try", "typedef", "typename", "union", "using", "virtual",
    "volatile", "while",
  ];

  const CUDA_TYPES = [
    "bool", "char", "char2", "char3", "char4", "dim3", "double", "double2",
    "double4", "float", "float2", "float3", "float4", "half", "int", "int2",
    "int3", "int4", "long", "short", "size_t", "uchar4", "uint", "uint3",
    "unsigned", "ushort", "void", "cudaError_t", "cudaStream_t",
  ];

  const CUDA_BUILTINS = [
    "blockDim", "blockIdx", "gridDim", "threadIdx", "warpSize", "atomicAdd",
    "atomicMax", "atomicMin", "atomicCAS", "atomicExch", "fma", "sqrtf",
    "expf", "logf", "powf", "rsqrtf", "fabsf", "min", "max", "abs", "__ldg",
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync",
    "cudaDeviceSynchronize", "cudaGetLastError", "cudaMemset", "cudaEventCreate",
    "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime",
    "cudaStreamCreate", "cudaStreamSynchronize",
  ];

  const PY_KEYWORDS = [
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "False", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "None", "nonlocal", "not",
    "or", "pass", "raise", "return", "True", "try", "while", "with", "yield",
  ];

  const PY_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "map", "max", "min", "print", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
  ];

  const TORCH_NAMES = [
    "torch", "triton", "tl", "nn", "F", "Tensor", "tensor", "dtype",
    "device", "shape", "arange", "empty_like", "zeros", "ones", "full",
    "float32", "float16", "bfloat16", "matmul", "softmax", "relu", "sum",
    "exp", "sqrt", "rsqrt", "load", "store", "program_id", "num_warps",
    "num_stages", "constexpr", "autotune", "cdiv", "where", "maximum",
  ];

  const LANG_SETS = {
    cuda: {
      keywords: new Set([...CPP_KEYWORDS, ...CUDA_KEYWORDS]),
      types: new Set(CUDA_TYPES),
      builtins: new Set(CUDA_BUILTINS),
      comments: ["//", "/*"],
      hint: [...CUDA_KEYWORDS, ...CUDA_BUILTINS, ...CUDA_TYPES, ...CPP_KEYWORDS],
    },
    triton: {
      keywords: new Set(PY_KEYWORDS),
      types: new Set(),
      builtins: new Set([...PY_BUILTINS, ...TORCH_NAMES]),
      comments: ["#"],
      hint: [...TORCH_NAMES, ...PY_KEYWORDS, ...PY_BUILTINS],
    },
    pytorch: {
      keywords: new Set(PY_KEYWORDS),
      types: new Set(),
      builtins: new Set([...PY_BUILTINS, ...TORCH_NAMES]),
      comments: ["#"],
      hint: [...TORCH_NAMES, ...PY_KEYWORDS, ...PY_BUILTINS],
    },
  };

  const ESCAPE = { "&": "&amp;", "<": "&lt;", ">": "&gt;" };

  function esc(text) {
    return text.replace(/[&<>]/g, (ch) => ESCAPE[ch]);
  }

  function span(cls, text) {
    return `<span class="${cls}">${esc(text)}</span>`;
  }

  // One pass over the source, longest-match-first alternation.
  // The comment and preprocessor patterns differ per language family:
  //   C-like  : // and /* */, with #… as a preprocessor directive
  //   Python  : # only, no preprocessor
  const PATTERNS = {
    cuda: [
      "(\\/\\*[\\s\\S]*?\\*\\/|\\/\\/[^\\n]*)", // 1 comment
      "(\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*')", // 2 string
      "(^[ \\t]*#[^\\n]*)", // 3 preprocessor
      "\\b(\\d+\\.?\\d*(?:[eE][+-]?\\d+)?[fFuUlL]*|0[xX][0-9a-fA-F]+)\\b", // 4 number
      "\\b([A-Za-z_]\\w*)\\b(?=\\s*\\()", // 5 call
      "\\b([A-Za-z_]\\w*)\\b", // 6 identifier
      "([+\\-*/%=<>!&|^~?:;,.(){}\\[\\]])", // 7 punctuation
    ],
    python: [
      "(#[^\\n]*)", // 1 comment
      "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*')", // 2 string
      "()", // 3 no preprocessor
      "\\b(\\d+\\.?\\d*(?:[eE][+-]?\\d+)?|0[xX][0-9a-fA-F]+)\\b", // 4 number
      "\\b([A-Za-z_]\\w*)\\b(?=\\s*\\()", // 5 call
      "\\b([A-Za-z_]\\w*)\\b", // 6 identifier
      "([+\\-*/%=<>!&|^~?:;,.(){}\\[\\]])", // 7 punctuation
    ],
  };

  const REGEXES = {};
  function regexFor(language) {
    const family = language === "cuda" ? "cuda" : "python";
    if (!REGEXES[family]) REGEXES[family] = new RegExp(PATTERNS[family].join("|"), "gm");
    return REGEXES[family];
  }

  function highlight(source, language) {
    const set = LANG_SETS[language] || LANG_SETS.cuda;
    const token = regexFor(language);
    let out = "";
    let last = 0;
    let match;
    token.lastIndex = 0;
    while ((match = token.exec(source)) !== null) {
      if (match.index > last) out += esc(source.slice(last, match.index));
      const [full, comment, string, preproc, number, call, ident, punct] = match;
      if (comment !== undefined && comment !== "") out += span("tok-comment", full);
      else if (string !== undefined && string !== "") out += span("tok-string", full);
      else if (preproc !== undefined && preproc !== "") out += span("tok-preproc", full);
      else if (number !== undefined) out += span("tok-number", full);
      else if (call !== undefined) out += span("tok-call", full);
      else if (ident !== undefined) {
        if (set.keywords.has(ident)) out += span("tok-keyword", ident);
        else if (set.types.has(ident)) out += span("tok-type", ident);
        else if (set.builtins.has(ident)) out += span("tok-builtin", ident);
        else out += esc(ident);
      } else if (punct !== undefined) out += span("tok-punct", punct);
      else out += esc(full);
      last = match.index + full.length;
      if (full.length === 0) token.lastIndex++; // safety
    }
    out += esc(source.slice(last));
    // A trailing newline collapses visually; pad so the layer keeps its height.
    if (out.endsWith("\n")) out += " ";
    return out;
  }

  const HINT_ICONS = { keyword: "kw", builtin: "fn", type: "ty" };

  function measure(textarea) {
    const cs = getComputedStyle(textarea);
    const probe = document.createElement("span");
    probe.textContent = "0".repeat(100);
    probe.style.cssText =
      "position:absolute;visibility:hidden;white-space:pre;padding:0;border:0;margin:0";
    for (const key of ["fontFamily", "fontSize", "fontWeight", "letterSpacing"]) {
      probe.style[key] = cs[key];
    }
    document.body.appendChild(probe);
    const width = probe.getBoundingClientRect().width / 100;
    probe.remove();
    const size = parseFloat(cs.fontSize) || 12;
    return {
      charWidth: width || size * 0.6,
      lineHeight: parseFloat(cs.lineHeight) || size * 1.6,
      padLeft: parseFloat(cs.paddingLeft) || 0,
      padTop: parseFloat(cs.paddingTop) || 0,
    };
  }

  function attachHints(textarea, popup, getLanguage) {
    let items = [];
    let index = 0;
    let wordStart = -1;
    let metrics = null;

    function close() {
      items = [];
      index = 0;
      popup.classList.add("hidden");
      popup.innerHTML = "";
    }

    function prefix() {
      const pos = textarea.selectionStart;
      if (pos !== textarea.selectionEnd) return null;
      // A trailing period is a member access — don't spam hints there.
      const text = textarea.value.slice(0, pos);
      const m = text.match(/[A-Za-z_]\w*$/);
      if (!m) return null;
      const start = pos - m[0].length;
      const before = text[start - 1];
      if (before === "." || before === "#") return null;
      return { text: m[0], start };
    }

    function render() {
      popup.innerHTML = items
        .map(
          (item, i) =>
            `<div class="hint-item${i === index ? " active" : ""}" data-i="${i}">
               <span class="hint-badge ${HINT_ICONS[item.kind]}"></span>
               <span class="hint-label">${esc(item.label)}</span>
             </div>`,
        )
        .join("");
    }

    function position() {
      metrics = metrics || measure(textarea);
      const before = textarea.value.slice(0, wordStart);
      const lines = before.split("\n");
      const line = lines.length - 1;
      const column = lines[line].length;
      const host = popup.parentElement;
      const limit = host ? host.clientWidth : 400;

      let top = metrics.padTop + (line + 1) * metrics.lineHeight - textarea.scrollTop;
      let left = metrics.padLeft + column * metrics.charWidth - textarea.scrollLeft;
      if (top + popup.offsetHeight > (host ? host.clientHeight : 400)) {
        top = metrics.padTop + line * metrics.lineHeight - textarea.scrollTop - popup.offsetHeight;
      }
      left = Math.max(4, Math.min(left, Math.max(4, limit - popup.offsetWidth - 4)));
      popup.style.top = `${Math.max(4, top)}px`;
      popup.style.left = `${left}px`;
    }

    function open(list, start) {
      items = list;
      index = 0;
      wordStart = start;
      render();
      popup.classList.remove("hidden");
      position();
    }

    function accept() {
      if (!items.length) return;
      const textareaValue = textarea.value;
      const pos = textarea.selectionStart;
      textarea.value =
        textareaValue.slice(0, wordStart) + items[index].label + textareaValue.slice(pos);
      const caret = wordStart + items[index].label.length;
      textarea.setSelectionRange(caret, caret);
      close();
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function update() {
      const info = prefix();
      if (!info) return close();
      const set = LANG_SETS[getLanguage()] || LANG_SETS.cuda;
      const seen = new Set();
      const matches = [];
      for (const candidate of set.hint) {
        if (candidate.length > 1 && candidate.startsWith(info.text) && !seen.has(candidate)) {
          seen.add(candidate);
          matches.push({
            label: candidate,
            kind: set.keywords.has(candidate)
              ? "keyword"
              : set.types.has(candidate)
                ? "type"
                : "builtin",
          });
        }
        if (matches.length >= 8) break;
      }
      if (!info.text || !matches.length) return close();
      open(matches, info.start);
    }

    popup.addEventListener("mousedown", (event) => {
      const item = event.target.closest(".hint-item");
      if (!item) return;
      event.preventDefault();
      index = Number(item.dataset.i);
      accept();
    });

    textarea.addEventListener("input", update);
    textarea.addEventListener("blur", () => setTimeout(close, 120));
    textarea.addEventListener("scroll", close);
    window.addEventListener("resize", close);
    textarea.addEventListener("keydown", (event) => {
      if (popup.classList.contains("hidden")) return;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        index = (index + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        render();
      } else if (event.key === "Tab" || event.key === "Enter") {
        event.preventDefault();
        event.stopImmediatePropagation();
        accept();
      } else if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        close();
      }
    }, true);

    return { close, update };
  }

  window.CodeHighlight = { highlight, attachHints };
})();
