/* CodeMirror 5 setup for the LeetGPU editor.
 *
 * Wraps the vendored CodeMirror (see web/vendor/codemirror/README.md) with:
 *   - a CUDA mode (clike/cpp plus the CUDA keywords and builtins)
 *   - the real Vim keymap from CodeMirror's keymap/vim.js
 *   - indentation guides (CodeMirror 5 ships none, so a tiny overlay draws them)
 *
 * Exposes window.LeetGPUEditor.
 */
(function () {
  "use strict";

  const MODE_BY_LANGUAGE = {
    cuda: "text/x-cuda",
    triton: "text/x-python",
    pytorch: "text/x-python",
  };

  const CUDA_KEYWORDS = [
    "__global__", "__device__", "__host__", "__shared__", "__constant__",
    "__managed__", "__restrict__", "__inline__", "__forceinline__",
    "__launch_bounds__", "__noinline__", "__syncthreads", "__syncwarp",
    "__threadfence", "__threadfence_block", "__ballot", "__shfl",
    "__shfl_down", "__any", "__all", "__popc", "__clz", "__ffs",
    "__float2int_rn", "__int_as_float", "__float_as_int", "cooperative_groups",
  ];

  const CUDA_TYPES = [
    "char2", "char3", "char4", "uchar4", "short2", "short4", "int2", "int3",
    "int4", "uint2", "uint3", "uint4", "longlong2", "float2", "float3",
    "float4", "double2", "double3", "double4", "dim3", "half", "half2",
    "cudaError_t", "cudaStream_t", "cudaEvent_t", "cudaDeviceProp",
  ];

  const CUDA_BUILTINS = [
    "blockIdx", "blockDim", "gridDim", "threadIdx", "warpSize",
    "atomicAdd", "atomicSub", "atomicMax", "atomicMin", "atomicCAS",
    "atomicExch", "atomicInc", "atomicDec", "atomicAnd", "atomicOr",
    "atomicXor", "cudaMalloc", "cudaMallocManaged", "cudaFree",
    "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyToSymbol", "cudaMemset",
    "cudaDeviceSynchronize", "cudaGetLastError", "cudaGetErrorString",
    "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize",
    "cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize",
    "cudaEventElapsedTime", "cudaOccupancyMaxPotentialBlockSize",
    "make_float2", "make_float3", "make_float4", "make_int2", "make_int3",
    "make_int4", "make_uint3", "make_uint4",
    "fmaf", "sqrtf", "rsqrtf", "expf", "logf", "powf", "fabsf", "fminf",
    "fmaxf", "min", "max", "abs", "fmin", "fmax",
  ];

  function words(list) {
    // CodeMirror's own `words()` builds a plain object, and mode code calls
    // `keywords.propertyIsEnumerable(...)`, so a null-prototype object breaks it.
    const map = {};
    for (const word of list) map[word] = true;
    return map;
  }

  /** Mirror of clike's own `contains`: a keyword set may be a map or a function. */
  function contains(source, word) {
    if (typeof source === "function") return source(word);
    if (!source) return false;
    return Object.prototype.propertyIsEnumerable.call(source, word);
  }

  function registerCudaMode() {
    const cpp = CodeMirror.resolveMode("text/x-c++src");
    const cudaTypes = words(CUDA_TYPES);
    CodeMirror.defineMIME("text/x-cuda", Object.assign({}, cpp, {
      keywords: Object.assign({}, cpp.keywords, words(CUDA_KEYWORDS)),
      // clike's `types` is a predicate function, so compose instead of replacing.
      types: (word) => contains(cpp.types, word) || contains(cudaTypes, word),
      builtin: words(CUDA_BUILTINS),
    }));
  }

  /* Indentation guides. The overlay tags the leading whitespace of each line;
     the CSS paints a repeating gradient at every indent stop. */
  const indentGuideOverlay = {
    token(stream) {
      if (stream.sol()) {
        const match = /^[ \t]+/.exec(stream.string);
        if (match) {
          stream.pos = match[0].length;
          return "indent-guide";
        }
      }
      stream.skipToEnd();
      return null;
    },
  };

  /* CodeMirror 5's Vim only does multi-line insert for *blockwise* visual mode.
     In real Vim, `V` (linewise) then `I`/`A` inserts into every selected line.
     We implement that directly with CodeMirror multi-selections. */
  function installLinewiseVisualInsert() {
    const Vim = window.CodeMirror.Vim;
    if (!Vim || typeof Vim.defineAction !== "function") return;

    for (const key of ["I", "A"]) {
      const actionArgs = {
        insertAt: key === "I" ? "startOfSelectedArea" : "endOfSelectedArea",
      };
      Vim.defineAction(`leetLinewise${key}`, function (cm, _args, vim) {
        if (!vim.visualMode) return;
        const selections = cm.listSelections();
        if (!selections.length) return;
        const sel = selections[0];
        const first = Math.min(sel.anchor.line, sel.head.line);
        const last = Math.max(sel.anchor.line, sel.head.line);
        const atEnd = key === "A";

        const ranges = [];
        for (let line = first; line <= last; line++) {
          const ch = atEnd ? cm.getLine(line).length : 0;
          ranges.push({
            anchor: CodeMirror.Pos(line, ch),
            head: CodeMirror.Pos(line, ch),
          });
        }

        Vim.exitVisualMode(cm, false);
        vim.insertMode = true;
        vim.insertModeRepeat = 1;
        // vim keeps `disableInput` on in normal mode; insert mode must turn it
        // back off or CodeMirror ignores every keystroke.
        cm.setOption("disableInput", false);
        cm.toggleOverwrite(false);
        // Switch the keymap first: vim's keyMap option handler resets the
        // selection, so the multi-cursor ranges must be applied last.
        cm.setOption("keyMap", "vim-insert");
        CodeMirror.signal(cm, "vim-mode-change", { mode: "insert" });
        cm.setSelections(ranges, 0, { scroll: true });
        cm.focus();
      });
      // mapCommand(keys, type, name, args, extra): `type` is the action type
      // ("action"), and the keymap context goes in `extra`.
      Vim.mapCommand(key, "action", `leetLinewise${key}`, actionArgs, {
        context: "visual",
        isEdit: true,
      });
    }
  }

  function modeFor(language) {
    return MODE_BY_LANGUAGE[language] || MODE_BY_LANGUAGE.cuda;
  }

  /**
   * Build the editor over an existing <textarea>.
   * Returns { editor, setVim, setLanguage, onVimModeChange, onSaveRequest }.
   */
  function create(textarea, options) {
    registerCudaMode();
    installLinewiseVisualInsert();
    const settings = options || {};

    const editor = CodeMirror.fromTextArea(textarea, {
      value: settings.value || "",
      mode: modeFor(settings.language),
      theme: "leetgpu",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      smartIndent: true,
      electricChars: true,
      lineWrapping: false,
      viewportMargin: Infinity,
      styleActiveLine: true,
      matchBrackets: true,
      autoCloseBrackets: true,
      showCursorWhenSelecting: true,
      keyMap: settings.vim === false ? "default" : "vim",
    });

    editor.addOverlay(indentGuideOverlay);

    // Let Vim's `:w` / `:wq` persist the draft instead of doing nothing.
    if (window.CodeMirror.Vim && typeof window.CodeMirror.Vim.defineEx === "function") {
      window.CodeMirror.Vim.defineEx("write", "w", () => {
        if (typeof settings.onSave === "function") settings.onSave();
        editor.setOption("keyMap", editor.getOption("keyMap"));
      });
      window.CodeMirror.Vim.defineEx("wq", "wq", () => {
        if (typeof settings.onSave === "function") settings.onSave();
      });
    }

    const modeHandlers = [];
    editor.on("vim-mode-change", (event) => {
      for (const handler of modeHandlers) handler(event.mode, event.subMode);
    });

    return {
      editor,
      setLanguage(language) {
        editor.setOption("mode", modeFor(language));
      },
      setVim(enabled) {
        // Leave insert mode *before* the vim keymap is detached: once the
        // keymap changes, vim.js has torn down `editor.state.vim` and
        // exitInsertMode would throw on the null state.
        if (!enabled && window.CodeMirror.Vim) {
          try {
            window.CodeMirror.Vim.exitInsertMode(editor);
          } catch {
            /* vim state already gone — nothing to exit */
          }
        }
        editor.setOption("keyMap", enabled ? "vim" : "default");
        editor.focus();
      },
      onVimModeChange(handler) {
        modeHandlers.push(handler);
      },
      refresh() {
        editor.refresh();
      },
    };
  }

  window.LeetGPUEditor = { create, modeFor };
})();
