const elements = {
  artifactBar: document.querySelector("#artifact-bar"),
  challengeCount: document.querySelector("#challenge-count"),
  challengeList: document.querySelector("#challenge-list"),
  challengeNumber: document.querySelector("#challenge-number"),
  challengeSearch: document.querySelector("#challenge-search"),
  challengeTitle: document.querySelector("#challenge-title"),
  codeEditor: document.querySelector("#code-editor"),
  compilerOutput: document.querySelector("#compiler-output"),
  compilerPanel: document.querySelector("#compiler-panel"),
  copySubmission: document.querySelector("#copy-submission"),
  deleteSubmission: document.querySelector("#delete-submission"),
  difficultyBadge: document.querySelector("#difficulty-badge"),
  draftState: document.querySelector("#draft-state"),
  executionOutput: document.querySelector("#execution-output"),
  languageSwitch: document.querySelector(".language-switch"),
  leaderboardBody: document.querySelector("#leaderboard-body"),
  leaderboardEmpty: document.querySelector("#leaderboard-empty"),
  leaderboardLanguage: document.querySelector("#leaderboard-language"),
  leaderboardSubtitle: document.querySelector("#leaderboard-subtitle"),
  lineNumbers: document.querySelector("#line-numbers"),
  metricsStrip: document.querySelector("#metrics-strip"),
  ncuExportDir: document.querySelector("#ncu-export-dir"),
  pickNcuDirectory: document.querySelector("#pick-ncu-directory"),
  problemBody: document.querySelector("#problem-body"),
  profileEnabled: document.querySelector("#profile-enabled"),
  queueState: document.querySelector("#queue-state"),
  redoCode: document.querySelector("#redo-code"),
  resetCode: document.querySelector("#reset-code"),
  resultIndicator: document.querySelector("#result-indicator"),
  resultSummary: document.querySelector("#result-summary"),
  runButton: document.querySelector("#run-button"),
  runnerName: document.querySelector("#runner-name"),
  sidebar: document.querySelector("#sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  saveSettings: document.querySelector("#save-settings"),
  settingsButton: document.querySelector("#settings-button"),
  settingsModal: document.querySelector("#settings-modal"),
  solutionFilename: document.querySelector("#solution-filename"),
  systemStatus: document.querySelector("#system-status"),
  submissionModal: document.querySelector("#submission-modal"),
  submissionModalMeta: document.querySelector("#submission-modal-meta"),
  submissionModalTitle: document.querySelector("#submission-modal-title"),
  submissionSource: document.querySelector("#submission-source"),
  testResults: document.querySelector("#test-results"),
  testButton: document.querySelector("#test-button"),
  toastRegion: document.querySelector("#toast-region"),
  undoCode: document.querySelector("#undo-code"),
  viewTabs: document.querySelector(".view-tabs"),
  adminButton: document.querySelector("#admin-button"),
  adminModal: document.querySelector("#admin-modal"),
  announcementsBar: document.querySelector("#announcements-bar"),
  authForm: document.querySelector("#auth-form"),
  authModal: document.querySelector("#auth-modal"),
  authModalTitle: document.querySelector("#auth-modal-title"),
  authSubmit: document.querySelector("#auth-submit"),
  authSwitch: document.querySelector("#auth-switch"),
  authConfirmWrap: document.querySelector("#auth-confirm-wrap"),
  authConfirm: document.querySelector("#auth-confirm"),
  authError: document.querySelector("#auth-error"),
  authUsername: document.querySelector("#auth-username"),
  authPassword: document.querySelector("#auth-password"),
  loginButton: document.querySelector("#login-button"),
  logoutButton: document.querySelector("#logout-button"),
  userMenuButton: document.querySelector("#user-menu-button"),
  usernameLabel: document.querySelector("#username-label"),
  registrationOpen: document.querySelector("#registration-open"),
  adminUsersList: document.querySelector("#admin-users-list"),
  adminAnnouncementsList: document.querySelector("#admin-announcements-list"),
  announcementForm: document.querySelector("#announcement-form"),
  announcementTitle: document.querySelector("#announcement-title"),
  announcementBody: document.querySelector("#announcement-body"),
  vimToggle: document.querySelector("#vim-toggle"),
  vimMode: document.querySelector("#vim-mode"),
};

const session = {
  user: null,
  authRequired: false,
  registrationOpen: true,
  authMode: "login",
};

const state = {
  challenges: [],
  currentKey: null,
  detail: null,
  difficulty: "all",
  language: localStorage.getItem("leetgpu.language") || "cuda",
  activeJob: null,
  pollTimer: null,
  saveTimer: null,
  settings: null,
  selectedSubmission: null,
  editorHistory: {
    entries: [],
    index: -1,
    applying: false,
    lastRecordedAt: 0,
  },
};

const filenames = {
  cuda: "solution.cu",
  triton: "solution.triton.py",
  pytorch: "solution.pytorch.py",
};

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && session.authRequired) {
      session.user = null;
      renderAuthState();
      openAuthModal();
    }
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return body;
}

function toast(message, type = "info") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  elements.toastRegion.append(node);
  setTimeout(() => node.remove(), 4200);
}

// ---------- auth ----------

function renderAuthState() {
  const loggedIn = !!session.user;
  elements.loginButton.classList.toggle("hidden", loggedIn || !session.authRequired);
  elements.userMenuButton.classList.toggle("hidden", !loggedIn);
  elements.logoutButton.classList.toggle("hidden", !loggedIn);
  elements.adminButton.classList.toggle(
    "hidden",
    !(loggedIn && session.user.role === "admin"),
  );
  if (loggedIn) elements.usernameLabel.textContent = session.user.username;
  refreshIcons();
}

function openAuthModal(mode = "login") {
  session.authMode = mode;
  elements.authModalTitle.textContent = mode === "login" ? "登录" : "注册";
  elements.authSubmit.textContent = mode === "login" ? "登录" : "注册";
  elements.authSwitch.textContent =
    mode === "login" ? "没有账号？注册" : "已有账号？登录";
  elements.authSwitch.style.display =
    mode === "login" && !session.registrationOpen ? "none" : "";
  elements.authConfirmWrap.classList.toggle("hidden", mode === "login");
  elements.authError.textContent = "";
  elements.authModal.showModal();
}

async function submitAuth(event) {
  event.preventDefault();
  const username = elements.authUsername.value.trim();
  const password = elements.authPassword.value;
  elements.authError.textContent = "";
  try {
    if (session.authMode === "register") {
      if (password !== elements.authConfirm.value) {
        throw new Error("两次输入的密码不一致");
      }
      await api("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
    } else {
      await api("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
    }
    elements.authModal.close();
    await refreshSession();
    initialize();
  } catch (error) {
    elements.authError.textContent = error.message;
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST", body: "{}" });
  } finally {
    session.user = null;
    location.reload();
  }
}

async function refreshSession() {
  const me = await api("/api/me");
  session.user = me.user;
  session.authRequired = me.auth_required;
  session.registrationOpen = me.registration_open;
  renderAuthState();
}

function draftKey() {
  return `leetgpu.draft.${state.currentKey}.${state.language}`;
}

function currentStarter() {
  return state.detail?.starters?.[state.language] || "";
}

function updateLineNumbers() {
  const count = Math.max(1, elements.codeEditor.value.split("\n").length);
  elements.lineNumbers.textContent = Array.from({ length: count }, (_, index) => index + 1).join(
    "\n",
  );
}

function saveDraft() {
  if (!state.currentKey) return;
  const source = elements.codeEditor.value;
  localStorage.setItem(draftKey(), source);
  elements.draftState.textContent = "saved";
  if (session.user) {
    api("/api/solutions", {
      method: "POST",
      body: JSON.stringify({
        challenge: state.currentKey,
        language: state.language,
        source,
      }),
    }).catch(() => {});
  }
}

function editorSnapshot() {
  return {
    value: elements.codeEditor.value,
    start: elements.codeEditor.selectionStart,
    end: elements.codeEditor.selectionEnd,
  };
}

function resetEditorHistory() {
  state.editorHistory.entries = [editorSnapshot()];
  state.editorHistory.index = 0;
  state.editorHistory.lastRecordedAt = 0;
  updateHistoryButtons();
}

function recordEditorHistory(forceNew = false) {
  if (state.editorHistory.applying) return;
  const snapshot = editorSnapshot();
  const current = state.editorHistory.entries[state.editorHistory.index];
  if (current?.value === snapshot.value) return;
  state.editorHistory.entries = state.editorHistory.entries.slice(
    0,
    state.editorHistory.index + 1,
  );
  const now = Date.now();
  const canCoalesce =
    !forceNew &&
    state.editorHistory.entries.length > 1 &&
    now - state.editorHistory.lastRecordedAt < 500;
  if (canCoalesce) {
    state.editorHistory.entries[state.editorHistory.index] = snapshot;
  } else {
    state.editorHistory.entries.push(snapshot);
    state.editorHistory.index += 1;
  }
  if (state.editorHistory.entries.length > 200) {
    state.editorHistory.entries.shift();
    state.editorHistory.index -= 1;
  }
  state.editorHistory.lastRecordedAt = now;
  updateHistoryButtons();
}

function applyEditorHistory(index) {
  const snapshot = state.editorHistory.entries[index];
  if (!snapshot) return;
  state.editorHistory.applying = true;
  state.editorHistory.index = index;
  elements.codeEditor.value = snapshot.value;
  elements.codeEditor.setSelectionRange(snapshot.start, snapshot.end);
  state.editorHistory.applying = false;
  elements.draftState.textContent = "edited";
  updateLineNumbers();
  updateHistoryButtons();
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveDraft, 200);
}

function undoEditor() {
  if (state.editorHistory.index > 0) applyEditorHistory(state.editorHistory.index - 1);
}

function redoEditor() {
  if (state.editorHistory.index < state.editorHistory.entries.length - 1) {
    applyEditorHistory(state.editorHistory.index + 1);
  }
}

function updateHistoryButtons() {
  elements.undoCode.disabled = state.editorHistory.index <= 0;
  elements.redoCode.disabled =
    state.editorHistory.index < 0 ||
    state.editorHistory.index >= state.editorHistory.entries.length - 1;
}

function recordEditorEdit(forceNew = false) {
  recordEditorHistory(forceNew);
  updateLineNumbers();
  elements.draftState.textContent = "edited";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveDraft, 500);
}

function mapEditedOffset(offset, changes, affinity) {
  let delta = 0;
  for (const change of changes) {
    const changeEnd = change.index + change.remove;
    if (offset < change.index || (offset === change.index && affinity === "before")) break;
    if (offset >= changeEnd) {
      delta += change.insert.length - change.remove;
      continue;
    }
    return change.index + delta + (affinity === "after" ? change.insert.length : 0);
  }
  return offset + delta;
}

function toggleLineComments() {
  const editor = elements.codeEditor;
  const source = editor.value;
  const selectionStart = editor.selectionStart;
  const selectionEnd = editor.selectionEnd;
  const blockStart = source.lastIndexOf("\n", selectionStart - 1) + 1;
  const effectiveEnd =
    selectionEnd > selectionStart && source[selectionEnd - 1] === "\n"
      ? selectionEnd - 1
      : selectionEnd;
  const followingNewline = source.indexOf("\n", effectiveEnd);
  const blockEnd = followingNewline === -1 ? source.length : followingNewline;
  const block = source.slice(blockStart, blockEnd);
  const marker = state.language === "cuda" ? "//" : "#";
  const lines = block.split("\n");
  const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
  const markerPattern = new RegExp(`^(\\s*)${marker === "//" ? "\\/\\/" : "#"}( ?)`);
  const shouldUncomment =
    nonEmptyLines.length > 0 && nonEmptyLines.every((line) => markerPattern.test(line));
  const changes = [];
  let lineOffset = 0;

  for (const line of lines) {
    if (line.trim().length > 0) {
      const indentation = line.match(/^\s*/)[0].length;
      if (shouldUncomment) {
        const match = line.match(markerPattern);
        changes.push({
          index: lineOffset + indentation,
          remove: marker.length + match[2].length,
          insert: "",
        });
      } else {
        changes.push({
          index: lineOffset + indentation,
          remove: 0,
          insert: `${marker} `,
        });
      }
    }
    lineOffset += line.length + 1;
  }

  if (changes.length === 0) return;
  let editedBlock = block;
  for (const change of [...changes].reverse()) {
    editedBlock =
      editedBlock.slice(0, change.index) +
      change.insert +
      editedBlock.slice(change.index + change.remove);
  }

  const relativeStart = selectionStart - blockStart;
  const relativeEnd = selectionEnd - blockStart;
  const collapsed = selectionStart === selectionEnd;
  const nextStart = mapEditedOffset(
    relativeStart,
    changes,
    collapsed ? "after" : "before",
  );
  const nextEnd = collapsed
    ? nextStart
    : mapEditedOffset(relativeEnd, changes, "after");
  editor.setRangeText(editedBlock, blockStart, blockEnd, "preserve");
  editor.setSelectionRange(blockStart + nextStart, blockStart + nextEnd);
  recordEditorEdit(true);
}

async function loadCode() {
  const local = localStorage.getItem(draftKey());
  let server = null;
  if (session.user) {
    try {
      const data = await api(
        `/api/solution?challenge=${encodeURIComponent(state.currentKey)}&language=${encodeURIComponent(state.language)}`,
      );
      server = data.source;
    } catch {}
  }
  const source = server ?? local ?? currentStarter();
  elements.codeEditor.value = source;
  elements.draftState.textContent =
    server !== null ? "server" : local === null ? "starter" : "saved";
  elements.solutionFilename.textContent = filenames[state.language];
  updateLineNumbers();
  elements.codeEditor.scrollTop = 0;
  elements.lineNumbers.scrollTop = 0;
  resetEditorHistory();
}

function renderChallengeList() {
  const search = elements.challengeSearch.value.trim().toLowerCase();
  const filtered = state.challenges.filter((challenge) => {
    const difficultyMatches =
      state.difficulty === "all" || challenge.difficulty === state.difficulty;
    const searchMatches =
      !search ||
      challenge.title.toLowerCase().includes(search) ||
      String(challenge.id).includes(search);
    return difficultyMatches && searchMatches;
  });
  elements.challengeList.innerHTML = filtered
    .map(
      (challenge) => `
        <button class="challenge-item ${challenge.key === state.currentKey ? "active" : ""}"
                data-key="${escapeHtml(challenge.key)}">
          <span class="challenge-item-number">${challenge.id}</span>
          <span class="challenge-item-main">
            <span class="challenge-item-title">${escapeHtml(challenge.title)}</span>
            <span class="challenge-item-sub">${challenge.submission_count || 0} 次提交</span>
          </span>
          <span class="challenge-item-status">
            ${challenge.passed ? '<i data-lucide="circle-check"></i>' : `<span class="challenge-item-dot ${challenge.difficulty}"></span>`}
          </span>
        </button>`,
    )
    .join("");
  elements.challengeCount.textContent = `${filtered.length} 道题`;
  elements.challengeList.querySelectorAll(".challenge-item").forEach((button) => {
    button.addEventListener("click", () => selectChallenge(button.dataset.key));
  });
  refreshIcons();
}

async function selectChallenge(key) {
  if (!key) return;
  if (key === state.currentKey) {
    if (window.innerWidth <= 820) elements.sidebar.classList.remove("open");
    return;
  }
  state.currentKey = key;
  localStorage.setItem("leetgpu.challenge", key);
  renderChallengeList();
  elements.problemBody.innerHTML = '<div class="empty-state">加载中</div>';
  try {
    state.detail = await api(`/api/challenge?key=${encodeURIComponent(key)}`);
    const detail = state.detail;
    elements.challengeNumber.textContent = `#${detail.id}`;
    elements.challengeTitle.textContent = detail.title;
    elements.difficultyBadge.textContent = detail.difficulty;
    elements.difficultyBadge.className = `difficulty-badge ${detail.difficulty}`;
    document.querySelector("#access-tier").textContent = detail.access_tier;
    elements.problemBody.innerHTML = detail.problem_html;
    elements.leaderboardSubtitle.textContent = detail.title;
    setLanguage(state.language);
    renderChallengeList();
    loadLeaderboard();
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetClear?.([elements.problemBody]);
      window.MathJax.typesetPromise([elements.problemBody]).catch(() => {});
    }
    if (window.innerWidth <= 820) elements.sidebar.classList.remove("open");
  } catch (error) {
    toast(error.message, "error");
  }
}

function setLanguage(language) {
  if (!filenames[language]) return;
  state.language = language;
  localStorage.setItem("leetgpu.language", language);
  elements.languageSwitch.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.language === language);
    const available = state.detail?.starters?.[button.dataset.language] !== null;
    button.disabled = state.detail ? !available : false;
  });
  loadCode();
}

function setView(view) {
  elements.viewTabs.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view-content").forEach((content) => {
    content.classList.toggle("active", content.id === `${view}-view`);
  });
  if (view === "leaderboard") loadLeaderboard();
}

function stateIcon(kind) {
  if (kind === "passed") return "circle-check-big";
  if (kind === "failed" || kind === "error") return "circle-x";
  return "loader-circle";
}

function renderRunning(job) {
  const queued = job.status === "queued";
  const label = queued ? `队列第 ${job.queue_position || 1} 位` : "GPU 正在执行";
  const actionLabel = job.mode === "test" ? "测试" : "提交";
  elements.resultSummary.innerHTML = `
    <div class="result-state running">
      <div class="result-state-icon"><i data-lucide="${stateIcon("running")}"></i></div>
      <div><strong>${queued ? `等待${actionLabel}` : `${actionLabel}中`}</strong><span>${label}</span></div>
    </div>`;
  elements.executionOutput.textContent = job.output || "";
  elements.executionOutput.scrollTop = elements.executionOutput.scrollHeight;
  elements.queueState.textContent = queued ? label : `${actionLabel}中`;
  elements.resultIndicator.classList.add("visible");
  refreshIcons();
}

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${number < 1 ? number.toFixed(4) : number.toFixed(3)} ms`;
}

function renderInputVariables(inputs) {
  return Object.entries(inputs || {})
    .map(([name, summary]) => {
      if (summary && typeof summary === "object" && summary.type === "tensor") {
        const shape = `[${(summary.shape || []).join(", ")}]`;
        const values = summary.values !== undefined
          ? summary.values
          : {
              head: summary.head,
              tail: summary.tail,
              min: summary.min,
              max: summary.max,
              mean: summary.mean,
            };
        return `
          <div class="input-variable">
            <div class="input-variable-head">
              <span class="input-variable-name">${escapeHtml(name)}</span>
              <span class="input-variable-meta">${escapeHtml(summary.dtype)}</span>
              <span class="input-variable-meta">shape ${escapeHtml(shape)}</span>
              <span class="input-variable-meta">${summary.numel} elements</span>
            </div>
            <pre>${escapeHtml(JSON.stringify(values, null, 2))}</pre>
          </div>`;
      }
      return `
        <div class="input-variable">
          <div class="input-variable-head"><span class="input-variable-name">${escapeHtml(name)}</span></div>
          <pre>${escapeHtml(JSON.stringify(summary, null, 2))}</pre>
        </div>`;
    })
    .join("");
}

function renderCompleted(job) {
  const result = job.result;
  const passed = result?.status === "passed";
  const displayStatus = result?.status || job.status;
  const isTest = job.mode === "test";
  const detail = passed
    ? isTest
      ? "完整正确性数据集已通过，不计入排行榜"
      : "完整提交已通过，成绩已写入排行榜"
    : result?.error || job.error || "查看执行日志";
  const statusLabel = passed
    ? isTest
      ? "测试通过"
      : "提交通过"
    : isTest
      ? "测试未通过"
      : "提交未通过";
  elements.resultSummary.innerHTML = `
    <div class="result-state ${passed ? "passed" : "failed"}">
      <div class="result-state-icon"><i data-lucide="${stateIcon(displayStatus)}"></i></div>
      <div><strong>${statusLabel}</strong><span>${escapeHtml(detail)}</span></div>
    </div>`;

  if (result?.benchmark) {
    const gpu = result.benchmark.gpu_ms;
    elements.metricsStrip.innerHTML = `
      <div class="metric"><span class="metric-label">中位耗时</span><span class="metric-value">${formatMs(gpu.median)}</span></div>
      <div class="metric"><span class="metric-label">最佳耗时</span><span class="metric-value">${formatMs(gpu.min)}</span></div>
      <div class="metric"><span class="metric-label">P95</span><span class="metric-value">${formatMs(gpu.p95)}</span></div>`;
    elements.metricsStrip.classList.remove("hidden");
  } else {
    elements.metricsStrip.classList.add("hidden");
  }

  if (result?.compiler_output) {
    elements.compilerOutput.textContent = result.compiler_output;
    elements.compilerPanel.classList.remove("hidden");
  } else {
    elements.compilerOutput.textContent = "";
    elements.compilerPanel.classList.add("hidden");
  }

  if (result?.tests?.length) {
    elements.testResults.innerHTML = result.tests
      .map(
        (test, index) => {
          const expandable = test.inputs && Object.keys(test.inputs).length > 0;
          return `
            <div class="test-entry">
              <button class="test-row ${expandable ? "expandable" : ""}" data-test-index="${index}" ${expandable ? "" : "disabled"}>
                <span class="${test.status === "passed" ? "pass" : "fail"}">${test.status === "passed" ? "OK" : "X"}</span>
                <span>${escapeHtml(test.name)}${test.error ? `: ${escapeHtml(test.error)}` : ""}</span>
                <span class="test-time">${formatMs(test.duration_ms)}</span>
                ${expandable ? '<i class="test-chevron" data-lucide="chevron-right"></i>' : "<span></span>"}
              </button>
              ${expandable ? `<div class="test-detail hidden" id="test-detail-${index}">${renderInputVariables(test.inputs)}</div>` : ""}
            </div>`;
        },
      )
      .join("");
    elements.testResults.querySelectorAll(".test-row.expandable").forEach((button) => {
      button.addEventListener("click", () => {
        const detail = document.querySelector(`#test-detail-${button.dataset.testIndex}`);
        const opening = detail.classList.contains("hidden");
        detail.classList.toggle("hidden", !opening);
        button.classList.toggle("open", opening);
      });
    });
    elements.testResults.classList.remove("hidden");
  } else {
    elements.testResults.classList.add("hidden");
  }

  const artifactBase = `/api/jobs/${job.id}/artifacts`;
  const links = [
    ["execution", "执行日志", "file-down"],
    ["result", "JSON 结果", "braces"],
  ];
  if (result?.nsight_compute?.status === "passed") {
    links.push(["nsight", "NCU 报告", "bar-chart-3"], ["nsight-csv", "NCU CSV", "table"]);
  }
  const downloads = links
    .map(
      ([kind, label, icon]) =>
        `<a class="artifact-link" href="${artifactBase}/${kind}"><i data-lucide="${icon}"></i>${label}</a>`,
    )
    .join("");
  const ncuActions = result?.nsight_compute?.status === "passed"
    ? `<button class="artifact-button" data-ncu-action="open"><i data-lucide="external-link"></i>打开 NCU</button>
       <button class="artifact-button" data-ncu-action="save"><i data-lucide="save"></i>保存 NCU</button>`
    : "";
  elements.artifactBar.innerHTML = downloads + ncuActions;
  elements.artifactBar.querySelector('[data-ncu-action="open"]')?.addEventListener("click", () =>
    openNsight(job.id),
  );
  elements.artifactBar.querySelector('[data-ncu-action="save"]')?.addEventListener("click", () =>
    saveNsight(job.id),
  );
  elements.artifactBar.classList.remove("hidden");
  elements.executionOutput.textContent = job.output || "";
  elements.queueState.textContent = "空闲";
  elements.runButton.disabled = false;
  elements.testButton.disabled = false;
  elements.resultIndicator.classList.remove("visible");
  refreshIcons();
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    state.activeJob = job;
    if (job.status === "queued" || job.status === "running") {
      renderRunning(job);
      state.pollTimer = setTimeout(() => pollJob(jobId), 650);
      return;
    }
    renderCompleted(job);
    if (job.mode === "submit") {
      await Promise.all([loadLeaderboard(), refreshChallengeProgress()]);
    }
  } catch (error) {
    elements.runButton.disabled = false;
    elements.testButton.disabled = false;
    elements.queueState.textContent = "错误";
    toast(error.message, "error");
  }
}

async function submitRun(mode) {
  if (!state.currentKey || !elements.codeEditor.value.trim()) return;
  elements.runButton.disabled = true;
  elements.testButton.disabled = true;
  localStorage.setItem("leetgpu.runner", elements.runnerName.value.trim() || "local");
  saveDraft();
  setView("results");
  elements.metricsStrip.classList.add("hidden");
  elements.compilerPanel.classList.add("hidden");
  elements.testResults.classList.add("hidden");
  elements.artifactBar.classList.add("hidden");
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        challenge: state.currentKey,
        language: state.language,
        source: elements.codeEditor.value,
        runner_name: elements.runnerName.value,
        profile: elements.profileEnabled.checked,
        mode,
      }),
    });
    state.activeJob = job;
    renderRunning(job);
    if (mode === "submit") await refreshChallengeProgress();
    pollJob(job.id);
  } catch (error) {
    elements.runButton.disabled = false;
    elements.testButton.disabled = false;
    toast(error.message, "error");
  }
}

async function refreshChallengeProgress() {
  const data = await api("/api/challenges");
  state.challenges = data.challenges;
  renderChallengeList();
}

async function openSubmission(jobId) {
  try {
    const submission = await api(`/api/leaderboard/source?job=${encodeURIComponent(jobId)}`);
    state.selectedSubmission = submission;
    elements.submissionModalTitle.textContent = `${submission.challenge_name} · ${submission.runner_name}`;
    elements.submissionModalMeta.textContent =
      `${submission.language.toUpperCase()} · ${formatMs(submission.median_ms)} · ${submission.solution_filename}`;
    elements.submissionSource.textContent = submission.source_code;
    elements.submissionModal.showModal();
    refreshIcons();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function deleteSubmission(jobId) {
  if (!window.confirm("删除这条排行榜成绩？")) return;
  try {
    await api(`/api/leaderboard/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    if (elements.submissionModal.open) elements.submissionModal.close();
    state.selectedSubmission = null;
    await loadLeaderboard();
    toast("排行榜成绩已删除");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function openNsight(jobId) {
  try {
    await api(`/api/jobs/${jobId}/open-nsight`, { method: "POST", body: "{}" });
    toast("已打开 Nsight Compute");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveNsight(jobId) {
  try {
    const result = await api(`/api/jobs/${jobId}/export-nsight`, {
      method: "POST",
      body: "{}",
    });
    toast(`已保存到 ${result.destination}`);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadLeaderboard() {
  if (!state.currentKey) return;
  const language = elements.leaderboardLanguage.value;
  try {
    const data = await api(
      `/api/leaderboard?challenge=${encodeURIComponent(state.currentKey)}&language=${encodeURIComponent(language)}`,
    );
    elements.leaderboardBody.innerHTML = data.scores
      .map(
        (score) => `
          <tr class="leaderboard-row" data-job-id="${score.job_id}">
            <td class="rank-cell ${score.rank <= 3 ? "top" : ""}">#${score.rank}</td>
            <td>${escapeHtml(score.runner_name)}</td>
            <td class="username-label">${escapeHtml(score.username || "—")}</td>
            <td class="language-label">${escapeHtml(score.language)}</td>
            <td>${formatMs(score.median_ms)}</td>
            <td>${formatMs(score.min_ms)}</td>
            <td>${new Date(score.created_at).toLocaleString()}</td>
            <td class="leaderboard-actions">
              <button class="icon-button" data-delete-job="${score.job_id}" title="删除成绩" aria-label="删除成绩">
                <i data-lucide="trash-2"></i>
              </button>
            </td>
          </tr>`,
      )
      .join("");
    elements.leaderboardBody.querySelectorAll(".leaderboard-row").forEach((row) => {
      row.addEventListener("click", () => openSubmission(row.dataset.jobId));
    });
    elements.leaderboardBody.querySelectorAll("[data-delete-job]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSubmission(button.dataset.deleteJob);
      });
    });
    elements.leaderboardEmpty.classList.toggle("hidden", data.scores.length > 0);
    refreshIcons();
  } catch (error) {
    toast(error.message, "error");
  }
}

// ---------- announcements & admin ----------

async function loadAnnouncements() {
  try {
    const data = await api("/api/announcements");
    const items = data.announcements || [];
    elements.announcementsBar.classList.toggle("hidden", items.length === 0);
    elements.announcementsBar.innerHTML = items
      .slice(0, 3)
      .map(
        (a) => `
        <div class="announcement">
          <span class="announcement-title">${escapeHtml(a.title)}</span>
          <span class="announcement-body">${escapeHtml(a.body)}</span>
          <span class="announcement-meta">${escapeHtml(a.author)} · ${new Date(a.created_at).toLocaleDateString()}</span>
        </div>`,
      )
      .join("");
  } catch {}
}

async function loadAdminAnnouncements() {
  try {
    const data = await api("/api/announcements");
    elements.adminAnnouncementsList.innerHTML = (data.announcements || [])
      .map(
        (a) => `
        <div class="admin-row">
          <span class="admin-row-main">${escapeHtml(a.title)}</span>
          <button class="icon-button" data-del-ann="${a.id}" title="删除">
            <i data-lucide="trash-2"></i>
          </button>
        </div>`,
      )
      .join("");
    elements.adminAnnouncementsList
      .querySelectorAll("[data-del-ann]")
      .forEach((b) =>
        b.addEventListener("click", async () => {
          await api(`/api/admin/announcements/${b.dataset.delAnn}`, { method: "DELETE" });
          await loadAdminAnnouncements();
          await loadAnnouncements();
        }),
      );
    refreshIcons();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadAdminUsers() {
  try {
    const data = await api("/api/admin/users");
    elements.adminUsersList.innerHTML = data.users
      .map(
        (u) => `
        <div class="admin-row">
          <span class="admin-row-main">
            ${escapeHtml(u.username)}
            <span class="admin-role">${u.role}${u.disabled ? " · 已禁用" : ""}</span>
          </span>
          <button class="icon-button" data-toggle-user="${escapeHtml(u.username)}" data-disabled="${u.disabled ? 1 : 0}"
                  title="${u.disabled ? "启用" : "禁用"}">
            <i data-lucide="${u.disabled ? "user-check" : "user-x"}"></i>
          </button>
        </div>`,
      )
      .join("");
    elements.adminUsersList
      .querySelectorAll("[data-toggle-user]")
      .forEach((b) =>
        b.addEventListener("click", async () => {
          const action = b.dataset.disabled === "1" ? "enable" : "disable";
          await api(`/api/admin/users/${b.dataset.toggleUser}/${action}`, { method: "DELETE" });
          await loadAdminUsers();
        }),
      );
    refreshIcons();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function openAdmin() {
  elements.adminModal.showModal();
  elements.registrationOpen.checked = session.registrationOpen;
  await loadAdminAnnouncements();
  await loadAdminUsers();
  refreshIcons();
}

async function saveSettings() {
  try {
    const settings = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({ ncu_export_dir: elements.ncuExportDir.value }),
    });
    state.settings = settings;
    elements.ncuExportDir.value = settings.ncu_export_dir;
    elements.settingsModal.close();
    toast("设置已保存");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function pickNcuDirectory() {
  elements.pickNcuDirectory.disabled = true;
  try {
    const settings = await api("/api/settings/pick-directory", {
      method: "POST",
      body: "{}",
    });
    state.settings = settings;
    elements.ncuExportDir.value = settings.ncu_export_dir;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    elements.pickNcuDirectory.disabled = false;
  }
}

async function initialize() {
  refreshIcons();
  elements.runnerName.value = localStorage.getItem("leetgpu.runner") || "local";
  await refreshSession().catch(() => {});
  if (session.authRequired && !session.user) {
    openAuthModal();
    return;
  }
  loadAnnouncements();
  try {
    const [challengeData, system, settings] = await Promise.all([
      api("/api/challenges"),
      api("/api/system"),
      api("/api/settings"),
    ]);
    state.settings = settings;
    elements.ncuExportDir.value = settings.ncu_export_dir;
    state.challenges = challengeData.challenges;
    if (system.available) {
      elements.systemStatus.classList.add("ready");
      elements.systemStatus.querySelector(".system-name").textContent =
        `${system.gpu} · ${Math.round(system.memory_mb / 1024)} GB`;
    } else {
      elements.systemStatus.querySelector(".system-name").textContent = "GPU 不可用";
    }
    renderChallengeList();
    const saved = localStorage.getItem("leetgpu.challenge");
    const initial = state.challenges.some((item) => item.key === saved)
      ? saved
      : state.challenges[0]?.key;
    await selectChallenge(initial);
  } catch (error) {
    toast(error.message, "error");
  }
}

elements.challengeSearch.addEventListener("input", renderChallengeList);
document.querySelectorAll(".difficulty-filter button").forEach((button) => {
  button.addEventListener("click", () => {
    state.difficulty = button.dataset.difficulty;
    document.querySelectorAll(".difficulty-filter button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderChallengeList();
  });
});
elements.languageSwitch.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
});
elements.viewTabs.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
elements.codeEditor.addEventListener("input", () => recordEditorEdit());
elements.codeEditor.addEventListener("beforeinput", (event) => {
  if (event.inputType === "historyUndo" || event.inputType === "historyRedo") {
    event.preventDefault();
    if (event.inputType === "historyUndo") undoEditor();
    else redoEditor();
  }
});
elements.codeEditor.addEventListener("scroll", () => {
  elements.lineNumbers.scrollTop = elements.codeEditor.scrollTop;
});
elements.codeEditor.addEventListener("keydown", (event) => {
  const modifier = event.ctrlKey || event.metaKey;
  if (modifier && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redoEditor();
    else undoEditor();
    return;
  }
  if (modifier && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redoEditor();
    return;
  }
  if (modifier && (event.key === "/" || event.code === "Slash")) {
    event.preventDefault();
    toggleLineComments();
    return;
  }
  if (event.key !== "Tab") return;
  event.preventDefault();
  const start = elements.codeEditor.selectionStart;
  const end = elements.codeEditor.selectionEnd;
  elements.codeEditor.setRangeText("    ", start, end, "end");
  elements.codeEditor.dispatchEvent(new Event("input"));
});
elements.undoCode.addEventListener("click", undoEditor);
elements.redoCode.addEventListener("click", redoEditor);
elements.resetCode.addEventListener("click", () => {
  if (!window.confirm("恢复当前语言的 Starter？")) return;
  localStorage.removeItem(draftKey());
  loadCode();
});
elements.testButton.addEventListener("click", () => submitRun("test"));
elements.runButton.addEventListener("click", () => submitRun("submit"));
elements.leaderboardLanguage.addEventListener("change", loadLeaderboard);
elements.sidebarToggle.addEventListener("click", () => elements.sidebar.classList.toggle("open"));
elements.settingsButton.addEventListener("click", () => {
  elements.ncuExportDir.value = state.settings?.ncu_export_dir || "";
  elements.settingsModal.showModal();
  refreshIcons();
});
elements.saveSettings.addEventListener("click", saveSettings);
elements.pickNcuDirectory.addEventListener("click", pickNcuDirectory);
elements.deleteSubmission.addEventListener("click", () => {
  if (state.selectedSubmission) deleteSubmission(state.selectedSubmission.job_id);
});
elements.copySubmission.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.submissionSource.textContent);
    toast("代码已复制");
  } catch {
    toast("无法访问剪贴板", "error");
  }
});
document.querySelectorAll("[data-close-dialog]").forEach((button) => {
  button.addEventListener("click", () => document.querySelector(`#${button.dataset.closeDialog}`).close());
});
document.querySelectorAll("dialog").forEach((dialog) => {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
});

// ---------- auth / admin / vim listeners ----------

elements.loginButton.addEventListener("click", () => openAuthModal("login"));
elements.logoutButton.addEventListener("click", logout);
elements.authForm.addEventListener("submit", submitAuth);
elements.authSwitch.addEventListener("click", () => {
  openAuthModal(session.authMode === "login" ? "register" : "login");
});
elements.adminButton.addEventListener("click", openAdmin);
elements.announcementForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/admin/announcements", {
      method: "POST",
      body: JSON.stringify({
        title: elements.announcementTitle.value,
        body: elements.announcementBody.value,
      }),
    });
    elements.announcementTitle.value = "";
    elements.announcementBody.value = "";
    await loadAdminAnnouncements();
    await loadAnnouncements();
    toast("公告已发布");
  } catch (error) {
    toast(error.message, "error");
  }
});
elements.registrationOpen.addEventListener("change", async () => {
  try {
    const res = await api("/api/admin/settings", {
      method: "POST",
      body: JSON.stringify({ registration_open: elements.registrationOpen.checked }),
    });
    session.registrationOpen = res.registration_open;
    toast(res.registration_open ? "已开放注册" : "已关闭注册");
  } catch (error) {
    toast(error.message, "error");
    elements.registrationOpen.checked = !elements.registrationOpen.checked;
  }
});
document.querySelectorAll("[data-admin-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-admin-tab]").forEach((b) =>
      b.classList.toggle("active", b === button),
    );
    document.querySelectorAll(".admin-pane").forEach((p) =>
      p.classList.toggle("active", p.id === `admin-${button.dataset.adminTab}`),
    );
  });
});
elements.vimToggle.addEventListener("click", () => {
  const on = window.LeetGPUVim && window.LeetGPUVim.toggle();
  elements.vimToggle.classList.toggle("active", !!on);
  elements.vimMode.classList.toggle("hidden", !on);
  toast(on ? "Vim 模式已开启" : "Vim 模式已关闭");
});
if (window.LeetGPUVim && window.LeetGPUVim.enabled) {
  elements.vimToggle.classList.add("active");
  elements.vimMode.classList.remove("hidden");
}

initialize();
