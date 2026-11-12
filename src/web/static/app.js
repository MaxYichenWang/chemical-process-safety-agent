/**
 * Frontend for the Chemical Process Safety LLM web UI.
 *
 * Multi-round ChatGPT/Gemini-style conversation with right-aligned user bubbles,
 * left-aligned assistant responses, and real-time agent pipeline streaming.
 */

let META = null;

// ---------------------------------------------------------------------
// Reducer (port of src/web/reducer.py)
// ---------------------------------------------------------------------

function initialState() {
  return {
    phase: "idle",
    completedSteps: [],
    activeStep: null,
    activeWorkers: [],
    workerGroupStep: 0,
    finalState: null,
    errorMessage: null,
    errorTraceback: null,
    _stepCount: 0,
    _lastElapsedMs: 0,
  };
}

function detectSupervisorPhase(update) {
  if (update.sub_tasks && update.sub_tasks.length) return "Plan";
  if (update.draft_response) return "Synthesize";
  return "Dispatch";
}

function predictNextNode(nodeName, update) {
  if (nodeName === "CondenseNode") return "PlanNode";
  if (nodeName === "PlanNode") return null;
  if (META && META.worker_nodes && META.worker_nodes.includes(nodeName)) return "SynthesizeNode";
  if (nodeName === "SynthesizeNode") return "CriticNode";

  if (nodeName === "Supervisor") {
    const active = update.active_agent || "";
    if (active === "CRITIC") return "CriticNode";
    if (active === "FINISH" || active === "") return null;
    return (META && META.agent_key_to_node && META.agent_key_to_node[active]) || null;
  }

  if (nodeName === "CriticNode") {
    const score = update.validation_score || 0;
    return score >= (META ? META.validation_score_threshold : 0.7) ? "FinalizeNode" : "RetryNode";
  }

  if (nodeName === "RetryNode") return "RevisionNode";
  if (nodeName === "RevisionNode") return "CriticNode";

  return null;
}

function reduceEvent(state, event) {
  if (!event || typeof event !== "object") return state;
  const kind = event.event;
  const now = Date.now();

  if (kind === "graph_ready") {
    return { ...state, phase: "running" };
  }

  if (kind === "step_start") {
    const node = event.node;
    if (state.activeWorkers.some((w) => w.node === node)) return state;
    if (state.activeStep && state.activeStep.node === node) return state;
    const stepCount = state._stepCount + 1;
    return {
      ...state,
      _stepCount: stepCount,
      activeStep: { stepNumber: stepCount, node, startedAtMs: now },
    };
  }

  if (kind === "step_complete") {
    return reduceStepComplete(state, event, now);
  }

  if (kind === "pipeline_complete") {
    return { ...state, phase: "complete", activeStep: null, finalState: event.state };
  }

  if (kind === "error") {
    return { ...state, phase: "error", activeStep: null, errorMessage: event.message, errorTraceback: event.traceback };
  }

  return state;
}

function reduceStepComplete(state, event, now) {
  const node = event.node;
  const update = event.update || {};
  const elapsedMs = event.elapsed_ms || 0;
  const stepDuration = elapsedMs - state._lastElapsedMs;

  const planned = update.planned_agents;
  if (node === "PlanNode" && planned && planned.length) {
    const workers = planned.map((agentKey) => ({
      node: (META && META.agent_key_to_node && META.agent_key_to_node[agentKey]) || agentKey,
      status: "active",
      startedAtMs: now,
    }));
    const completed = {
      stepNumber: (state.activeStep || {}).stepNumber ?? state._stepCount,
      node,
      elapsedMs,
      stepDurationMs: stepDuration,
      update,
      errors: event.errors,
    };
    const currStep = completed.stepNumber;
    return {
      ...state,
      _stepCount: currStep,
      _lastElapsedMs: elapsedMs,
      completedSteps: [...state.completedSteps, completed],
      activeStep: null,
      activeWorkers: workers,
      workerGroupStep: currStep + 1,
    };
  }

  const isWorker = META && META.worker_nodes && META.worker_nodes.includes(node);
  if (isWorker && state.activeWorkers.length) {
    const workerDuration = event.step_duration_s != null ? Math.round(event.step_duration_s * 1000) : Math.max(0, elapsedMs - state._lastElapsedMs);
    const updatedWorkers = state.activeWorkers.map((w) =>
      w.node === node
        ? {
            ...w,
            status: "done",
            completedAtMs: now,
            durationMs: now - w.startedAtMs,
            update,
            errors: event.errors,
          }
        : w
    );
    const allDone = updatedWorkers.every((w) => w.status === "done");

    const stepCount = state._stepCount + 1;
    const workerCompleted = {
      stepNumber: stepCount,
      node,
      elapsedMs,
      stepDurationMs: workerDuration,
      update,
      errors: event.errors,
    };

    let nextActive = null;
    let finalStepCount = stepCount;
    if (allDone) {
      finalStepCount = stepCount + 1;
      nextActive = { stepNumber: finalStepCount, node: "SynthesizeNode", startedAtMs: now };
    }

    return {
      ...state,
      _stepCount: finalStepCount,
      _lastElapsedMs: elapsedMs,
      completedSteps: [...state.completedSteps, workerCompleted],
      activeWorkers: updatedWorkers,
      activeStep: nextActive,
    };
  }

  const completed = {
    stepNumber: (state.activeStep || {}).stepNumber ?? state._stepCount,
    node,
    elapsedMs,
    stepDurationMs: stepDuration,
    update,
    errors: event.errors,
  };

  const nextNode = predictNextNode(node, update);
  let nextActive = null;
  let stepCount = state._stepCount;
  if (nextNode) {
    stepCount += 1;
    nextActive = { stepNumber: stepCount, node: nextNode, startedAtMs: now };
  }

  return {
    ...state,
    _stepCount: stepCount,
    _lastElapsedMs: elapsedMs,
    completedSteps: [...state.completedSteps, completed],
    activeStep: nextActive,
  };
}

// ---------------------------------------------------------------------
// Session & Turn Data Structures
// ---------------------------------------------------------------------

const MAX_HISTORY_TURNS = 5;

let currentSessionId = null; // Currently active/selected session ID
let currentEventSource = null; // Active SSE connection
let tickTimer = null;
let conversationHistory = []; // Active conversation turns: [{ query, response }]
let pendingDeleteSessionId = null; // Session ID awaiting delete confirmation modal

const knownSessions = new Map();
// sessionId -> { id, query, timestamp, status, turns: [ { run_id, query, state, startMs } ] }

function getSessionHistory(session) {
  if (session && Array.isArray(session.turns) && session.turns.length > 0) {
    const completed = [];
    for (const t of session.turns) {
      const fs = t.state && t.state.finalState;
      const resp =
        (fs && fs.final_response) ||
        t.streamingText ||
        (t.dom && t.dom.answerPane && t.dom.answerPane.innerText) ||
        "";
      if (t.query && resp) {
        completed.push({ query: t.query, response: resp });
      }
    }
    if (completed.length > 0) {
      return completed.slice(-MAX_HISTORY_TURNS);
    }
  }
  return conversationHistory.slice(-MAX_HISTORY_TURNS);
}

function nodeColorClass(node) {
  if (META && META.worker_nodes && META.worker_nodes.includes(node)) return "node-worker";
  return `node-${node}`;
}

function fmtSeconds(ms) {
  return (ms / 1000).toFixed(1);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

function scrollToBottom() {
  const area = document.getElementById("chat-scroll-area");
  if (area) {
    area.scrollTop = area.scrollHeight;
  }
}

function startTicking() {
  stopTicking();
  tickTimer = setInterval(() => {
    document.querySelectorAll("[data-tick-since]").forEach((el) => {
      const since = Number(el.dataset.tickSince);
      if (!since) return;
      el.textContent = `[${fmtSeconds(Date.now() - since).padStart(6)}s]`;
    });
  }, 200);
}

function stopTicking() {
  if (tickTimer) clearInterval(tickTimer);
  tickTimer = null;
}

// ---------------------------------------------------------------------
// Turn DOM Construction & Rendering
// ---------------------------------------------------------------------

function createTurnElements(turn) {
  const turnDiv = document.createElement("div");
  turnDiv.className = "chat-turn";
  turnDiv.id = `turn-${turn.run_id}`;

  // User message on the RIGHT (like ChatGPT / Gemini)
  const userRow = document.createElement("div");
  userRow.className = "user-message-row";
  const userBubble = document.createElement("div");
  userBubble.className = "user-bubble";
  userBubble.textContent = turn.query;
  userRow.appendChild(userBubble);
  turnDiv.appendChild(userRow);

  // Assistant response on the LEFT
  const asstRow = document.createElement("div");
  asstRow.className = "assistant-message-row";

  // Pipeline card (collapsible, collapsed by default if finished)
  const isFinished = turn.isFinished || (turn.state && (turn.state.phase === "complete" || turn.state.phase === "error" || turn.state.finalState));
  const pipelineCard = document.createElement("details");
  pipelineCard.className = "pipeline-card";
  pipelineCard.open = !isFinished;

  const summary = document.createElement("summary");
  summary.className = "pipeline-summary";
  summary.innerHTML = `
    <span class="pipeline-summary-left">
      <span class="pipeline-toggle-icon">▸</span>
      <span>Agent Execution Pipeline</span>
    </span>
    <span class="pipeline-status-pill in_progress">Running…</span>
  `;

  const pipelineContent = document.createElement("div");
  pipelineContent.className = "pipeline-content";
  pipelineContent.innerHTML = `
    <div class="pipeline-tree"></div>
    <div class="workers-panel"></div>
  `;

  pipelineCard.appendChild(summary);
  pipelineCard.appendChild(pipelineContent);
  asstRow.appendChild(pipelineCard);

  // Low confidence banner
  const warningBanner = document.createElement("div");
  warningBanner.className = "banner banner-warning";
  warningBanner.hidden = true;
  asstRow.appendChild(warningBanner);

  // Answer pane
  const answerPane = document.createElement("div");
  answerPane.className = "answer-pane";
  asstRow.appendChild(answerPane);

  // Citations / References wrapper (collapsible details, collapsed by default)
  const citationsWrapper = document.createElement("details");
  citationsWrapper.className = "references-details";
  citationsWrapper.hidden = true;
  citationsWrapper.innerHTML = `
    <summary class="references-summary">
      <span class="references-summary-left">
        <span class="references-toggle-icon">▸</span>
        <span>References</span>
      </span>
      <span class="references-count"></span>
    </summary>
    <div class="references-content">
      <table class="citations-table">
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  `;
  asstRow.appendChild(citationsWrapper);

  // Response Action Bar (Copy, Export .md, Export PDF)
  const responseActions = document.createElement("div");
  responseActions.className = "response-actions";
  responseActions.hidden = true;
  responseActions.innerHTML = `
    <button class="action-btn copy-btn" title="Copy response markdown to clipboard">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span class="btn-label">Copy</span>
    </button>
    <button class="action-btn export-md-btn" title="Download response as Markdown (.md)">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
      </svg>
      <span class="btn-label">Export .md</span>
    </button>
    <button class="action-btn export-pdf-btn" title="Export or print response as PDF">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M6 9V2h12v7"></path>
        <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
        <rect x="6" y="14" width="12" height="8"></rect>
      </svg>
      <span class="btn-label">Export PDF</span>
    </button>
  `;
  asstRow.appendChild(responseActions);

  // Copy button handler
  const copyBtn = responseActions.querySelector(".copy-btn");
  copyBtn.addEventListener("click", async () => {
    const fs = turn.state && turn.state.finalState;
    const textToCopy = (fs && fs.final_response) || answerPane.innerText || "";
    if (!textToCopy) return;

    const label = copyBtn.querySelector(".btn-label");
    const origText = label.textContent;
    try {
      await navigator.clipboard.writeText(textToCopy);
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = textToCopy;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    label.textContent = "Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      label.textContent = origText;
      copyBtn.classList.remove("copied");
    }, 2000);
  });

  // Export Markdown handler
  const exportMdBtn = responseActions.querySelector(".export-md-btn");
  exportMdBtn.addEventListener("click", () => {
    const fs = turn.state && turn.state.finalState;
    const text = (fs && fs.final_response) || answerPane.innerText || "";
    if (!text) return;
    const citations = (fs && fs.citations) || [];

    let md = `# Chemical Process Safety Report\n\n`;
    md += `**Query:** ${turn.query}\n\n`;
    md += `**Date:** ${new Date().toLocaleString()}\n\n`;
    md += `---\n\n`;
    md += `## Response\n\n${text}\n\n`;

    if (citations.length > 0) {
      md += `---\n\n## References\n\n`;
      const hasPage = citations.some((c) => c.page != null && String(c.page).trim() !== "" && String(c.page).trim().toLowerCase() !== "none");
      if (hasPage) {
        md += `| Collection | Source | Page |\n|---|---|---|\n`;
        for (const c of citations) {
          md += `| ${c.collection || "N/A"} | ${c.source || "N/A"} | ${c.page || "N/A"} |\n`;
        }
      } else {
        md += `| Collection | Source |\n|---|---|\n`;
        for (const c of citations) {
          md += `| ${c.collection || "N/A"} | ${c.source || "N/A"} |\n`;
        }
      }
    }

    const slug = (turn.query || "report").toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 32);
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `process_safety_${slug}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Export PDF handler
  const exportPdfBtn = responseActions.querySelector(".export-pdf-btn");
  exportPdfBtn.addEventListener("click", () => {
    const fs = turn.state && turn.state.finalState;
    const htmlContent = answerPane.innerHTML;
    if (!htmlContent) return;
    const queryEsc = escapeHtml(turn.query || "Chemical Process Safety Report");
    const citations = (fs && fs.citations) || [];

    let refsHtml = "";
    if (citations.length > 0) {
      const hasPage = citations.some((c) => c.page != null && String(c.page).trim() !== "" && String(c.page).trim().toLowerCase() !== "none");
      refsHtml += `<h3 style="margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:1rem;">References</h3><table style="width:100%;border-collapse:collapse;margin-top:0.5rem;font-size:0.85rem;"><thead><tr style="background:#f1f5f9;"><th style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;text-align:left;">Collection</th><th style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;text-align:left;">Source</th>`;
      if (hasPage) refsHtml += `<th style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;text-align:left;">Page</th>`;
      refsHtml += `</tr></thead><tbody>`;
      for (const c of citations) {
        refsHtml += `<tr><td style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;">${escapeHtml(c.collection || "N/A")}</td><td style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;">${escapeHtml(c.source || "N/A")}</td>`;
        if (hasPage) {
          const pg = (c.page != null && String(c.page).trim() !== "" && String(c.page).trim().toLowerCase() !== "none") ? escapeHtml(String(c.page)) : "N/A";
          refsHtml += `<td style="border:1px solid #cbd5e1;padding:0.4rem 0.6rem;">${pg}</td>`;
        }
        refsHtml += `</tr>`;
      }
      refsHtml += `</tbody></table>`;
    }

    const printWin = window.open("", "_blank");
    if (!printWin) {
      window.print();
      return;
    }

    printWin.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>${queryEsc} — Chemical Process Safety LLM</title>
        <style>
          @page { margin: 18mm; size: auto; }
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #0f172a;
            line-height: 1.6;
            max-width: 820px;
            margin: 0 auto;
            padding: 1.5rem;
          }
          h1.header-title { font-size: 1.35rem; color: #0284c7; margin-bottom: 0.5rem; }
          .meta-header { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem; font-size: 0.88rem; }
          .meta-row { margin-bottom: 0.25rem; }
          .meta-row:last-child { margin-bottom: 0; }
          .meta-label { font-weight: 600; color: #475569; margin-right: 0.4rem; }
          .response-body { font-size: 0.95rem; }
          table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }
          th, td { border: 1px solid #cbd5e1; padding: 0.4rem 0.6rem; text-align: left; }
          th { background: #f1f5f9; font-weight: 600; }
          blockquote { border-left: 3px solid #0284c7; margin-left: 0; padding-left: 1rem; color: #475569; }
          @media print {
            body { padding: 0; }
          }
        </style>
      </head>
      <body>
        <h1 class="header-title">Chemical Process Safety LLM</h1>
        <div class="meta-header">
          <div class="meta-row"><span class="meta-label">Query:</span> ${queryEsc}</div>
          <div class="meta-row"><span class="meta-label">Date:</span> ${new Date().toLocaleString()}</div>
        </div>
        <div class="response-body">
          ${htmlContent}
        </div>
        ${refsHtml}
        <script>
          window.onload = function() {
            window.print();
          };
        </script>
      </body>
      </html>
    `);
    printWin.document.close();
  });

  // Error banner
  const errorBanner = document.createElement("div");
  errorBanner.className = "banner banner-error";
  errorBanner.hidden = true;
  errorBanner.innerHTML = `
    <div class="error-msg"></div>
    <details class="error-details" hidden>
      <summary>Technical details</summary>
      <pre class="error-trace"></pre>
    </details>
  `;
  asstRow.appendChild(errorBanner);

  turnDiv.appendChild(asstRow);

  return {
    root: turnDiv,
    pipelineCard,
    statusPill: summary.querySelector(".pipeline-status-pill"),
    tree: pipelineContent.querySelector(".pipeline-tree"),
    workers: pipelineContent.querySelector(".workers-panel"),
    warningBanner,
    answerPane,
    citationsWrapper,
    referencesCount: citationsWrapper.querySelector(".references-count"),
    citationsThead: citationsWrapper.querySelector("thead"),
    citationsBody: citationsWrapper.querySelector("tbody"),
    responseActions,
    errorBanner,
    errorMsg: errorBanner.querySelector(".error-msg"),
    errorDetails: errorBanner.querySelector(".error-details"),
    errorTrace: errorBanner.querySelector(".error-trace"),
  };
}

function renderTurn(turn) {
  if (!turn.dom) return;
  const dom = turn.dom;
  const state = turn.state;

  // Status pill & auto-collapse execution pipeline when finished
  if (state.phase === "complete") {
    dom.statusPill.textContent = "Completed";
    dom.statusPill.className = "pipeline-status-pill done";
    if (!turn._hasAutoCollapsed) {
      turn._hasAutoCollapsed = true;
      dom.pipelineCard.open = false;
    }
  } else if (state.phase === "error") {
    dom.statusPill.textContent = "Failed";
    dom.statusPill.className = "pipeline-status-pill error";
    if (!turn._hasAutoCollapsed) {
      turn._hasAutoCollapsed = true;
      dom.pipelineCard.open = false;
    }
  } else {
    dom.statusPill.textContent = "Running…";
    dom.statusPill.className = "pipeline-status-pill in_progress";
  }

  // Pipeline tree
  dom.tree.innerHTML = "";
  for (const step of state.completedSteps) {
    dom.tree.appendChild(renderCompletedStepRow(step));
  }
  if (state.activeStep) {
    dom.tree.appendChild(renderActiveStepRow(state.activeStep, turn.startMs));
  } else if (state.phase === "running" && !state.activeWorkers.length) {
    dom.tree.appendChild(renderInitializingRow(turn.startMs));
  }

  // Workers panel
  dom.workers.innerHTML = "";
  if (state.activeWorkers && state.activeWorkers.length) {
    for (const w of state.activeWorkers) {
      const card = document.createElement("div");
      card.className = `worker-card ${w.status}`;
      const label = (META && META.node_labels && META.node_labels[w.node]) || w.node;
      card.innerHTML = `
        <span class="wlabel">${w.status === "done" ? "\u2713" : "\u25CF"} ${escapeHtml(label)}</span>
        <span class="wdetail">${escapeHtml(workerDetail(w))}</span>
      `;
      dom.workers.appendChild(card);
    }
  }

  // Answer pane
  if (state.finalState) {
    const fs = state.finalState;
    const score = fs.validation_score;
    if (typeof score === "number" && META && score < META.validation_score_threshold) {
      dom.warningBanner.hidden = false;
      dom.warningBanner.textContent = `\u26A0 Low confidence (score ${score.toFixed(2)}/1.00) — this response may not be fully validated against the retrieved sources. Verify critical details before relying on it.`;
    } else {
      dom.warningBanner.hidden = true;
    }

    dom.answerPane.innerHTML = fs.final_response_html || "";

    const citations = fs.citations || [];
    if (citations.length) {
      dom.citationsWrapper.hidden = false;
      dom.referencesCount.textContent = `${citations.length} source${citations.length === 1 ? "" : "s"}`;

      // Check if any citation has a non-empty page value
      const hasPage = citations.some((c) => {
        if (c.page == null) return false;
        const str = String(c.page).trim();
        return str !== "" && str.toLowerCase() !== "none" && str.toLowerCase() !== "null";
      });

      // Build header: Collection, Source, (optional) Page. Chunk ID is removed.
      let headHtml = "<tr><th>Collection</th><th>Source</th>";
      if (hasPage) {
        headHtml += "<th>Page</th>";
      }
      headHtml += "</tr>";
      dom.citationsThead.innerHTML = headHtml;

      // Build body: if page column displayed and value empty, put "N/A"
      dom.citationsBody.innerHTML = "";
      for (const c of citations) {
        const tr = document.createElement("tr");
        const coll = escapeHtml(c.collection || "N/A");
        const src = escapeHtml(c.source || "N/A");
        let rowHtml = `<td>${coll}</td><td>${src}</td>`;
        if (hasPage) {
          let pageVal = "N/A";
          if (c.page != null) {
            const str = String(c.page).trim();
            if (str !== "" && str.toLowerCase() !== "none" && str.toLowerCase() !== "null") {
              pageVal = escapeHtml(str);
            }
          }
          rowHtml += `<td>${pageVal}</td>`;
        }
        tr.innerHTML = rowHtml;
        dom.citationsBody.appendChild(tr);
      }
    } else {
      dom.citationsWrapper.hidden = true;
    }

    if (dom.responseActions) dom.responseActions.hidden = false;
  } else {
    dom.warningBanner.hidden = true;
    if (typeof turn.streamingHtml === "string" && turn.streamingHtml) {
      dom.answerPane.innerHTML = turn.streamingHtml;
    } else if (!turn.streamingText) {
      dom.answerPane.innerHTML = "";
    }
    dom.citationsWrapper.hidden = true;
    if (dom.responseActions) dom.responseActions.hidden = true;
  }

  // Error banner
  if (state.phase === "error") {
    dom.errorBanner.hidden = false;
    dom.errorMsg.textContent = state.errorMessage || "The pipeline encountered an error.";
    if (state.errorTraceback) {
      dom.errorDetails.hidden = false;
      dom.errorTrace.textContent = state.errorTraceback;
    } else {
      dom.errorDetails.hidden = true;
    }
  } else {
    dom.errorBanner.hidden = true;
  }
}

function renderCompletedStepRow(step) {
  const row = document.createElement("div");
  row.className = "step-row";
  const hasErrors = step.errors && step.errors.length;
  const iconClass = hasErrors ? "error" : "done";
  const icon = hasErrors ? "⚠" : "✓";
  let label = (META && META.node_labels && META.node_labels[step.node]) || step.node;
  if (step.node === "Supervisor") label += ` (${detectSupervisorPhase(step.update)})`;
  if (
    step.node === "CondenseNode" &&
    step.update &&
    step.update.query &&
    step.update.raw_query &&
    step.update.query !== step.update.raw_query
  ) {
    label += ` \u2192 "${step.update.query}"`;
  }

  row.innerHTML = `
    <span class="step-elapsed" title="Cumulative elapsed time from query start: ${fmtSeconds(step.elapsedMs)}s">[${fmtSeconds(step.elapsedMs).padStart(6)}s]</span>
    <span class="step-icon ${iconClass}">${icon}</span>
    <span class="step-label ${nodeColorClass(step.node)}">Step ${step.stepNumber}: ${escapeHtml(label)}</span>
    <span class="step-duration" title="Execution duration of this step: ${fmtSeconds(step.stepDurationMs)}s">(${fmtSeconds(step.stepDurationMs)}s)</span>
  `;

  if (hasErrors) {
    const errBox = document.createElement("div");
    errBox.className = "step-errors";
    errBox.textContent = step.errors
      .map((e) => `\u26A0 ${e.error}${e.impact ? ` — ${e.impact}` : ""}`)
      .join("  ");
    const wrap = document.createElement("div");
    wrap.appendChild(row);
    wrap.appendChild(errBox);
    return wrap;
  }
  return row;
}

function renderActiveStepRow(step, startMs) {
  const row = document.createElement("div");
  row.className = "step-row";
  const label = (META && META.node_labels && META.node_labels[step.node]) || step.node;
  const desc = (META && META.node_descriptions && META.node_descriptions[step.node]) || "Processing…";
  row.innerHTML = `
    <span class="step-elapsed" data-tick-since="${startMs}">[0.0s]</span>
    <span class="step-icon active">\u25CF</span>
    <span class="step-label ${nodeColorClass(step.node)}">Step ${step.stepNumber}: ${escapeHtml(label)}</span>
    <span class="step-desc">${escapeHtml(desc)}</span>
  `;
  return row;
}

function renderInitializingRow(startMs) {
  const row = document.createElement("div");
  row.className = "step-row";
  row.innerHTML = `
    <span class="step-elapsed" data-tick-since="${startMs}">[0.0s]</span>
    <span class="step-icon active">\u25CF</span>
    <span class="step-desc">Initializing agent graph…</span>
  `;
  return row;
}

function workerDetail(w) {
  if (w.status !== "done" || !w.update) {
    return (META && META.node_descriptions && META.node_descriptions[w.node]) || "Processing…";
  }
  const ctx = w.update.context_gathered || {};
  let totalChunks = 0;
  for (const v of Object.values(ctx)) {
    if (Array.isArray(v)) totalChunks += v.length;
  }
  const cites = (w.update.citations || []).length;
  if (totalChunks > 0) return `completed — ${totalChunks} chunks retrieved, ${cites} citations`;
  if (cites > 0) return `completed — ${cites} citations`;
  return "completed";
}

// ---------------------------------------------------------------------
// Sidebar Management
// ---------------------------------------------------------------------

function formatTime(isoString) {
  if (!isoString) return "Just now";
  try {
    const d = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - d) / 1000);
    if (diffSec < 60) return "Just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch (e) {
    return "";
  }
}

function renderSidebarSession(session, prepend = false) {
  const list = document.getElementById("history-list");
  if (!list) return;

  const existing = document.getElementById(`session-item-${session.id}`);
  const status = session.status || "complete";

  let statusHtml = "";
  if (status === "in_progress") {
    statusHtml = `
      <span class="session-loading-spinner" title="Running" aria-label="Running">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <circle cx="12" cy="12" r="9" stroke-opacity="0.25"></circle>
          <path d="M12 3a9 9 0 0 1 9 9"></path>
        </svg>
      </span>
    `;
  }

  const timeStr = formatTime(session.timestamp);
  const turnCount = Math.max(
    (session.turns && session.turns.length) || 0,
    session.turn_count || 1
  );
  const metaLabel = turnCount > 1 ? `${timeStr} \u00B7 ${turnCount} turns` : timeStr;

  if (existing) {
    existing.className = `history-item ${session.id === currentSessionId ? "active" : ""}`;
    const top = existing.querySelector(".history-item-top");
    if (top) {
      top.innerHTML = `<span class="hq">${escapeHtml(session.query || "(no query)")}</span>${statusHtml}`;
    }
    const metaEl = existing.querySelector(".hmeta");
    if (metaEl) {
      metaEl.innerHTML = `<span>${escapeHtml(metaLabel)}</span>`;
    }
    const delBtn = existing.querySelector(".session-delete-btn");
    if (delBtn) {
      delBtn.hidden = status === "in_progress";
    }
    return;
  }

  const item = document.createElement("div");
  item.id = `session-item-${session.id}`;
  item.className = `history-item ${session.id === currentSessionId ? "active" : ""}`;
  item.innerHTML = `
    <button type="button" class="history-item-select">
      <div class="history-item-top">
        <span class="hq">${escapeHtml(session.query || "(no query)")}</span>
        ${statusHtml}
      </div>
      <div class="hmeta">
        <span>${escapeHtml(metaLabel)}</span>
      </div>
    </button>
    <button type="button" class="session-delete-btn" title="Delete session" aria-label="Delete session" ${status === "in_progress" ? "hidden" : ""}>
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
        <path d="M10 11v6"></path>
        <path d="M14 11v6"></path>
        <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"></path>
      </svg>
    </button>
  `;
  const selectBtn = item.querySelector(".history-item-select");
  if (selectBtn) {
    selectBtn.addEventListener("click", () => switchToSession(session.id));
  }
  const deleteBtn = item.querySelector(".session-delete-btn");
  if (deleteBtn) {
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      requestDeleteSession(session.id);
    });
  }

  if (prepend && list.firstChild) {
    list.insertBefore(item, list.firstChild);
  } else {
    list.appendChild(item);
  }
}

function updateActiveSidebarHighlight() {
  document.querySelectorAll(".history-item").forEach((item) => {
    if (currentSessionId && item.id === `session-item-${currentSessionId}`) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

function updateSessionHeaderActions() {
  const btn = document.getElementById("export-session-btn");
  if (!btn) return;
  const session = currentSessionId ? knownSessions.get(currentSessionId) : null;
  const hasCompletedTurn = Boolean(
    session &&
      Array.isArray(session.turns) &&
      session.turns.some(
        (t) => t.state && t.state.finalState && t.state.finalState.final_response
      )
  );
  btn.hidden = !hasCompletedTurn;
}

function exportCurrentSessionMarkdown() {
  const session = currentSessionId ? knownSessions.get(currentSessionId) : null;
  if (!session || !Array.isArray(session.turns)) return;

  const completedTurns = session.turns.filter(
    (t) => t.state && t.state.finalState && t.state.finalState.final_response
  );
  if (!completedTurns.length) return;

  let md = `# Chemical Process Safety Session Report\n\n`;
  if (session.timestamp) {
    md += `**Session Started:** ${new Date(session.timestamp).toLocaleString()}\n\n`;
  }
  md += `**Exported:** ${new Date().toLocaleString()}\n\n`;
  md += `**Turns:** ${completedTurns.length}\n\n`;

  completedTurns.forEach((turn, idx) => {
    const turnNum = idx + 1;
    const fs = turn.state.finalState;
    const text = fs.final_response || "";
    const citations = fs.citations || [];

    md += `---\n\n`;
    md += `## Turn ${turnNum}: ${turn.query}\n\n`;
    if (fs.query && fs.raw_query && fs.query !== fs.raw_query) {
      md += `> **Standalone Query Rewrite:** ${fs.query}\n\n`;
    }
    md += `${text}\n\n`;

    if (citations.length > 0) {
      md += `### References (Turn ${turnNum})\n\n`;
      const hasPage = citations.some(
        (c) =>
          c.page != null &&
          String(c.page).trim() !== "" &&
          String(c.page).trim().toLowerCase() !== "none" &&
          String(c.page).trim().toLowerCase() !== "null"
      );
      if (hasPage) {
        md += `| Collection | Source | Page |\n|---|---|---|\n`;
        for (const c of citations) {
          const pg =
            c.page != null &&
            String(c.page).trim() !== "" &&
            String(c.page).trim().toLowerCase() !== "none" &&
            String(c.page).trim().toLowerCase() !== "null"
              ? String(c.page).trim()
              : "N/A";
          md += `| ${c.collection || "N/A"} | ${c.source || "N/A"} | ${pg} |\n`;
        }
      } else {
        md += `| Collection | Source |\n|---|---|\n`;
        for (const c of citations) {
          md += `| ${c.collection || "N/A"} | ${c.source || "N/A"} |\n`;
        }
      }
      md += `\n`;
    }
  });

  const slug = (session.query || completedTurns[0].query || "session")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .slice(0, 32);
  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `process_safety_session_${slug}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function requestDeleteSession(sessionId) {
  const dialog = document.getElementById("delete-session-dialog");
  if (!dialog || typeof dialog.showModal !== "function") return;
  pendingDeleteSessionId = sessionId;
  const session = knownSessions.get(sessionId);
  const desc = document.getElementById("delete-dialog-desc");
  if (desc) {
    const label = session && session.query ? `"${session.query}"` : "this session";
    desc.textContent = `Permanently remove ${label} and all turns in this session from the run history?`;
  }
  dialog.showModal();
}

async function deleteConfirmedSession(sessionId) {
  let errDetail = "";
  try {
    const resp = await fetch(`/api/runs/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      if (resp.status === 405) {
        errDetail =
          "Server returned 405 Method Not Allowed — restart `make web` so the running server loads the new delete endpoint.";
      } else {
        errDetail = body.detail || `Could not delete session (HTTP ${resp.status}).`;
      }
    }
  } catch (e) {
    errDetail = "Could not reach the server to delete this session.";
  }

  if (errDetail) {
    const dialog = document.getElementById("delete-session-dialog");
    const desc = document.getElementById("delete-dialog-desc");
    if (desc) desc.textContent = errDetail;
    pendingDeleteSessionId = sessionId;
    if (dialog && typeof dialog.showModal === "function" && !dialog.open) {
      dialog.showModal();
    }
    return;
  }

  knownSessions.delete(sessionId);
  const el = document.getElementById(`session-item-${sessionId}`);
  if (el && el.parentNode) {
    el.parentNode.removeChild(el);
  }
  historyOffset = Math.max(0, historyOffset - 1);

  if (currentSessionId === sessionId) {
    startNewSession();
  } else {
    updateSessionHeaderActions();
  }
}

// ---------------------------------------------------------------------
// Multi-Round Query Submission & Live Streaming
// ---------------------------------------------------------------------

function setAsking(isAsking) {
  const askBtn = document.getElementById("ask-btn");
  const queryInput = document.getElementById("query-input");
  if (askBtn) askBtn.disabled = isAsking;
  if (queryInput) queryInput.disabled = isAsking;
}

function showWelcomeView() {
  const welcome = document.getElementById("welcome-view");
  const sessionView = document.getElementById("session-view");
  const turnsContainer = document.getElementById("turns-container");
  const exportBtn = document.getElementById("export-session-btn");

  if (welcome) welcome.hidden = false;
  if (sessionView) sessionView.hidden = true;
  if (turnsContainer) turnsContainer.innerHTML = "";
  if (exportBtn) exportBtn.hidden = true;
}

function showSessionView() {
  const welcome = document.getElementById("welcome-view");
  const sessionView = document.getElementById("session-view");

  if (welcome) welcome.hidden = true;
  if (sessionView) sessionView.hidden = false;
  updateSessionHeaderActions();
}

async function submitQuery(query) {
  query = (query || "").trim();
  if (!query) return;

  // Clear query textarea & reset height
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = "";
    queryInput.style.height = "auto";
  }

  // Ensure session container is visible
  showSessionView();
  setAsking(true);

  // If no active session, create one
  let session = currentSessionId ? knownSessions.get(currentSessionId) : null;
  const isNewSession = !session;
  const history = getSessionHistory(session);

  // Temporary turn placeholder while requesting run_id
  const tempRunId = `temp_${Date.now()}`;
  const turn = {
    run_id: tempRunId,
    query,
    startMs: Date.now(),
    state: { ...initialState(), phase: "initializing" },
    dom: null,
  };

  const dom = createTurnElements(turn);
  turn.dom = dom;
  document.getElementById("turns-container").appendChild(dom.root);
  renderTurn(turn);
  scrollToBottom();
  startTicking();

  // Call backend to start run
  let resp;
  try {
    resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history,
        session_id: currentSessionId || undefined,
      }),
    });
  } catch (e) {
    turn.state.phase = "error";
    turn.state.errorMessage = "Could not reach the server.";
    renderTurn(turn);
    setAsking(false);
    return;
  }

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    turn.state.phase = "error";
    turn.state.errorMessage = body.detail || `Request failed (${resp.status}).`;
    renderTurn(turn);
    setAsking(false);
    return;
  }

  const { run_id } = await resp.json();
  turn.run_id = run_id;
  turn.dom.root.id = `turn-${run_id}`;

  if (isNewSession) {
    currentSessionId = run_id;
    session = {
      id: run_id,
      query,
      timestamp: new Date().toISOString(),
      status: "in_progress",
      turns: [turn],
    };
    knownSessions.set(run_id, session);
    renderSidebarSession(session, true);
    updateActiveSidebarHighlight();
  } else {
    session.turns.push(turn);
    session.status = "in_progress";
    renderSidebarSession(session, false);
  }

  // Start streaming for this turn
  startStreamForTurn(turn, session);
}

function startStreamForTurn(turn, session) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }

  currentEventSource = new EventSource(`/api/stream/${turn.run_id}`);

  currentEventSource.onmessage = (msg) => {
    let event;
    try {
      event = JSON.parse(msg.data);
    } catch (e) {
      return;
    }

    if (event.event === "synthesis_token") {
      if (turn._resetOnNextToken) {
        turn.streamingText = "";
        turn._resetOnNextToken = false;
      }
      turn.streamingText = (turn.streamingText || "") + (event.token || "");
      if (typeof event.html === "string") {
        turn.streamingHtml = event.html;
      }
      if (turn.dom && (!turn.state || !turn.state.finalState)) {
        if (typeof turn.streamingHtml === "string") {
          turn.dom.answerPane.innerHTML = turn.streamingHtml;
        }
      }
      scrollToBottom();
      return;
    }

    if (
      event.event === "step_complete" &&
      (event.node === "SynthesizeNode" || event.node === "RevisionNode")
    ) {
      if (event.update && event.update.draft_response_html) {
        turn.streamingHtml = event.update.draft_response_html;
      }
      turn._resetOnNextToken = true;
    }

    turn.state = reduceEvent(turn.state, event);
    renderTurn(turn);
    scrollToBottom();

    if (event.event === "pipeline_complete") {
      const finalResp = event.state && event.state.final_response;
      if (turn.query && finalResp) {
        conversationHistory.push({ query: turn.query, response: finalResp });
      }
      session.status = "complete";
      renderSidebarSession(session);
      updateSessionHeaderActions();
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
      stopTicking();
      setAsking(false);
      const queryInput = document.getElementById("query-input");
      if (queryInput) queryInput.focus();
    } else if (event.event === "error") {
      session.status = "error";
      renderSidebarSession(session);
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
      stopTicking();
      setAsking(false);
      const queryInput = document.getElementById("query-input");
      if (queryInput) queryInput.focus();
    }
  };
}

// ---------------------------------------------------------------------
// Switching Sessions & History
// ---------------------------------------------------------------------

let historyOffset = 0;
const HISTORY_PAGE_SIZE = 20;

async function loadHistory(reset = true) {
  if (reset) {
    historyOffset = 0;
    const list = document.getElementById("history-list");
    if (list) list.innerHTML = "";
  }

  try {
    const resp = await fetch(`/api/runs?limit=${HISTORY_PAGE_SIZE}&offset=${historyOffset}`);
    if (!resp.ok) return;
    const data = await resp.json();

    for (const run of data.runs) {
      if (!knownSessions.has(run.run_id)) {
        knownSessions.set(run.run_id, {
          id: run.run_id,
          query: run.query,
          timestamp: run.timestamp,
          status: run.status || "complete",
          turn_count: run.turn_count || 1,
          turns: [],
        });
      } else {
        const existingSession = knownSessions.get(run.run_id);
        existingSession.turn_count = Math.max(
          existingSession.turn_count || 1,
          run.turn_count || 1
        );
      }
      renderSidebarSession(knownSessions.get(run.run_id), false);
    }

    historyOffset += data.runs.length;
    const loadMore = document.getElementById("load-more-btn");
    if (loadMore) {
      loadMore.hidden = historyOffset >= data.total;
    }
  } catch (e) {
    // best-effort history load
  }
}

async function switchToSession(sessionId) {
  if (currentSessionId === sessionId && document.getElementById("turns-container").children.length > 0) {
    return;
  }

  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  stopTicking();

  currentSessionId = sessionId;
  updateActiveSidebarHighlight();
  showSessionView();

  const container = document.getElementById("turns-container");
  container.innerHTML = "";

  let session = knownSessions.get(sessionId);
  if (!session) {
    session = { id: sessionId, query: "", timestamp: "", status: "complete", turns: [] };
    knownSessions.set(sessionId, session);
  }

  // Fetch run detail
  const resp = await fetch(`/api/runs/${sessionId}`);
  if (resp.ok) {
    const data = await resp.json();
    session.query = data.query || session.query;
    session.status = data.done ? (data.error ? "error" : "complete") : "in_progress";

    const rawTurns =
      Array.isArray(data.turns) && data.turns.length > 0
        ? data.turns
        : [data];

    if (session.turns.length < rawTurns.length || (session.turns.length <= 1 && data.done)) {
      session.turns = rawTurns.map((tData) => {
        const turn = {
          run_id: tData.run_id || sessionId,
          query: tData.query || "(no query)",
          startMs: tData.created_at ? tData.created_at * 1000 : Date.now(),
          state: initialState(),
          dom: null,
        };

        if (tData.final_state) {
          turn.state.phase = tData.error ? "error" : "complete";
          turn.state.finalState = tData.final_state;
          if (Array.isArray(tData.steps)) {
            turn.state.completedSteps = tData.steps.map((s, idx) => ({
              stepNumber: idx + 1,
              node: s.node_name,
              elapsedMs: (s.elapsed_s || 0) * 1000,
              stepDurationMs: ((s.step_duration_s ?? s.duration_s) || 0) * 1000,
              update: s.update || {},
              errors: s.errors || [],
            }));
          }
        } else if (tData.events) {
          for (const ev of tData.events) {
            if (ev.event === "synthesis_token") {
              turn.streamingText = (turn.streamingText || "") + (ev.token || "");
              if (ev.html) turn.streamingHtml = ev.html;
            } else {
              if (ev.event === "step_complete" && ev.update && ev.update.draft_response_html) {
                turn.streamingHtml = ev.update.draft_response_html;
              }
              turn.state = reduceEvent(turn.state, ev);
            }
          }
        }
        return turn;
      });
      if (session.turns.length > 0) {
        session.query = session.turns[0].query || session.query;
        session.turn_count = session.turns.length;
      }
    }
  }

  // Render all turns in this session
  conversationHistory = [];
  for (const t of session.turns) {
    const fs = t.state && t.state.finalState;
    if (fs && Array.isArray(fs.conversation_history) && conversationHistory.length === 0) {
      for (const h of fs.conversation_history) {
        if (h && h.query && h.response) {
          conversationHistory.push({ query: h.query, response: h.response });
        }
      }
    }
    const respText = fs && fs.final_response;
    if (t.query && respText) {
      conversationHistory.push({ query: t.query, response: respText });
    }
    t.dom = createTurnElements(t);
    // If completed or error, collapse pipeline card by default
    if (t.state.phase === "complete" || t.state.phase === "error" || t.state.finalState) {
      t.dom.pipelineCard.open = false;
    }
    container.appendChild(t.dom.root);
    renderTurn(t);
  }

  // If latest turn is still running, resume live stream
  const latestTurn = session.turns[session.turns.length - 1];
  if (latestTurn && latestTurn.state.phase === "running" && !latestTurn.state.finalState) {
    startTicking();
    setAsking(true);
    startStreamForTurn(latestTurn, session);
  } else {
    setAsking(false);
  }

  updateSessionHeaderActions();
  scrollToBottom();
}

function startNewSession() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  stopTicking();
  setAsking(false);

  currentSessionId = null;
  conversationHistory = [];
  updateActiveSidebarHighlight();
  showWelcomeView();

  const area = document.getElementById("chat-scroll-area");
  if (area) {
    area.scrollTop = 0;
  }

  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = "";
    queryInput.style.height = "auto";
    queryInput.focus();
  }
}

// ---------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------

async function init() {
  try {
    const metaResp = await fetch("/api/meta");
    META = await metaResp.json();
  } catch (e) {
    META = {
      node_labels: {},
      node_descriptions: {},
      agent_key_to_node: {},
      worker_nodes: [
        "OshaRegsWorker",
        "OshaEnfWorker",
        "IncidentWorker",
        "RecommendationsWorker",
        "RootCauseWorker",
        "DataAnalystWorker",
      ],
      validation_score_threshold: 0.7,
    };
  }

  const askBtn = document.getElementById("ask-btn");
  const queryInput = document.getElementById("query-input");
  const newQueryBtn = document.getElementById("new-query-btn");
  const loadMoreBtn = document.getElementById("load-more-btn");

  if (askBtn && queryInput) {
    askBtn.addEventListener("click", () => {
      submitQuery(queryInput.value);
    });

    queryInput.addEventListener("input", () => {
      queryInput.style.height = "auto";
      queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
    });

    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        if (e.isComposing || e.keyCode === 229) return;
        e.preventDefault();
        submitQuery(queryInput.value);
      }
    });
  }

  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q;
      if (q) {
        submitQuery(q);
      }
    });
  });

  if (newQueryBtn) {
    newQueryBtn.addEventListener("click", startNewSession);
  }

  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", () => loadHistory(false));
  }

  const exportSessionBtn = document.getElementById("export-session-btn");
  if (exportSessionBtn) {
    exportSessionBtn.addEventListener("click", exportCurrentSessionMarkdown);
  }

  const deleteDialog = document.getElementById("delete-session-dialog");
  if (deleteDialog) {
    deleteDialog.addEventListener("close", () => {
      const action = deleteDialog.returnValue;
      deleteDialog.returnValue = "";
      const targetId = pendingDeleteSessionId;
      pendingDeleteSessionId = null;
      if (action === "delete" && targetId) {
        deleteConfirmedSession(targetId);
      }
    });
  }

  if (typeof HTMLDialogElement !== "undefined" && !("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dialog) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dialog.addEventListener("pointerdown", (e) => {
        pressedOutside = outside(e, dialog);
      });
      dialog.addEventListener("pointerup", (e) => {
        if (pressedOutside && outside(e, dialog)) dialog.close();
      });
    }
  }

  showWelcomeView();
  await loadHistory();
}

document.addEventListener("DOMContentLoaded", init);
