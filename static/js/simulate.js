/**
 * Simulate Incoming Message Workspace Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const messageInput = document.getElementById('customer-message-input');
  const runBtn = document.getElementById('run-agent-btn');
  const timelineContainer = document.getElementById('agent-timeline');
  const resultContainer = document.getElementById('case-result-container');

  const providerSelect = document.getElementById('llm-provider-select');
  const providerBadge = document.getElementById('provider-status-badge');

  // Load available LLM providers status from backend
  let providerCatalog = [];
  async function loadProviders() {
    try {
      const resp = await fetch('/api/agent/providers');
      if (resp.ok) {
        providerCatalog = await resp.json();
        updateProviderBadge();
      }
    } catch (e) {
      console.warn('Could not fetch provider list:', e);
    }
  }

  function updateProviderBadge() {
    if (!providerSelect || !providerBadge) return;
    const selectedId = providerSelect.value;
    const meta = providerCatalog.find(p => p.id === selectedId);
    if (meta) {
      if (meta.configured) {
        providerBadge.className = 'badge badge-auto';
        providerBadge.innerText = 'Ready';
      } else {
        providerBadge.className = 'badge badge-human';
        providerBadge.innerText = 'Key Missing (Fallback Active)';
      }
    } else {
      providerBadge.className = 'badge badge-auto';
      providerBadge.innerText = 'Ready';
    }
  }

  if (providerSelect) {
    providerSelect.addEventListener('change', updateProviderBadge);
    loadProviders();
  }

  // Check URL query parameters for preloaded message (e.g. from Inbox)
  const urlParams = new URLSearchParams(window.location.search);
  const preloadedMsg = urlParams.get('msg');
  if (preloadedMsg && messageInput) {
    messageInput.value = preloadedMsg;
    messageInput.focus();
  }
  const preloadedCustomerHandle = urlParams.get('handle') || urlParams.get('user') || null;

  function resetTimeline() {
    const steps = [
      { id: 'step-received', num: '1', title: 'Message Received & Context Assembled', desc: 'Ready for pipeline execution' },
      { id: 'step-intent', num: '2', title: 'Intent Classification', desc: 'Awaiting execution' },
      { id: 'step-retrieval', num: '3', title: 'Historical Evidence Retrieval', desc: 'Awaiting execution' },
      { id: 'step-generation', num: '4', title: 'Grounded Reply Generation', desc: 'Awaiting execution' },
      { id: 'step-validation', num: '5', title: 'Response Validation', desc: 'Awaiting execution' },
      { id: 'step-decision', num: '6', title: 'Escalation / Automation Policy', desc: 'Awaiting execution' },
    ];
    steps.forEach(s => {
      const el = document.getElementById(s.id);
      if (!el) return;
      el.className = 'timeline-step';
      const icon = el.querySelector('.step-icon');
      if (icon) icon.innerText = s.num;
      const desc = el.querySelector('.step-desc');
      if (desc) desc.innerText = s.desc;
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', () => {
      const text = messageInput.value.trim();
      if (!text) {
        alert('Please enter a customer message to analyze.');
        return;
      }

      const selectedProvider = providerSelect ? providerSelect.value : 'groq';

      // Reset UI state
      runBtn.disabled = true;
      runBtn.innerText = 'Running AI Agent...';
      if (resultContainer) resultContainer.style.display = 'none';

      resetTimeline();

      // Step 1: Active
      updateTimelineStep('step-received', 'active', 'Assembling message payload...');
      const startTime = performance.now();

      // Attempt real-time SSE streaming for live pipeline updates
      let sseActive = false;
      let completed = false;

      let sseUrl = `/api/agent/stream?customer_message=${encodeURIComponent(text)}&provider=${encodeURIComponent(selectedProvider)}`;
      if (preloadedCustomerHandle) {
        sseUrl += `&customer_handle=${encodeURIComponent(preloadedCustomerHandle)}`;
      }
      let evtSource = null;

      try {
        evtSource = new EventSource(sseUrl);

        evtSource.addEventListener('MESSAGE_RECEIVED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          updateTimelineStep('step-received', 'completed', `Received (${data.customer_message.length} chars)`);
          updateTimelineStep('step-intent', 'active', 'Classifying inquiry intent & domain signals...');
        });

        evtSource.addEventListener('INTENT_CLASSIFICATION_STARTED', () => {
          sseActive = true;
          updateTimelineStep('step-intent', 'active', 'Extracting domain entities & signals...');
        });

        evtSource.addEventListener('INTENT_CLASSIFICATION_COMPLETED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          updateTimelineStep('step-intent', 'completed', `${data.intent} (${Math.round(data.confidence * 100)}% conf, ${data.elapsed_ms}ms)`);
          updateTimelineStep('step-retrieval', 'active', 'Querying dense FAISS vector index...');
        });

        evtSource.addEventListener('RETRIEVAL_STARTED', () => {
          sseActive = true;
          updateTimelineStep('step-retrieval', 'active', 'Scanning 2,245 historical resolved cases...');
        });

        evtSource.addEventListener('RETRIEVAL_COMPLETED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          updateTimelineStep('step-retrieval', 'completed', `${data.count} resolved cases retrieved (sim: ${data.top_similarity.toFixed(2)}, ${data.elapsed_ms}ms)`);
          updateTimelineStep('step-generation', 'active', `Drafting grounded reply via ${selectedProvider}...`);
        });

        evtSource.addEventListener('GENERATION_STARTED', () => {
          sseActive = true;
          updateTimelineStep('step-generation', 'active', `Generating response with grounding constraints...`);
        });

        evtSource.addEventListener('GENERATION_COMPLETED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          updateTimelineStep('step-generation', 'completed', `Grounded draft generated via ${data.provider} (${data.elapsed_ms}ms)`);
          updateTimelineStep('step-validation', 'active', 'Auditing 6 safety barriers & URL whitelists...');
        });

        evtSource.addEventListener('VALIDATION_STARTED', () => {
          sseActive = true;
          updateTimelineStep('step-validation', 'active', 'Running deterministic safety & PII checks...');
        });

        evtSource.addEventListener('VALIDATION_COMPLETED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          const valText = data.all_passed
            ? `All safety checks passed (overlap: ${(data.grounding_score * 100).toFixed(0)}%)`
            : `Validation warnings flagged (overlap: ${(data.grounding_score * 100).toFixed(0)}%)`;
          updateTimelineStep('step-validation', 'completed', valText);
          updateTimelineStep('step-decision', 'active', 'Evaluating escalation threshold policies...');
        });

        evtSource.addEventListener('ESCALATION_STARTED', () => {
          sseActive = true;
          updateTimelineStep('step-decision', 'active', 'Applying deterministic escalation criteria...');
        });

        evtSource.addEventListener('ESCALATION_COMPLETED', (e) => {
          sseActive = true;
          const data = JSON.parse(e.data);
          const totalDuration = Math.round(performance.now() - startTime);
          updateTimelineStep('step-decision', 'completed', `${data.decision} (${totalDuration}ms total)`);
        });

        evtSource.addEventListener('RESULT', (e) => {
          completed = true;
          const resultData = JSON.parse(e.data);
          evtSource.close();
          runBtn.disabled = false;
          runBtn.innerText = 'Run AI Agent';
          renderCaseResults(resultData);
        });

        evtSource.addEventListener('ERROR', (e) => {
          console.warn('SSE stream error event received:', e);
          evtSource.close();
          if (!completed) {
            fallbackSyncRun(text, selectedProvider, startTime);
          }
        });

        evtSource.onerror = (e) => {
          if (!sseActive && !completed) {
            // If SSE couldn't connect at all, fallback immediately to synchronous POST
            evtSource.close();
            fallbackSyncRun(text, selectedProvider, startTime);
          } else if (!completed) {
            evtSource.close();
            runBtn.disabled = false;
            runBtn.innerText = 'Run AI Agent';
          }
        };
      } catch (sseErr) {
        console.warn('EventSource failed to initialize, falling back to POST:', sseErr);
        fallbackSyncRun(text, selectedProvider, startTime);
      }
    });
  }

  // Fallback synchronous POST runner if SSE is blocked
  async function fallbackSyncRun(text, selectedProvider, startTime) {
    try {
      updateTimelineStep('step-received', 'active', 'Analyzing text payload...');
      const payload = {
        customer_message: text,
        provider: selectedProvider,
      };
      if (preloadedCustomerHandle) {
        payload.customer_handle = preloadedCustomerHandle;
      }
      const response = await fetch('/api/agent/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      const totalDuration = Math.round(performance.now() - startTime);

      updateTimelineStep('step-received', 'completed', `Received (${data.customer_message.length} chars)`);
      updateTimelineStep('step-intent', 'completed', `${data.intent.name} (${Math.round(data.intent.confidence * 100)}% conf)`);
      updateTimelineStep('step-retrieval', 'completed', `${data.retrieval.evidence.length} resolved cases retrieved`);
      updateTimelineStep('step-generation', 'completed', `Grounded draft generated via ${data.generation.provider}`);
      updateTimelineStep('step-validation', 'completed', data.validation.all_passed ? 'All safety checks passed' : 'Validation warnings');
      updateTimelineStep('step-decision', 'completed', `${data.routing.decision} (${totalDuration}ms total)`);

      renderCaseResults(data);
    } catch (err) {
      console.error('Agent execution error:', err);
      alert('An error occurred while executing the AI agent. Check console or backend logs.');
    } finally {
      runBtn.disabled = false;
      runBtn.innerText = 'Run AI Agent';
    }
  }
});

function updateTimelineStep(elementId, state, detailText) {
  const el = document.getElementById(elementId);
  if (!el) return;

  el.className = `timeline-step ${state}`;
  const descEl = el.querySelector('.step-desc');
  if (descEl && detailText) {
    descEl.innerText = detailText;
  }

  const iconEl = el.querySelector('.step-icon');
  if (iconEl) {
    if (state === 'completed') {
      iconEl.innerText = '✓';
    } else if (state === 'active') {
      iconEl.innerHTML = '<span class="pulse-dot">●</span>';
    } else {
      const stepNums = {
        'step-received': '1',
        'step-intent': '2',
        'step-retrieval': '3',
        'step-generation': '4',
        'step-validation': '5',
        'step-decision': '6',
      };
      iconEl.innerText = stepNums[elementId] || '•';
    }
  }
}

function renderCaseResults(data) {
  const container = document.getElementById('case-result-container');
  if (!container) return;

  // 1. AI Analysis
  const intentEl = document.getElementById('res-intent-name');
  const confEl = document.getElementById('res-intent-conf');
  const signalsEl = document.getElementById('res-intent-signals');

  if (intentEl) intentEl.innerText = data.intent.name;
  if (confEl) confEl.innerText = `${Math.round(data.intent.confidence * 100)}%`;
  if (signalsEl) {
    signalsEl.innerHTML = (data.intent.signals || [])
      .map(s => `<span class="chip-btn" style="cursor:default;">✓ ${escapeHtml(s)}</span>`)
      .join(' ');
  }

  // 2. Historical Evidence
  const evidenceListEl = document.getElementById('res-evidence-list');
  if (evidenceListEl) {
    if (!data.retrieval.evidence || data.retrieval.evidence.length === 0) {
      evidenceListEl.innerHTML = '<p class="step-desc">No historical resolved cases matched.</p>';
    } else {
      evidenceListEl.innerHTML = data.retrieval.evidence.map((c, idx) => `
        <div class="evidence-card">
          <div class="evidence-header">
            <span>CASE #${escapeHtml(c.case_id)}</span>
            <span class="badge badge-neutral">Similarity: ${c.similarity.toFixed(2)}</span>
          </div>
          <div class="evidence-body">
            <div><strong>Customer:</strong> "${escapeHtml(c.customer_text)}"</div>
            <div class="evidence-brand-reply"><strong>Brand Resolution:</strong> "${escapeHtml(c.brand_response)}"</div>
          </div>
        </div>
      `).join('');
    }
  }

  // 3. Grounded Reply
  const replyEl = document.getElementById('res-grounded-reply');
  const providerEl = document.getElementById('res-provider-badge');
  if (replyEl) replyEl.innerText = data.generation.draft_reply;
  if (providerEl) providerEl.innerText = `Generated via ${data.generation.provider}`;

  // 4. Decision Banner
  const decisionBanner = document.getElementById('res-decision-banner');
  const decisionTitle = document.getElementById('res-decision-title');
  const decisionReasons = document.getElementById('res-decision-reasons');

  if (decisionBanner && decisionTitle && decisionReasons) {
    const isAuto = data.routing.decision === 'AUTO_HANDLE';
    decisionBanner.className = `decision-banner ${isAuto ? 'auto' : 'escalate'}`;
    decisionTitle.innerHTML = isAuto
      ? `<span class="badge badge-auto">🟢 AUTO-HANDLE</span> <strong>Automated Dispatch Approved</strong>`
      : `<span class="badge badge-human">🔴 HUMAN REVIEW</span> <strong>Escalation Required</strong>`;

    decisionReasons.innerHTML = data.routing.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('');
  }

  container.style.display = 'block';
  container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}
