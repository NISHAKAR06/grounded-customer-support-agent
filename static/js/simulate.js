/**
 * Simulate Incoming Message Workspace Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const messageInput = document.getElementById('customer-message-input');
  const runBtn = document.getElementById('run-agent-btn');
  const timelineContainer = document.getElementById('agent-timeline');
  const resultContainer = document.getElementById('case-result-container');

  // Example message loaders
  document.querySelectorAll('.example-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleText = btn.getAttribute('data-sample');
      if (sampleText && messageInput) {
        messageInput.value = sampleText;
      }
    });
  });

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      const text = messageInput.value.trim();
      if (!text) {
        alert('Please enter a customer message or select an example.');
        return;
      }

      // Reset UI state
      runBtn.disabled = true;
      runBtn.innerText = 'Running AI Agent...';
      if (resultContainer) resultContainer.style.display = 'none';

      // Update timeline to processing state
      updateTimelineStep('step-received', 'active', 'Analyzing text payload...');

      try {
        const startTime = performance.now();
        const response = await fetch('/api/agent/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ customer_message: text }),
        });

        if (!response.ok) {
          throw new Error(`Server returned status ${response.status}`);
        }

        const data = await response.json();
        const totalDuration = Math.round(performance.now() - startTime);

        // Update Timeline to Completed States
        updateTimelineStep('step-received', 'completed', `Received (${data.customer_message.length} chars)`);
        updateTimelineStep('step-intent', 'completed', `${data.intent.name} (${Math.round(data.intent.confidence * 100)}% conf)`);
        updateTimelineStep('step-retrieval', 'completed', `${data.retrieval.evidence.length} resolved cases retrieved`);
        updateTimelineStep('step-generation', 'completed', `Grounded draft generated via ${data.generation.provider}`);
        updateTimelineStep('step-validation', 'completed', data.validation.all_passed ? 'All safety checks passed' : 'Validation warnings');
        updateTimelineStep('step-decision', 'completed', `${data.routing.decision} (${totalDuration}ms total)`);

        // Render Case Results
        renderCaseResults(data);
      } catch (err) {
        console.error('Agent execution error:', err);
        alert('An error occurred while executing the AI agent. Check console or backend logs.');
      } finally {
        runBtn.disabled = false;
        runBtn.innerText = 'Run AI Agent';
      }
    });
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
