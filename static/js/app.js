/**
 * Grounded Customer Support Agent - Global Application Script
 */

document.addEventListener('DOMContentLoaded', () => {
  // Check backend health asynchronously
  checkBackendHealth();
});

async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      const statusEl = document.getElementById('system-status-indicator');
      if (statusEl) {
        statusEl.innerHTML = `<span class="status-dot-pulse"></span> ${data.status === 'healthy' ? 'Online' : 'Degraded'}`;
      }
    }
  } catch (err) {
    console.warn('Health check warning:', err);
  }
}
