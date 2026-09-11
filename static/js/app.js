/**
 * Grounded Customer Support Agent - Global Application Script
 */

document.addEventListener('DOMContentLoaded', () => {
  // Check backend health asynchronously
  checkBackendHealth();

  // Initialize responsive mobile navigation drawer
  initMobileNavigation();
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

function initMobileNavigation() {
  const toggleBtn = document.getElementById('mobileNavToggle');
  const sidebar = document.querySelector('.app-sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');

  if (!toggleBtn || !sidebar || !backdrop) return;

  function openSidebar() {
    sidebar.classList.add('open');
    backdrop.classList.add('active');
    toggleBtn.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    backdrop.classList.remove('active');
    toggleBtn.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (sidebar.classList.contains('open')) {
      closeSidebar();
    } else {
      openSidebar();
    }
  });

  backdrop.addEventListener('click', closeSidebar);

  // Close sidebar on link click on smaller viewports
  const navLinks = sidebar.querySelectorAll('.nav-link');
  navLinks.forEach((link) => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 768) {
        closeSidebar();
      }
    });
  });

  // Close on Escape key press
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) {
      closeSidebar();
    }
  });

  // Handle resize from mobile to desktop
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768 && sidebar.classList.contains('open')) {
      closeSidebar();
    }
  });
}
