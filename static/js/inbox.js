/**
 * Inbox workspace interactions
 * Only explicit clicks on the "Simulate ↗" button navigate to /simulate.
 * Clicking elsewhere on the row does not trigger navigation.
 */

document.addEventListener('DOMContentLoaded', () => {
  const rows = document.querySelectorAll('.inbox-row');
  rows.forEach(row => {
    row.addEventListener('click', (e) => {
      // If clicking the Simulate button or an anchor, let native navigation handle it
      if (e.target.closest('a')) return;

      // Highlight selected row without navigating away
      rows.forEach(r => r.classList.remove('selected'));
      row.classList.add('selected');
    });
  });
});
