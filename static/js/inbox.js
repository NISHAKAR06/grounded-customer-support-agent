/**
 * Inbox workspace interactions
 */

document.addEventListener('DOMContentLoaded', () => {
  const rows = document.querySelectorAll('.inbox-row');
  rows.forEach(row => {
    row.addEventListener('click', () => {
      const ticketId = row.getAttribute('data-ticket-id');
      const sampleText = row.getAttribute('data-message');
      if (ticketId && sampleText) {
        // Option to test this row in Simulate
        if (confirm(`Open "${ticketId}" in the Simulate workspace?`)) {
          window.location.href = `/simulate?msg=${encodeURIComponent(sampleText)}`;
        }
      }
    });
  });
});
