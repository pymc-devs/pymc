// Mobile menu fix: make touch interactions on mobile (iOS/Safari) reliably trigger
// the mobile navigation toggle. Minimal, defensive enhancement.
document.addEventListener('DOMContentLoaded', function () {
  try {
    const selectors = [
      '.primary-toggle',
      '.secondary-toggle',
      '[data-toggle="collapse"]',
      '[aria-controls][aria-expanded]'
    ].join(',');
    const controls = Array.from(document.querySelectorAll(selectors));
    controls.forEach(el => {
      if (el.__pymc_touchbound) return;
      el.__pymc_touchbound = true;
      el.addEventListener('touchstart', function (e) {
        // iOS sometimes doesn't synthesize click events for transformed
        // elements or elements in certain stacking contexts. Best-effort
        // bridge: prevent default then dispatch a MouseEvent click.
        try {
          e.preventDefault();
          this.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        } catch (err) {
          // ignore errors
        }
      }, { passive: false });
    });
  } catch (err) {
    // swallow errors to avoid breaking docs
  }
});
