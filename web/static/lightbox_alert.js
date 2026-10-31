/**
 * UC Health / SupplyCopia - Universal Clean Lightbox Popup System
 * Automatically intercepts and replaces native window.alert(), window.confirm(),
 * and window.prompt() with a modern, clinical-grade lightbox modal.
 */
(function(window) {
  'use strict';

  let dialogEl = null;
  let activePromiseResolve = null;
  const alertQueue = [];
  let isDisplaying = false;

  function createDialogDOM() {
    if (document.getElementById('cqo-lightbox-dialog')) {
      return document.getElementById('cqo-lightbox-dialog');
    }

    const div = document.createElement('div');
    div.id = 'cqo-lightbox-dialog';
    div.className = 'cqo-lightbox-overlay';
    div.setAttribute('role', 'dialog');
    div.setAttribute('aria-modal', 'true');
    div.setAttribute('aria-labelledby', 'cqo-lightbox-title');
    div.innerHTML = `
      <div class="cqo-lightbox-card">
        <div class="cqo-lightbox-header">
          <div class="cqo-lightbox-badge info" id="cqo-lightbox-badge">
            <span id="cqo-lightbox-icon">ℹ️</span>
          </div>
          <div class="cqo-lightbox-heading">
            <h3 id="cqo-lightbox-title">Clinical Quality Optimization</h3>
            <p id="cqo-lightbox-subtitle">UCHealth • SupplyCopia Clinical Studio</p>
          </div>
          <button type="button" class="cqo-lightbox-close" id="cqo-lightbox-close-btn" aria-label="Close dialog">&times;</button>
        </div>
        <div class="cqo-lightbox-body" id="cqo-lightbox-body">
          <div id="cqo-lightbox-message"></div>
          <input type="text" id="cqo-lightbox-input" class="cqo-lightbox-input" style="display: none;" autocomplete="off" />
        </div>
        <div class="cqo-lightbox-footer" id="cqo-lightbox-footer">
          <button type="button" class="cqo-lightbox-btn cqo-lightbox-btn-cancel" id="cqo-lightbox-cancel-btn" style="display: none;">Cancel</button>
          <button type="button" class="cqo-lightbox-btn cqo-lightbox-btn-primary" id="cqo-lightbox-primary-btn">OK</button>
        </div>
      </div>
    `;

    // Ensure document.body exists
    if (document.body) {
      document.body.appendChild(div);
    } else {
      document.addEventListener('DOMContentLoaded', () => {
        if (!document.getElementById('cqo-lightbox-dialog')) {
          document.body.appendChild(div);
        }
      });
    }

    // Bind event handlers
    const closeBtn = div.querySelector('#cqo-lightbox-close-btn');
    const cancelBtn = div.querySelector('#cqo-lightbox-cancel-btn');
    const primaryBtn = div.querySelector('#cqo-lightbox-primary-btn');
    const inputEl = div.querySelector('#cqo-lightbox-input');

    function closeWith(val) {
      if (!isDisplaying) return;
      div.classList.remove('active');
      document.body.style.removeProperty('overflow');
      
      setTimeout(() => {
        div.style.display = 'none';
        isDisplaying = false;
        if (activePromiseResolve) {
          const resolve = activePromiseResolve;
          activePromiseResolve = null;
          resolve(val);
        }
        processQueue();
      }, 200);
    }

    if (closeBtn) closeBtn.onclick = () => closeWith(div.dataset.mode === 'confirm' ? false : (div.dataset.mode === 'prompt' ? null : true));
    if (cancelBtn) cancelBtn.onclick = () => closeWith(div.dataset.mode === 'confirm' ? false : (div.dataset.mode === 'prompt' ? null : true));

    if (primaryBtn) {
      primaryBtn.onclick = () => {
        if (div.dataset.mode === 'prompt') {
          closeWith(inputEl ? inputEl.value : '');
        } else if (div.dataset.mode === 'confirm') {
          closeWith(true);
        } else {
          closeWith(true);
        }
      };
    }

    div.onclick = (e) => {
      if (e.target === div) {
        closeWith(div.dataset.mode === 'confirm' ? false : (div.dataset.mode === 'prompt' ? null : true));
      }
    };

    document.addEventListener('keydown', (e) => {
      if (isDisplaying && div.classList.contains('active')) {
        if (e.key === 'Escape') {
          e.preventDefault();
          closeWith(div.dataset.mode === 'confirm' ? false : (div.dataset.mode === 'prompt' ? null : true));
        } else if (e.key === 'Enter') {
          const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
          if (activeTag !== 'textarea') {
            e.preventDefault();
            if (div.dataset.mode === 'prompt') {
              closeWith(inputEl ? inputEl.value : '');
            } else if (div.dataset.mode === 'confirm') {
              closeWith(true);
            } else {
              closeWith(true);
            }
          }
        }
      }
    });

    return div;
  }

  function processQueue() {
    if (isDisplaying || alertQueue.length === 0) return;
    const task = alertQueue.shift();
    task();
  }

  function detectType(message) {
    const s = String(message || '').toLowerCase();
    if (s.includes('error') || s.includes('fail') || s.includes('exception') || s.includes('denied') || s.includes('could not')) {
      return 'error';
    }
    if (s.includes('warn') || s.includes('please select') || s.includes('please allow') || s.includes('required') || s.includes('notice') || s.includes('attention')) {
      return 'warning';
    }
    if (s.includes('success') || s.includes('saved') || s.includes('complete') || s.includes('purged') || s.includes('updated') || s.includes('flagged')) {
      return 'success';
    }
    return 'info';
  }

  function renderDialog(config) {
    return new Promise((resolve) => {
      const task = () => {
        isDisplaying = true;
        activePromiseResolve = resolve;
        if (!dialogEl) dialogEl = createDialogDOM();

        const badge = document.getElementById('cqo-lightbox-badge');
        const icon = document.getElementById('cqo-lightbox-icon');
        const titleEl = document.getElementById('cqo-lightbox-title');
        const msgEl = document.getElementById('cqo-lightbox-message');
        const inputEl = document.getElementById('cqo-lightbox-input');
        const cancelBtn = document.getElementById('cqo-lightbox-cancel-btn');
        const primaryBtn = document.getElementById('cqo-lightbox-primary-btn');

        dialogEl.dataset.mode = config.mode || 'alert';

        const type = config.type || detectType(config.message);
        const iconMap = {
          info: 'ℹ️',
          warning: '⚠️',
          error: '⛔',
          success: '✅'
        };

        if (badge) {
          badge.className = `cqo-lightbox-badge ${type}`;
        }
        if (icon) {
          icon.textContent = iconMap[type] || 'ℹ️';
        }
        if (titleEl) {
          titleEl.textContent = config.title || (
            type === 'error' ? 'System Error' :
            type === 'warning' ? 'Clinical Action Required' :
            type === 'success' ? 'Action Completed' :
            'Clinical Quality Alert'
          );
        }
        if (msgEl) {
          msgEl.innerHTML = String(config.message || '').replace(/\n/g, '<br>');
        }

        if (config.mode === 'prompt') {
          if (inputEl) {
            inputEl.style.display = 'block';
            inputEl.value = config.defaultValue || '';
            inputEl.placeholder = config.placeholder || 'Enter rationale or details...';
          }
          if (cancelBtn) {
            cancelBtn.style.display = 'inline-block';
            cancelBtn.textContent = config.cancelText || 'Cancel';
          }
          if (primaryBtn) {
            primaryBtn.textContent = config.confirmText || 'Submit';
            primaryBtn.className = 'cqo-lightbox-btn cqo-lightbox-btn-primary';
          }
        } else if (config.mode === 'confirm') {
          if (inputEl) inputEl.style.display = 'none';
          if (cancelBtn) {
            cancelBtn.style.display = 'inline-block';
            cancelBtn.textContent = config.cancelText || 'Cancel';
          }
          if (primaryBtn) {
            primaryBtn.textContent = config.confirmText || 'Confirm';
            primaryBtn.className = config.danger ? 'cqo-lightbox-btn cqo-lightbox-btn-danger' : 'cqo-lightbox-btn cqo-lightbox-btn-primary';
          }
        } else {
          // Alert mode
          if (inputEl) inputEl.style.display = 'none';
          if (cancelBtn) cancelBtn.style.display = 'none';
          if (primaryBtn) {
            primaryBtn.textContent = config.confirmText || 'OK';
            primaryBtn.className = 'cqo-lightbox-btn cqo-lightbox-btn-primary';
          }
        }

        dialogEl.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        requestAnimationFrame(() => {
          dialogEl.classList.add('active');
          if (config.mode === 'prompt' && inputEl) {
            inputEl.focus();
            inputEl.select();
          } else if (primaryBtn) {
            primaryBtn.focus();
          }
        });
      };

      alertQueue.push(task);
      processQueue();
    });
  }

  function showLightboxAlert(message, title, type) {
    return renderDialog({
      mode: 'alert',
      message: message,
      title: title,
      type: type
    });
  }

  function showLightboxConfirm(message, title, options = {}) {
    return renderDialog({
      mode: 'confirm',
      message: message,
      title: title,
      type: options.type || (options.danger ? 'warning' : 'info'),
      confirmText: options.confirmText,
      cancelText: options.cancelText,
      danger: options.danger
    });
  }

  function showLightboxPrompt(message, defaultValue = '', title = '', options = {}) {
    return renderDialog({
      mode: 'prompt',
      message: message,
      defaultValue: defaultValue,
      title: title || 'Clinical Rationale Required',
      type: 'info',
      confirmText: options.confirmText || 'Submit',
      cancelText: options.cancelText || 'Cancel',
      placeholder: options.placeholder
    });
  }

  // Globally replace native alert
  window.alert = function(msg, title, type) {
    return showLightboxAlert(msg, title, type);
  };

  // Expose standard API
  window.showLightboxAlert = showLightboxAlert;
  window.showLightboxConfirm = showLightboxConfirm;
  window.showLightboxPrompt = showLightboxPrompt;

  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      dialogEl = createDialogDOM();
    });
  } else {
    dialogEl = createDialogDOM();
  }
})(window);
