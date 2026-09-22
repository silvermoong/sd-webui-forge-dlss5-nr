(() => {
    const watched = new WeakSet();
    let revision = 0;

    function paint(root, toolbar) {
        const prefix = toolbar.id.replace(/_preset_toolbar$/, '');
        const name = toolbar.querySelector('input[role="listbox"]')?.value.trim() || '';
        const state = root.querySelector(`#${prefix}_preset_state span`);
        if (!state) return;
        let records;
        try { records = JSON.parse(root.querySelector(`#${prefix}_preset_catalog textarea`)?.value || '{}'); }
        catch { return; }
        const selected = Object.hasOwn(records, name) ? records[name] : null;
        const matches = selected?.every((record, index) => {
            const pagePrefix = prefix + '_' + (index ? `pass_${index + 1}_` : '');
            const value = key => root.querySelector(`#${pagePrefix}${key} input`);
            const stages = [...root.querySelectorAll(`#${pagePrefix}stage input[type="radio"]`)];
            const stage = stages.length > 1 && stages[1].checked ? 'after_hr' : 'before_hr';
            return record.enabled === root.querySelector(`#${prefix}_pass_${index + 1}_enabled input`)?.checked &&
                record.stage === stage && Object.entries(record.params).every(([key, expected]) => {
                    const input = root.querySelector(`#${pagePrefix}${key} input[type="number"]`) || value(key);
                    if (!input) return false;
                    return expected === (input.type === 'checkbox' ? input.checked : input.value === '' ? null : Number(input.value));
                });
        });
        const text = selected ? (matches ? state.dataset.saved : state.dataset.modified) : name ? state.dataset.new : '';
        if (state.textContent !== text) state.textContent = text;
    }

    function install() {
        const root = (typeof gradioApp === 'function' && gradioApp()) || document;
        if (!watched.has(root)) {
            watched.add(root);
            new MutationObserver(install).observe(root, {childList: true, subtree: true});
        }
        for (const toolbar of root.querySelectorAll('.forge-nr-preset-toolbar')) {
            paint(root, toolbar);
            const input = toolbar.querySelector('input[role="listbox"]');
            if (!input || toolbar.dataset.forgeNrPresetsReady) continue;
            toolbar.dataset.forgeNrPresetsReady = 'true';
            input.placeholder = input.getAttribute('aria-label') || '';
            for (const button of toolbar.querySelectorAll('button.forge-nr-preset-button')) {
                button.title = button.textContent.trim();
                button.setAttribute('aria-label', button.title);
            }
            const prefix = toolbar.id.replace(/_preset_toolbar$/, '');
            const picker = input.closest('[id$="_preset_list"]');
            function select(option) {
                if (!option) return;
                const target = root.querySelector(`#${prefix}_preset_selection textarea`);
                if (!target) return;
                target.value = JSON.stringify({name: option.getAttribute('aria-label'), revision: ++revision});
                target.dispatchEvent(new Event('input', {bubbles: true}));
            }
            picker.addEventListener('mousedown', event => {
                if (event.button === 0) select(event.target.closest('[role="option"]'));
            }, true);
            input.addEventListener('keydown', event => {
                if (event.key === 'Enter') select(picker.querySelector('[role="option"].active'));
            }, true);
            const panel = toolbar.closest('[id$="_presets"]').parentElement;
            for (const type of ['input', 'change']) panel.addEventListener(type, () => queueMicrotask(() => paint(root, toolbar)));
        }
    }

    window.forgeNRPresets = {refresh: install};
    install();
    if (typeof onUiLoaded === 'function') onUiLoaded(install);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install, {once: true});
})();