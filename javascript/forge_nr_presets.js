(() => {
    const icons = {
        load_preset: [
            ['path', {d: 'M2 9V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H20a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-1'}],
            ['path', {d: 'M2 13h10'}],
            ['path', {d: 'm9 16 3-3-3-3'}]
        ],
        save_preset: [
            ['path', {d: 'M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z'}],
            ['path', {d: 'M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7'}],
            ['path', {d: 'M7 3v4a1 1 0 0 0 1 1h7'}]
        ],
        delete_preset: [
            ['path', {d: 'M3 6h18'}],
            ['path', {d: 'M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6'}],
            ['path', {d: 'M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2'}],
            ['line', {x1: '10', x2: '10', y1: '11', y2: '17'}],
            ['line', {x1: '14', x2: '14', y1: '11', y2: '17'}]
        ],
        refresh_presets: [
            ['path', {d: 'M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'}],
            ['path', {d: 'M21 3v5h-5'}],
            ['path', {d: 'M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'}],
            ['path', {d: 'M8 16H3v5'}]
        ]
    };

    function element(tag, attributes) {
        const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        return node;
    }

    function install() {
        const root = typeof gradioApp === 'function' ? gradioApp() : document;
        function decorate() {
            for (const input of root.querySelectorAll('.forge-nr-preset-toolbar input[role="listbox"]')) {
                input.placeholder = input.getAttribute('aria-label') || '';
            }
            for (const button of root.querySelectorAll('button.forge-nr-preset-button')) {
                if (button.querySelector('[data-forge-nr-preset-icon]')) continue;
                const action = Object.keys(icons).find(key => button.classList.contains('forge-nr-' + key));
                if (!action) continue;
                const label = button.textContent.trim();
                button.title = label;
                button.setAttribute('aria-label', label);
                const icon = element('svg', {viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
                    'stroke-width': '2', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
                    'aria-hidden': 'true', focusable: 'false', 'data-forge-nr-preset-icon': action});
                for (const [tag, attributes] of icons[action]) icon.appendChild(element(tag, attributes));
                button.appendChild(icon);
            }
        }
        decorate();
        new MutationObserver(decorate).observe(root, {childList: true, subtree: true});
    }

    if (typeof onUiLoaded === 'function') onUiLoaded(install);
    else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install, {once: true});
    else install();
})();