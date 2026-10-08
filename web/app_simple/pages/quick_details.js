/**
 * SpatialVector-HMI — Quick Details Expandable Panel.
 *
 * Provides instant technical breakdown of the primary tracked obstacle:
 * Class Name, TTC, CPA, Intersection status, and Relative Direction derived from bearing.
 */

window.QuickDetailsPage = (function () {
    "use strict";

    let containerEl = null;
    let isOpen = false;

    // Matches _LEFT_BOUNDARY (-pi/6) and _RIGHT_BOUNDARY (pi/6) from risk_engine.py
    const LEFT_BOUNDARY = -Math.PI / 6;
    const RIGHT_BOUNDARY = Math.PI / 6;

    function init(container) {
        containerEl = container;
        render();
    }

    function render() {
        containerEl.innerHTML = `
            <div id="quick-details-panel" class="expand-panel">
                <button class="expand-header" onclick="QuickDetailsPage.toggle()">
                    <span>Primary Track Details</span>
                    <span class="expand-chevron">▼</span>
                </button>
                <div class="expand-body">
                    <div class="kv-row">
                        <span class="kv-key">Target Entity</span>
                        <span id="qd-entity" class="kv-val">None</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Time to Collision (TTC)</span>
                        <span id="qd-ttc" class="kv-val">--</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Closest Approach (CPA)</span>
                        <span id="qd-cpa" class="kv-val">--</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Trajectory Intersection</span>
                        <span id="qd-intersect" class="pill-badge gray">NO</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Relative Heading</span>
                        <span id="qd-heading" class="kv-val">--</span>
                    </div>
                    <div class="kv-row" style="flex-direction: column; align-items: flex-start; gap: 4px;">
                        <span class="kv-key">Active Risk Rationale</span>
                        <div id="qd-reasons" style="font-size: 11px; color: var(--text-secondary); width: 100%;">
                            No collision hazard predicted.
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function toggle() {
        isOpen = !isOpen;
        const panel = document.getElementById("quick-details-panel");
        if (panel) {
            if (isOpen) panel.classList.add("open");
            else panel.classList.remove("open");
        }
    }

    function update(primaryTrack, riskState) {
        const entityEl = document.getElementById("qd-entity");
        const ttcEl = document.getElementById("qd-ttc");
        const cpaEl = document.getElementById("qd-cpa");
        const intEl = document.getElementById("qd-intersect");
        const headEl = document.getElementById("qd-heading");
        const reasonsEl = document.getElementById("qd-reasons");

        if (!primaryTrack) {
            if (entityEl) entityEl.textContent = "None (Corridor Clear)";
            if (ttcEl) ttcEl.textContent = "Safe";
            if (cpaEl) cpaEl.textContent = "--";
            if (intEl) { intEl.textContent = "NO"; intEl.className = "pill-badge green"; }
            if (headEl) headEl.textContent = "Forward clear";
            if (reasonsEl) reasonsEl.innerHTML = "<span>No active risk factors.</span>";
            return;
        }

        if (entityEl) {
            entityEl.textContent = `${primaryTrack.class_name || "Obstacle"} #${primaryTrack.track_id}`;
        }
        if (ttcEl) {
            ttcEl.textContent = (primaryTrack.ttc_s !== null && primaryTrack.ttc_s !== undefined)
                ? `${Number(primaryTrack.ttc_s).toFixed(1)} s`
                : "None (Safe)";
        }
        if (cpaEl) {
            cpaEl.textContent = (primaryTrack.cpa !== null && primaryTrack.cpa !== undefined)
                ? `${Number(primaryTrack.cpa).toFixed(2)} m`
                : "--";
        }
        if (intEl) {
            const isInt = Boolean(primaryTrack.intersect);
            intEl.textContent = isInt ? "YES" : "NO";
            intEl.className = isInt ? "pill-badge red" : "pill-badge green";
        }

        if (headEl) {
            const b = Number(primaryTrack.bearing || 0.0);
            if (b < LEFT_BOUNDARY) headEl.textContent = "From the left";
            else if (b > RIGHT_BOUNDARY) headEl.textContent = "From the right";
            else headEl.textContent = "Toward you (Center path)";
        }

        if (reasonsEl && riskState) {
            const formatted = ReasoningFormatter.formatList(riskState.reason_codes, primaryTrack.track_id);
            reasonsEl.innerHTML = formatted.map(r => `• ${r}`).join("<br>");
        }
    }

    return {
        init: init,
        toggle: toggle,
        update: update,
    };
})();
