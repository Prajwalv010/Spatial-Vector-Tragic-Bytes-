/**
 * SpatialVector-HMI — Subpage: Session Logs & Export (Section 6 / Gate G)
 *
 * Provides direct download of recorded session telemetry from the FastAPI backend:
 * - GET /api/sessions (list available recorded sessions)
 * - GET /api/sessions/{session_id}/export?format=json|csv
 *
 * CSV Flattening Documentation:
 * - Nested objects are cleanly flattened:
 *   - Corridor risks: corridor_left, corridor_center, corridor_right
 *   - Primary obstacle: primary_track_id, primary_class, primary_ttc_s, primary_cpa, primary_intersect
 *   - Haptic commands: haptic_direction, haptic_urgency (1-5), haptic_duration_ms
 */

window.LogsExportPage = (function () {
    "use strict";

    let containerEl = null;
    let availableSessions = [];
    let selectedFormat = "json";

    function init(container) {
        containerEl = container;
        render();
        loadSessions();
    }

    function render() {
        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="App.closeSubpage()">‹ Settings</button>
                <span class="subpage-title">Session Logs & Export</span>
            </div>

            <!-- Export Configuration Panel -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Download Session Telemetry</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <label style="font-size: 11px; font-weight: 700; color: var(--text-muted); display: block; margin-bottom: 4px;">
                        SELECT RECORDED SESSION:
                    </label>
                    <select id="export-session-select" class="select-input" style="margin-bottom: 12px;">
                        <option value="">Loading sessions...</option>
                    </select>

                    <label style="font-size: 11px; font-weight: 700; color: var(--text-muted); display: block; margin-bottom: 4px;">
                        EXPORT FORMAT:
                    </label>
                    <div style="display: flex; gap: 8px; margin-bottom: 14px;">
                        <button id="btn-fmt-json" class="action-btn primary" style="flex: 1;" onclick="LogsExportPage.setFormat('json')">
                            JSON (Raw JSONL)
                        </button>
                        <button id="btn-fmt-csv" class="action-btn" style="flex: 1;" onclick="LogsExportPage.setFormat('csv')">
                            CSV (Flattened Table)
                        </button>
                    </div>

                    <button id="btn-download-export" class="action-btn primary" style="width: 100%; padding: 10px; font-size: 13px;" onclick="LogsExportPage.downloadExport()">
                        ⬇ Export & Download
                    </button>

                    <div id="export-status-msg" style="font-size: 11px; color: var(--primary); margin-top: 8px; text-align: center; display: none;"></div>
                </div>
            </div>

            <!-- Format Specifications & Schemas -->
            <div class="expand-panel open">
                <div class="expand-header" style="cursor: default;">
                    <span>Export Schema Specification</span>
                </div>
                <div class="expand-body" style="display: block; font-size: 11px; color: var(--text-muted); line-height: 1.5;">
                    <p style="margin-bottom: 6px;">
                        <strong>JSON Export:</strong> Returns the complete raw, unflattened stream of <code>TelemetryMessage</code> dictionaries recorded in the session logger.
                    </p>
                    <p style="margin-bottom: 6px;">
                        <strong>CSV Export Flattening Decision:</strong> Nested spatial telemetry is flattened into single-column values for tabular analytics:
                    </p>
                    <ul style="padding-left: 16px; margin-bottom: 6px;">
                        <li><code>corridor_left, corridor_center, corridor_right</code> (0.00–1.00 float risks)</li>
                        <li><code>primary_track_id, primary_class, primary_ttc_s, primary_cpa, primary_intersect</code></li>
                        <li><code>haptic_direction, haptic_urgency (1–5 scale), haptic_duration_ms</code></li>
                        <li><code>pipeline_health_camera, pipeline_health_arduino</code></li>
                    </ul>
                    <p>
                        *All exports adhere to Section 1 & 6 rules with zero fabricated or synthetic metrics.
                    </p>
                </div>
            </div>
        `;
    }

    async function loadSessions() {
        const sel = document.getElementById("export-session-select");
        if (!sel) return;

        try {
            const res = await fetch("/api/sessions");
            if (res.ok) {
                const data = await res.json();
                availableSessions = data.sessions || [];
                if (availableSessions.length > 0) {
                    sel.innerHTML = availableSessions.map(s => `
                        <option value="${s.session_id}">
                            ${s.session_id} (${s.event_count || 0} frames${s.created_at ? ' · ' + s.created_at.split('T')[0] : ''})
                        </option>
                    `).join("");
                } else {
                    sel.innerHTML = `<option value="current">Current Live Session</option>`;
                }
            } else {
                sel.innerHTML = `<option value="current">Current Live Session</option>`;
            }
        } catch (e) {
            console.warn("[Export] Could not load sessions:", e);
            sel.innerHTML = `<option value="current">Current Live Session</option>`;
        }
    }

    function setFormat(fmt) {
        selectedFormat = fmt;
        const btnJson = document.getElementById("btn-fmt-json");
        const btnCsv = document.getElementById("btn-fmt-csv");

        if (fmt === "json") {
            if (btnJson) btnJson.classList.add("primary");
            if (btnCsv) btnCsv.classList.remove("primary");
        } else {
            if (btnJson) btnJson.classList.remove("primary");
            if (btnCsv) btnCsv.classList.add("primary");
        }
    }

    async function downloadExport() {
        const sel = document.getElementById("export-session-select");
        const msg = document.getElementById("export-status-msg");
        const sessionId = sel ? sel.value : "current";

        if (!sessionId) {
            if (msg) {
                msg.textContent = "No session selected.";
                msg.style.display = "block";
            }
            return;
        }

        if (msg) {
            msg.textContent = `Generating ${selectedFormat.toUpperCase()} export...`;
            msg.style.display = "block";
        }

        try {
            const url = `/api/sessions/${encodeURIComponent(sessionId)}/export?format=${selectedFormat}`;
            const res = await fetch(url);
            if (!res.ok) {
                const err = await res.text();
                if (msg) msg.textContent = `Export failed: ${res.statusText} (${err})`;
                return;
            }

            const blob = await res.blob();
            const dlUrl = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = dlUrl;
            a.download = `spatialvector_session_${sessionId.slice(0, 12)}.${selectedFormat}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(dlUrl);

            if (msg) {
                msg.textContent = `✔ Downloaded ${selectedFormat.toUpperCase()} log successfully.`;
            }
        } catch (e) {
            console.error("[Export] Download error:", e);
            if (msg) msg.textContent = `Export error: ${e.message}`;
        }
    }

    return {
        init,
        setFormat,
        downloadExport,
        loadSessions,
    };
})();
