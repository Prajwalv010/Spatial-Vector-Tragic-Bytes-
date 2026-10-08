/**
 * SpatialVector-HMI — Tab 2: Haptics Page
 * Redesigned: 2-motor hardware, clean status, no battery, no motor count mismatch
 */

window.HapticsPage = (function () {
    "use strict";

    let containerEl = null;
    let diagramController = null;

    function init(container) {
        containerEl = container;
        renderShell();

        const vestBox = document.getElementById("haptics-vest-diagram");
        diagramController = HapticBodyDiagram.create(vestBox, { maxUrgency: 5 });
    }

    function renderShell() {
        containerEl.innerHTML = `
            <!-- Active Status Card -->
            <div class="expand-panel open" style="margin-bottom: 14px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Haptic Feedback Status</span>
                    <span id="haptic-status-chip" class="pill-badge gray">○ Standby</span>
                </div>
                <div class="expand-body" style="display: block; background: #ffffff; padding: 14px 10px;">
                    <div id="haptics-vest-diagram"></div>
                </div>
            </div>

            <!-- Current Command Card -->
            <div class="expand-panel open" style="margin-bottom: 14px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Active Motor Command</span>
                    <span id="haptic-urg-badge" class="pill-badge blue">Urgency 1/5</span>
                </div>
                <div class="expand-body" style="display: block; background: #ffffff;">

                    <!-- Big direction display -->
                    <div id="haptic-direction-display" style="
                        text-align: center;
                        padding: 14px 0 10px;
                        font-size: 32px;
                        font-weight: 800;
                        color: var(--brand-primary);
                        letter-spacing: 0.04em;
                    ">—</div>

                    <div class="kv-row">
                        <span class="kv-key">What this means</span>
                        <span id="haptic-meaning-val" style="font-size: 12px; font-weight: 600; color: var(--text-primary); text-align: right; max-width: 60%;">
                            No obstacle — path is clear.
                        </span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Motor Pattern</span>
                        <span id="haptic-pattern-val" class="kv-val">ALL_CLEAR</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Pulse Duration</span>
                        <span id="haptic-dur-val" class="kv-val">200 ms</span>
                    </div>
                </div>
            </div>

            <!-- 2-Motor Hardware Status -->
            <div class="expand-panel open" style="margin-bottom: 14px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Hardware — 2 Vibration Motors</span>
                    <span id="dev-hw-status" class="pill-badge green">Connected</span>
                </div>
                <div class="expand-body" style="display: block; background: #ffffff;">
                    <div class="haptic-motor-grid">
                        <div class="haptic-motor-card" id="motor-left-card">
                            <div class="motor-icon">📳</div>
                            <div class="motor-name">Left Motor</div>
                            <div id="motor-left-state" class="motor-state idle">Idle</div>
                        </div>
                        <div class="haptic-motor-card" id="motor-right-card">
                            <div class="motor-icon">📳</div>
                            <div class="motor-name">Right Motor</div>
                            <div id="motor-right-state" class="motor-state idle">Idle</div>
                        </div>
                    </div>
                    <div class="kv-row" style="margin-top: 8px;">
                        <span class="kv-key">Interface</span>
                        <span id="dev-link-status" class="kv-val">Arduino Uno Q · Serial</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Command Latency</span>
                        <span id="dev-latency-val" class="kv-val">< 4 ms</span>
                    </div>
                </div>
            </div>

            <!-- Haptic Language Guide -->
            <div class="guidance-prompt-card" style="margin-bottom: 14px;">
                <span class="guidance-icon">💡</span>
                <span class="guidance-text">
                    <strong>How to read vibrations:</strong> Left motor = steer left. Right motor = steer right. Both motors = stop.
                </span>
            </div>

            <!-- Pattern Reference (Expandable) -->
            <div class="expand-panel" id="panel-haptic-patterns">
                <button class="expand-header" onclick="HapticsPage.togglePanel('panel-haptic-patterns')">
                    <span>Haptic Pattern Reference</span>
                    <span class="expand-chevron">▼</span>
                </button>
                <div class="expand-body">
                    <div class="kv-row">
                        <span class="kv-key"><code>ALL_CLEAR</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Silent — both motors idle. Path is safe.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>LEFT_MED</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Left motor pulses — steer left.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>RIGHT_MED</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Right motor pulses — steer right.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>LEFT_FAST</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Left motor rapid — urgent, move left now.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>RIGHT_FAST</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Right motor rapid — urgent, move right now.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>STOP_CRITICAL</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Both motors rapid — halt immediately.</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key"><code>DEGRADED_WARN</code></span>
                        <span style="font-size: 11px; color: var(--text-secondary);">Slow alternating pulse — sensor issue, use caution.</span>
                    </div>
                </div>
            </div>
        `;
    }

    function togglePanel(id) {
        const p = document.getElementById(id);
        if (p) p.classList.toggle("open");
    }

    function updateTelemetry(msg) {
        if (!msg) return;
        const haptic = msg.haptic || {};
        const health = msg.pipeline_health || {};

        if (diagramController) {
            diagramController.update(haptic);
        }

        const dir = (haptic.direction || "STOP").toUpperCase();
        const pattern = haptic.pattern_id || "ALL_CLEAR";
        const urgency = haptic.urgency || 1;
        const dur = haptic.duration_ms || 200;
        const isVibrating = pattern !== "ALL_CLEAR" && pattern !== "DEGRADED_WARN";

        // Status Chip
        const statusChip = document.getElementById("haptic-status-chip");
        if (statusChip) {
            statusChip.textContent = isVibrating ? "● Active" : "○ Standby";
            statusChip.className = isVibrating ? "pill-badge blue" : "pill-badge gray";
        }

        // Urgency Badge
        const urgBadge = document.getElementById("haptic-urg-badge");
        if (urgBadge) {
            urgBadge.textContent = `Urgency ${urgency}/5`;
            urgBadge.className = urgency >= 4 ? "pill-badge red" : (urgency >= 3 ? "pill-badge orange" : "pill-badge blue");
        }

        // Big Direction Display
        const dirDisplay = document.getElementById("haptic-direction-display");
        if (dirDisplay) {
            if (dir === "LEFT") dirDisplay.innerHTML = `<span style="color:#2563eb">← Left</span>`;
            else if (dir === "RIGHT") dirDisplay.innerHTML = `<span style="color:#2563eb">Right →</span>`;
            else if (dir === "STOP" && isVibrating) dirDisplay.innerHTML = `<span style="color:#ef4444">⏹ Stop</span>`;
            else dirDisplay.innerHTML = `<span style="color:#10b981">✓ Clear</span>`;
        }

        // Text fields
        const patVal = document.getElementById("haptic-pattern-val");
        const durVal = document.getElementById("haptic-dur-val");
        const meaningVal = document.getElementById("haptic-meaning-val");

        if (patVal) patVal.textContent = pattern;
        if (durVal) durVal.textContent = `${dur} ms`;
        if (meaningVal) meaningVal.textContent = HapticBodyDiagram.getDescription(pattern);

        // 2-Motor state display
        const leftCard = document.getElementById("motor-left-card");
        const rightCard = document.getElementById("motor-right-card");
        const leftState = document.getElementById("motor-left-state");
        const rightState = document.getElementById("motor-right-state");

        const leftActive = dir === "LEFT" || (dir === "STOP" && isVibrating);
        const rightActive = dir === "RIGHT" || (dir === "STOP" && isVibrating);

        if (leftState) {
            leftState.textContent = leftActive ? "● Active" : "Idle";
            leftState.className = leftActive ? "motor-state active" : "motor-state idle";
        }
        if (rightState) {
            rightState.textContent = rightActive ? "● Active" : "Idle";
            rightState.className = rightActive ? "motor-state active" : "motor-state idle";
        }
        if (leftCard) leftCard.classList.toggle("motor-firing", leftActive);
        if (rightCard) rightCard.classList.toggle("motor-firing", rightActive);

        // Hardware Link
        const hwStatus = document.getElementById("dev-hw-status");
        if (hwStatus && health.arduino) {
            hwStatus.textContent = health.arduino.includes("OK") ? "Connected" : "Check Connection";
            hwStatus.className = health.arduino.includes("OK") ? "pill-badge green" : "pill-badge red";
        }
    }

    return {
        init: init,
        togglePanel: togglePanel,
        updateTelemetry: updateTelemetry,
    };
})();
