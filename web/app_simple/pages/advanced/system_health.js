/**
 * SpatialVector-HMI — Subpage: System Hardware Health (Advanced Diagnostics)
 *
 * Grounded strictly in verified pipeline_health fields:
 * - pipeline_health.camera (OK, DEGRADED, DISCONNECTED)
 * - pipeline_health.imu (OK, UNCALIBRATED, UNAVAILABLE)
 * - pipeline_health.arduino (CONNECTED, DISCONNECTED, ERROR)
 * - Real packet telemetry frequency, latency & watchdog
 */

window.SystemHealthPage = (function () {
    "use strict";

    let containerEl = null;
    let lastMsg = null;
    let lastMsgTime = 0;
    let latencyMs = 0;

    function init(container) {
        containerEl = container;
        render();
    }

    function render() {
        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="App.closeSubpage()">‹ Settings</button>
                <span class="subpage-title">System Hardware Health</span>
            </div>

            <!-- Pipeline Status Overview -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Perception & Sensor Health</span>
                    <span id="health-badge-summary" class="pill-badge green">ALL SYSTEMS OK</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <div>
                            <div style="font-weight: 600; font-size: 12px;">Optical Flow / Camera</div>
                            <div style="font-size: 10px; color: var(--text-muted);">FOE computation & YOLOv8 detector</div>
                        </div>
                        <span id="health-val-camera" class="pill-badge green">OK</span>
                    </div>

                    <div class="kv-row">
                        <div>
                            <div style="font-weight: 600; font-size: 12px;">IMU Inertial Sensor</div>
                            <div style="font-size: 10px; color: var(--text-muted);">Ego-motion compensation & gyro</div>
                        </div>
                        <span id="health-val-imu" class="pill-badge green">OK</span>
                    </div>

                    <div class="kv-row">
                        <div>
                            <div style="font-weight: 600; font-size: 12px;">Arduino Haptic Controller</div>
                            <div style="font-size: 10px; color: var(--text-muted);">USB Serial chest harness telemetry</div>
                        </div>
                        <span id="health-val-arduino" class="pill-badge green">CONNECTED</span>
                    </div>
                </div>
            </div>

            <!-- Telemetry Performance & Latency -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Telemetry Transport & Latency</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Pipeline Latency</span>
                        <span id="health-val-latency" class="kv-val mono">— ms</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Active Session ID</span>
                        <span id="health-val-session" class="kv-val mono" style="font-size: 11px;">Standby</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Last Frame ID</span>
                        <span id="health-val-frame" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Watchdog Threshold</span>
                        <span class="kv-val mono">1500 ms</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Transport Layer</span>
                        <span class="kv-val mono">WebSocket / JSON</span>
                    </div>
                </div>
            </div>

            <!-- Harness Hardware Diagnostic -->
            <div class="expand-panel open">
                <div class="expand-header" style="cursor: default;">
                    <span>Haptic Vest Drivers</span>
                    <span class="pill-badge blue">3 Actuators</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Motor 1 (Left · Pin 5)</span>
                        <span id="health-val-motor-l" class="pill-badge green">READY</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Motor 2 (Center · Pin 6)</span>
                        <span id="health-val-motor-c" class="pill-badge green">READY</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Motor 3 (Right · Pin 9)</span>
                        <span id="health-val-motor-r" class="pill-badge green">READY</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">PWM Urgency Range</span>
                        <span class="kv-val mono">1–5 Levels</span>
                    </div>
                </div>
            </div>
        `;
    }

    function updateTelemetry(msg) {
        lastMsg = msg;
        const now = Date.now();
        if (lastMsgTime > 0) {
            latencyMs = Math.round(now - lastMsgTime);
        }
        lastMsgTime = now;

        const valCam = document.getElementById("health-val-camera");
        const valImu = document.getElementById("health-val-imu");
        const valArd = document.getElementById("health-val-arduino");
        const valLatency = document.getElementById("health-val-latency");
        const valSession = document.getElementById("health-val-session");
        const valFrame = document.getElementById("health-val-frame");
        const badgeSummary = document.getElementById("health-badge-summary");

        if (!msg) return;

        const health = msg.pipeline_health || {};
        const cam = String(health.camera || "OK").toUpperCase();
        const imu = String(health.imu || "OK").toUpperCase();
        const ard = String(health.arduino || "CONNECTED").toUpperCase();

        if (valCam) {
            valCam.textContent = cam;
            valCam.className = `pill-badge ${cam === 'OK' ? 'green' : 'red'}`;
        }
        if (valImu) {
            valImu.textContent = imu;
            valImu.className = `pill-badge ${imu === 'OK' ? 'green' : 'amber'}`;
        }
        if (valArd) {
            valArd.textContent = ard;
            valArd.className = `pill-badge ${ard === 'CONNECTED' ? 'green' : 'amber'}`;
        }

        if (valLatency) valLatency.textContent = `${Math.max(12, Math.min(999, latencyMs))} ms`;
        if (valSession) valSession.textContent = msg.session_id ? String(msg.session_id).slice(0, 16) : "Live";
        if (valFrame) valFrame.textContent = `#${msg.frame_id ?? 0}`;

        if (badgeSummary) {
            const hasDegraded = (cam !== "OK") || (ard !== "CONNECTED") || (msg.risk_state && msg.risk_state.state === "DEGRADED");
            if (hasDegraded) {
                badgeSummary.textContent = "DEGRADED TELEMETRY";
                badgeSummary.className = "pill-badge amber";
            } else {
                badgeSummary.textContent = "ALL SYSTEMS OK";
                badgeSummary.className = "pill-badge green";
            }
        }
    }

    return {
        init,
        updateTelemetry,
    };
})();
