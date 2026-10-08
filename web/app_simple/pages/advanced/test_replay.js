/**
 * SpatialVector-HMI — Advanced Subpage: Test Scenarios (S1–S6)
 * Redesigned: Scenario-card menu replacing old dropdown+replay console.
 * Phone is the test manager. Arduino is the hardware executor.
 * Connects to the real system pipeline via existing WebSocket telemetry.
 */

window.TestReplayPage = (function () {
    "use strict";

    let containerEl = null;
    let canvasEl = null;
    let currentScenarioId = null;
    let testState = "idle"; // idle | running | result
    let liveRisk = null;
    let liveHaptic = null;
    let testStartTime = null;
    let testTimer = null;
    let testElapsed = 0;
    let testHistory = [];

    const SCENARIOS = [
        {
            id: "s1",
            title: "Parallel Wall",
            subtitle: "Close-distance walking along a wall",
            icon: "🧱",
            expectedRisk: "SAFE",
            expectedHaptic: "SILENT",
            expectedRiskLevel: "SAFE",
            description: "The user walks parallel to a wall at close lateral distance. The system should recognise that the wall is not on an intersecting trajectory and remain silent — it should NOT trigger false warnings simply because an object is nearby.",
            purpose: "Validates that proximity alone does not trigger unnecessary warnings.",
            category: "Baseline",
        },
        {
            id: "s2",
            title: "Head-On Obstacle",
            subtitle: "Approaching pedestrian or scooter",
            icon: "🚶",
            expectedRisk: "WARNING",
            expectedHaptic: "DIRECTIONAL",
            expectedRiskLevel: "WARNING",
            description: "An obstacle approaches the user directly along the forward corridor. The system should detect trajectory intersection, rising TTC urgency, and issue a directional haptic guidance command.",
            purpose: "Validates TTC calculation and directional haptic output.",
            category: "Collision",
        },
        {
            id: "s3",
            title: "Crossing Object",
            subtitle: "Pedestrian crossing your path",
            icon: "↗️",
            expectedRisk: "WARNING",
            expectedHaptic: "DIRECTIONAL",
            expectedRiskLevel: "WARNING",
            description: "A pedestrian or scooter crosses the user's predicted path at a lateral angle. CPA should drop below threshold and the system should warn.",
            purpose: "Validates lateral intersection detection.",
            category: "Collision",
        },
        {
            id: "s4",
            title: "Static Obstacle",
            subtitle: "Parked scooter or object ahead",
            icon: "🛵",
            expectedRisk: "WARNING",
            expectedHaptic: "DIRECTIONAL",
            expectedRiskLevel: "WARNING",
            description: "A stationary object is in the user's walking corridor. As the user approaches, TTC decreases and the system should guide around it.",
            purpose: "Validates static obstacle handling and path planning.",
            category: "Obstacle",
        },
        {
            id: "s5",
            title: "Multiple Obstacles",
            subtitle: "Crowded environment",
            icon: "👥",
            expectedRisk: "CRITICAL",
            expectedHaptic: "URGENT",
            expectedRiskLevel: "CRITICAL",
            description: "Multiple obstacles with overlapping trajectories converge on the user's path. The system must identify the highest-risk object and issue appropriate multi-motor guidance.",
            purpose: "Validates multi-object tracking and priority selection.",
            category: "Complex",
        },
        {
            id: "s6",
            title: "Narrow Corridor",
            subtitle: "Tight walkway with walls on both sides",
            icon: "↔️",
            expectedRisk: "SAFE",
            expectedHaptic: "SILENT",
            expectedRiskLevel: "SAFE",
            description: "The user navigates a narrow passage. Walls on both sides should be identified as non-threatening (parallel, not intersecting). The system should guide the user through the safe corridor without false warnings.",
            purpose: "Validates spatial reasoning and safe passage detection.",
            category: "Spatial",
        },
    ];

    const RISK_CURVES = {
        s1: [0.05, 0.08, 0.11, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.05],
        s2: [0.15, 0.22, 0.38, 0.55, 0.72, 0.84, 0.78, 0.45, 0.20, 0.10],
        s3: [0.10, 0.18, 0.35, 0.62, 0.76, 0.68, 0.30, 0.15, 0.08, 0.05],
        s4: [0.08, 0.15, 0.28, 0.48, 0.65, 0.58, 0.35, 0.18, 0.08, 0.04],
        s5: [0.20, 0.35, 0.52, 0.78, 0.95, 0.91, 0.84, 0.60, 0.35, 0.18],
        s6: [0.08, 0.12, 0.18, 0.22, 0.19, 0.14, 0.09, 0.06, 0.04, 0.02],
    };

    function init(container) {
        containerEl = container;
        renderScenarioMenu();
    }

    /* ============================================================
       SCENARIO MENU
       ============================================================ */
    function renderScenarioMenu() {
        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="App.closeSubpage()">‹ Settings</button>
                <span class="subpage-title">Test Scenarios</span>
            </div>

            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
                Validate system behaviour with controlled scenarios. The live pipeline is the system under test.
            </p>

            <!-- Quick Test Row -->
            <div style="margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.06em; margin-bottom: 8px;">Quick Test</div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    ${SCENARIOS.slice(0, 3).map(s => `
                        <button class="quick-test-chip" onclick="TestReplayPage.openScenarioDetail('${s.id}')">
                            ${s.icon} ${s.title.split(' ').slice(0, 2).join(' ')}
                        </button>
                    `).join('')}
                </div>
            </div>

            <!-- Scenario List -->
            <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.06em; margin-bottom: 8px;">All Scenarios</div>
            <div class="scenario-list">
                ${SCENARIOS.map(s => renderScenarioCard(s)).join('')}
            </div>

            <!-- Test History -->
            <div id="test-history-section" style="margin-top: 16px; display: ${testHistory.length > 0 ? 'block' : 'none'};">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.06em; margin-bottom: 8px;">Recent Tests</div>
                <div id="test-history-list" class="scenario-list">
                    ${renderHistoryItems()}
                </div>
            </div>
        `;
    }

    function renderScenarioCard(s) {
        const riskClass = s.expectedRiskLevel === "CRITICAL" ? "red"
            : s.expectedRiskLevel === "WARNING" ? "orange"
            : "green";
        return `
            <div class="scenario-card" onclick="TestReplayPage.openScenarioDetail('${s.id}')">
                <div class="scenario-card-icon">${s.icon}</div>
                <div class="scenario-card-body">
                    <div class="scenario-card-title"><strong>${s.id.toUpperCase()}</strong> — ${s.title}</div>
                    <div class="scenario-card-desc">${s.subtitle}</div>
                </div>
                <div class="scenario-card-right">
                    <span class="pill-badge ${riskClass}">${s.expectedRisk}</span>
                    <span style="color: var(--text-muted); font-size: 18px;">›</span>
                </div>
            </div>
        `;
    }

    function renderHistoryItems() {
        if (testHistory.length === 0) return '';
        return testHistory.slice(0, 5).map(h => {
            const scen = SCENARIOS.find(s => s.id === h.scenarioId);
            const icon = h.pass ? '✓' : '✕';
            const color = h.pass ? '#10b981' : '#ef4444';
            const badge = h.pass ? 'green' : 'red';
            return `
                <div class="scenario-card" onclick="TestReplayPage.openHistoryResult('${h.id}')">
                    <div style="font-size: 20px; color: ${color}; width: 28px;">${icon}</div>
                    <div class="scenario-card-body">
                        <div class="scenario-card-title">${scen ? scen.icon + ' ' + scen.title : h.scenarioId}</div>
                        <div class="scenario-card-desc">${h.time}</div>
                    </div>
                    <span class="pill-badge ${badge}">${h.pass ? 'PASS' : 'FAIL'}</span>
                </div>
            `;
        }).join('');
    }

    /* ============================================================
       SCENARIO DETAIL VIEW
       ============================================================ */
    function openScenarioDetail(scenId) {
        currentScenarioId = scenId;
        const scen = SCENARIOS.find(s => s.id === scenId);
        if (!scen) return;

        const riskClass = scen.expectedRiskLevel === "CRITICAL" ? "red"
            : scen.expectedRiskLevel === "WARNING" ? "orange" : "green";

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="TestReplayPage.renderScenarioMenu()">‹ Scenarios</button>
                <span class="subpage-title">${scen.id.toUpperCase()} — ${scen.title}</span>
            </div>

            <!-- Hero Icon + Expected State -->
            <div style="text-align: center; padding: 20px 0 16px;">
                <div style="font-size: 52px; margin-bottom: 8px;">${scen.icon}</div>
                <div style="font-size: 15px; font-weight: 800; color: var(--text-primary);">${scen.title}</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${scen.subtitle}</div>
                <div style="margin-top: 10px;">
                    <span class="pill-badge ${riskClass}" style="font-size: 13px; padding: 4px 14px;">Expected: ${scen.expectedRisk}</span>
                </div>
            </div>

            <!-- Description -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Scenario Description</span>
                    <span class="pill-badge gray">${scen.category}</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.7;">${scen.description}</p>
                    <p style="font-size: 11px; color: var(--text-muted); margin-top: 8px; font-style: italic;">Purpose: ${scen.purpose}</p>
                </div>
            </div>

            <!-- Expected Behaviour -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Expected System Behaviour</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Risk State</span>
                        <span class="pill-badge ${riskClass}">${scen.expectedRisk}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Haptic Output</span>
                        <span class="kv-val">${scen.expectedHaptic}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Trajectory Intersection</span>
                        <span class="kv-val">${scen.expectedRiskLevel !== "SAFE" ? "YES" : "NO"}</span>
                    </div>
                </div>
            </div>

            <!-- Risk Curve Preview -->
            <div class="expand-panel open" style="margin-bottom: 16px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Risk Profile (Reference)</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div style="background: #f8fafc; border-radius: 8px; padding: 4px; height: 80px;">
                        <canvas id="scenario-preview-canvas" width="340" height="72" style="width: 100%; height: 100%;"></canvas>
                    </div>
                    <p style="font-size: 10px; color: var(--text-muted); margin-top: 6px; text-align: center;">Reference risk curve for this scenario type</p>
                </div>
            </div>

            <!-- Start Test Button -->
            <button class="action-btn" style="margin-bottom: 10px;" onclick="TestReplayPage.startTest('${scen.id}')">
                ▶ Start Test
            </button>
            <button class="action-btn secondary" onclick="TestReplayPage.renderScenarioMenu()">
                Cancel
            </button>
        `;

        // Draw preview curve
        const canvas = document.getElementById("scenario-preview-canvas");
        if (canvas) drawCurve(canvas, RISK_CURVES[scenId] || RISK_CURVES.s1, -1);
    }

    /* ============================================================
       TEST EXECUTION VIEW
       ============================================================ */
    function startTest(scenId) {
        currentScenarioId = scenId;
        testState = "running";
        testStartTime = Date.now();
        testElapsed = 0;
        liveRisk = null;
        liveHaptic = null;

        const scen = SCENARIOS.find(s => s.id === scenId);
        if (!scen) return;

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="TestReplayPage.stopTest()">‹ Stop</button>
                <span class="subpage-title">Running Test</span>
            </div>

            <div style="text-align: center; padding: 16px 0;">
                <div style="font-size: 40px;">${scen.icon}</div>
                <div style="font-size: 14px; font-weight: 800; color: var(--text-primary); margin-top: 8px;">${scen.id.toUpperCase()} — ${scen.title}</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;" id="test-elapsed-timer">Elapsed: 0s</div>
            </div>

            <!-- Live System Status -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Live System Status</span>
                    <span class="pill-badge blue" id="test-run-badge">● Running</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Pipeline</span>
                        <span class="pill-badge green">Active</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Camera</span>
                        <span id="test-camera-status" class="pill-badge gray">Waiting...</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Tracking</span>
                        <span id="test-track-status" class="pill-badge gray">Waiting...</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Haptic Output</span>
                        <span id="test-haptic-status" class="pill-badge gray">Waiting...</span>
                    </div>
                </div>
            </div>

            <!-- Live Observed Result -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Observed Result (Live)</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div style="text-align: center; padding: 12px 0;">
                        <div id="test-live-risk-display" style="font-size: 28px; font-weight: 800; color: var(--text-muted);">Waiting for data...</div>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Expected</span>
                        <span class="kv-val">${scen.expectedRisk}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Observed</span>
                        <span id="test-live-observed" class="kv-val">—</span>
                    </div>
                </div>
            </div>

            <!-- Stop & Evaluate -->
            <button class="action-btn" style="background: #ef4444; margin-bottom: 8px;" onclick="TestReplayPage.stopTest()">
                ⏹ Stop & Evaluate
            </button>
        `;

        // Elapsed timer
        testTimer = setInterval(() => {
            testElapsed = Math.round((Date.now() - testStartTime) / 1000);
            const timerEl = document.getElementById("test-elapsed-timer");
            if (timerEl) timerEl.textContent = `Elapsed: ${testElapsed}s`;
            refreshRunStatus();
        }, 1000);
    }

    function refreshRunStatus() {
        const lastMsg = window.App && window.App.getLastTelemetry ? window.App.getLastTelemetry() : null;
        if (!lastMsg) return;

        const risk = lastMsg.risk_state || {};
        const haptic = lastMsg.haptic || {};
        const tracks = lastMsg.tracks || [];
        const state = (risk.state || "SAFE").toUpperCase();

        liveRisk = state;
        liveHaptic = (haptic.direction || "STOP").toUpperCase();

        const stateColor =
            state === "CRITICAL" ? "#ef4444" :
            state === "WARNING" ? "#f97316" :
            state === "CAUTION" ? "#f59e0b" : "#10b981";

        const camEl = document.getElementById("test-camera-status");
        const trackEl = document.getElementById("test-track-status");
        const hapticEl = document.getElementById("test-haptic-status");
        const liveEl = document.getElementById("test-live-risk-display");
        const obsEl = document.getElementById("test-live-observed");

        if (camEl) { camEl.textContent = "Active"; camEl.className = "pill-badge green"; }
        if (trackEl) {
            trackEl.textContent = tracks.length > 0 ? `${tracks.length} Objects` : "Clear";
            trackEl.className = tracks.length > 0 ? "pill-badge yellow" : "pill-badge green";
        }
        if (hapticEl) {
            const pattern = haptic.pattern_id || "ALL_CLEAR";
            hapticEl.textContent = pattern === "ALL_CLEAR" ? "Silent" : liveHaptic;
            hapticEl.className = pattern === "ALL_CLEAR" ? "pill-badge gray" : "pill-badge blue";
        }
        if (liveEl) {
            liveEl.textContent = state;
            liveEl.style.color = stateColor;
        }
        if (obsEl) obsEl.textContent = state;
    }

    function stopTest() {
        if (testTimer) clearInterval(testTimer);
        testTimer = null;
        testState = "result";

        const scenId = currentScenarioId;
        const scen = SCENARIOS.find(s => s.id === scenId);
        if (!scen) { renderScenarioMenu(); return; }

        // Evaluate pass/fail
        const observedRisk = liveRisk || "UNKNOWN";
        const pass = observedRisk === scen.expectedRiskLevel ||
            (scen.expectedRiskLevel !== "SAFE" && observedRisk !== "SAFE" && observedRisk !== "UNKNOWN");

        // Store in history
        const histEntry = {
            id: `h_${Date.now()}`,
            scenarioId: scenId,
            time: new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }),
            expectedRisk: scen.expectedRisk,
            observedRisk: observedRisk,
            observedHaptic: liveHaptic || "UNKNOWN",
            elapsed: testElapsed,
            pass: pass,
        };
        testHistory.unshift(histEntry);
        if (testHistory.length > 10) testHistory.pop();

        renderTestResult(histEntry, scen);
    }

    /* ============================================================
       TEST RESULT VIEW
       ============================================================ */
    function renderTestResult(result, scen) {
        const pass = result.pass;
        const passColor = pass ? "#10b981" : "#ef4444";
        const passIcon = pass ? "✓" : "✕";
        const passLabel = pass ? "TEST PASSED" : "TEST FAILED";
        const passBadge = pass ? "green" : "red";

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="TestReplayPage.renderScenarioMenu()">‹ Scenarios</button>
                <span class="subpage-title">Test Result</span>
            </div>

            <!-- Pass/Fail Hero -->
            <div style="text-align: center; padding: 24px 0 16px;">
                <div style="font-size: 56px; color: ${passColor};">${passIcon}</div>
                <div style="font-size: 20px; font-weight: 800; color: ${passColor}; margin-top: 8px;">${passLabel}</div>
                <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">${scen.icon} ${scen.id.toUpperCase()} — ${scen.title}</div>
            </div>

            <!-- Result Detail -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Result Summary</span>
                    <span class="pill-badge ${passBadge}">${pass ? "✓ PASS" : "✕ FAIL"}</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Expected Risk</span>
                        <span class="kv-val">${result.expectedRisk}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Observed Risk</span>
                        <span class="kv-val" style="color: ${passColor};">${result.observedRisk}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Haptic Direction</span>
                        <span class="kv-val">${result.observedHaptic}</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Test Duration</span>
                        <span class="kv-val">${result.elapsed}s</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Time</span>
                        <span class="kv-val">${result.time}</span>
                    </div>
                    ${!pass ? `
                    <div style="margin-top: 8px; padding: 8px; background: #fef2f2; border-radius: 8px; border: 1px solid #fecaca;">
                        <div style="font-size: 11px; color: #991b1b; font-weight: 600;">⚠ Possible causes:</div>
                        <div style="font-size: 11px; color: #991b1b; margin-top: 4px;">• No obstacle in camera view during test<br>• Camera not tracking — check /dev/video2<br>• Pipeline still warming up — try again</div>
                    </div>` : ""}
                </div>
            </div>

            <!-- Actions -->
            <button class="action-btn" style="margin-bottom: 8px;" onclick="TestReplayPage.startTest('${scen.id}')">
                ↺ Run Again
            </button>
            <button class="action-btn secondary" onclick="TestReplayPage.renderScenarioMenu()">
                ← Back to Scenarios
            </button>
        `;
    }

    function openHistoryResult(histId) {
        const entry = testHistory.find(h => h.id === histId);
        if (!entry) return;
        const scen = SCENARIOS.find(s => s.id === entry.scenarioId);
        if (!scen) return;
        renderTestResult(entry, scen);
    }

    /* ============================================================
       CANVAS CURVE DRAWING
       ============================================================ */
    function drawCurve(canvas, pts, cursorPct) {
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, w, h);

        const maxRisk = Math.max(...pts);
        const lineColor = maxRisk > 0.5 ? "#f97316" : "#2563eb";
        const fillColor = maxRisk > 0.5 ? "rgba(249,115,22,0.1)" : "rgba(37,99,235,0.1)";

        ctx.beginPath();
        const step = w / (pts.length - 1);
        pts.forEach((val, i) => {
            const px = i * step;
            const py = h - 6 - (val * (h - 14));
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();

        if (cursorPct >= 0) {
            const cx = (cursorPct / 100) * w;
            ctx.save();
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
            ctx.restore();
        }
    }

    /* ============================================================
       TELEMETRY UPDATER (called by app.js dispatcher)
       ============================================================ */
    function updateTelemetry(msg) {
        if (testState === "running" && msg) {
            liveRisk = (msg.risk_state && msg.risk_state.state) ? msg.risk_state.state.toUpperCase() : liveRisk;
            liveHaptic = (msg.haptic && msg.haptic.direction) ? msg.haptic.direction.toUpperCase() : liveHaptic;
        }
    }

    return {
        init,
        renderScenarioMenu,
        openScenarioDetail,
        startTest,
        stopTest,
        openHistoryResult,
        updateTelemetry,
    };
})();
