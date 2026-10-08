/**
 * SpatialVector-HMI — Tab 1: Home Page (Live Navigation View)
 * Redesigned: Human-readable labels, clean info hierarchy, no raw session IDs.
 */

window.HomePage = (function () {
    "use strict";

    let vdoManager = null;
    let containerEl = null;
    const recentHistory = [];

    function init(container) {
        containerEl = container;
        renderShell();

        const iframe = document.getElementById("home-vdo-frame");
        const canvas = document.getElementById("home-hud-canvas");

        vdoManager = VdoNinjaEmbed.createManager({
            iframeEl: iframe,
            canvasEl: canvas,
            onSourceResolved: (url) => {
                console.log("[Home] Camera loaded:", url);
            }
        });
        vdoManager.init();
    }

    function renderShell() {
        containerEl.innerHTML = `
            <!-- Primary Risk Status Banner -->
            <div id="home-status-banner" class="status-alert-card safe">
                <div class="alert-headline-box">
                    <span id="home-alert-tag" class="alert-state-tag">SAFE</span>
                    <span id="home-alert-msg" class="alert-text-main">Path Clear · Safe to Navigate</span>
                </div>
                <div id="home-alert-dir" class="alert-dir-arrow">↑</div>
            </div>

            <!-- 3 Key Metrics -->
            <div class="metrics-triplet">
                <div class="metric-cell">
                    <div class="gauge-ring-wrap">
                        <svg width="44" height="44" viewBox="0 0 36 36">
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#f1f5f9" stroke-width="3.5" />
                            <path id="home-gauge-circle-stroke" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#10b981" stroke-dasharray="0, 100" stroke-linecap="round" stroke-width="3.5" />
                        </svg>
                        <span id="home-val-risk-score" class="gauge-number mono">0.00</span>
                    </div>
                    <span class="metric-tag">Risk</span>
                </div>
                <div class="metric-cell">
                    <span id="home-val-ttc" class="metric-big-num mono">—</span>
                    <span class="metric-tag">Time to Impact</span>
                </div>
                <div class="metric-cell">
                    <span id="home-val-cpa" class="metric-big-num mono">—</span>
                    <span class="metric-tag">Closest Approach</span>
                </div>
            </div>

            <!-- Live Camera View with AR Bounding Box HUD -->
            <div class="video-card">
                <div class="video-toolbar">
                    <button class="tool-chip" onclick="HomePage.toggleAspect()">Fill/Fit</button>
                    <button class="tool-chip" onclick="HomePage.toggleHud()">HUD</button>
                </div>
                <iframe id="home-vdo-frame" class="video-frame" allow="autoplay; camera; microphone" src="about:blank"></iframe>
                <canvas id="home-hud-canvas" class="hud-canvas" width="400" height="224"></canvas>
            </div>

            <!-- 3 Corridor Risk Boxes -->
            <div class="corridor-triad">
                <div id="home-box-corr-left" class="corridor-box safe">
                    <span class="corridor-name">Left</span>
                    <span id="home-txt-corr-left" class="corridor-score mono">0.00</span>
                    <span id="home-lbl-corr-left" class="corridor-status-tag">CLEAR</span>
                </div>
                <div id="home-box-corr-center" class="corridor-box safe">
                    <span class="corridor-name">Center</span>
                    <span id="home-txt-corr-center" class="corridor-score mono">0.00</span>
                    <span id="home-lbl-corr-center" class="corridor-status-tag">CLEAR</span>
                </div>
                <div id="home-box-corr-right" class="corridor-box safe">
                    <span class="corridor-name">Right</span>
                    <span id="home-txt-corr-right" class="corridor-score mono">0.00</span>
                    <span id="home-lbl-corr-right" class="corridor-status-tag">CLEAR</span>
                </div>
            </div>

            <!-- Directional Guidance Callout -->
            <div class="guidance-prompt-card">
                <span class="guidance-icon">🧭</span>
                <span id="home-safer-line" class="guidance-text">Forward corridor is clear to navigate.</span>
            </div>

            <!-- Quick Details Panel (Expandable) -->
            <div id="quick-details-container"></div>

            <!-- Nearby Objects Count -->
            <div class="expand-panel">
                <div class="expand-header" style="cursor: default;">
                    <span>Nearby Objects</span>
                    <span id="home-nearby-badge" class="pill-badge gray">0 Detected</span>
                </div>
            </div>

            <!-- Recent Guidance History (Last 3 haptic events) -->
            <div class="expand-panel open">
                <div class="expand-header" style="cursor: default;">
                    <span>Recent Guidance</span>
                    <span class="pill-badge blue">Live</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div id="home-recent-actions-list" style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 11px; color: var(--text-muted);">No guidance events yet — path is clear.</span>
                    </div>
                </div>
            </div>
        `;

        // Mount quick details inside container
        const qdContainer = document.getElementById("quick-details-container");
        QuickDetailsPage.init(qdContainer);
    }

    function updateTelemetry(msg) {
        if (!msg) return;

        const risk = msg.risk_state || {};
        const haptic = msg.haptic || {};
        const tracks = msg.tracks || [];
        const state = (risk.state || "SAFE").toUpperCase();
        const globalRisk = risk.global_risk || 0.0;
        const dir = (haptic.direction || "STOP").toUpperCase();

        // 1. Status Banner
        const banner = document.getElementById("home-status-banner");
        const tag = document.getElementById("home-alert-tag");
        const mainMsg = document.getElementById("home-alert-msg");
        const dirEl = document.getElementById("home-alert-dir");

        if (banner && tag && mainMsg && dirEl) {
            banner.className = `status-alert-card ${state.toLowerCase()}`;
            tag.textContent = state;

            let arrowChar = "↑";
            if (dir === "LEFT") arrowChar = "←";
            else if (dir === "RIGHT") arrowChar = "→";
            else if (dir === "STOP") arrowChar = "⏹";
            dirEl.textContent = arrowChar;

            // Human-readable situation description
            const primaryTrack = tracks[0];
            if (state === "CRITICAL") {
                mainMsg.textContent = "⚠ Imminent Collision — Stop Immediately!";
            } else if (state === "WARNING") {
                const cls = (primaryTrack && primaryTrack.class_name) || "Obstacle";
                mainMsg.textContent = `${cls} approaching your path — Steer ${dir !== "STOP" ? dir : "carefully"}`;
            } else if (state === "CAUTION") {
                const cls = (primaryTrack && primaryTrack.class_name) || "Object";
                mainMsg.textContent = `${cls} nearby — Stay alert`;
            } else if (state === "DEGRADED") {
                mainMsg.textContent = "Sensor confidence low — Proceeding cautiously";
            } else {
                mainMsg.textContent = "Path Clear · Safe to Navigate";
            }
        }

        // 2. Metrics Triplet (Risk, Time-to-Impact, Closest Approach)
        const scoreEl = document.getElementById("home-val-risk-score");
        const gaugeEl = document.getElementById("home-gauge-circle-stroke");
        const ttcEl = document.getElementById("home-val-ttc");
        const cpaEl = document.getElementById("home-val-cpa");

        if (scoreEl) scoreEl.textContent = Number(globalRisk).toFixed(2);
        if (gaugeEl) {
            const pct = Math.min(100, Math.max(0, Math.round(globalRisk * 100)));
            gaugeEl.setAttribute("stroke-dasharray", `${pct}, 100`);
            const strokeColor =
                state === "SAFE" ? "#10b981" :
                state === "CAUTION" ? "#f59e0b" :
                state === "WARNING" ? "#f97316" : "#ef4444";
            gaugeEl.setAttribute("stroke", strokeColor);
        }

        let minTtc = null;
        let minCpa = null;
        for (const t of tracks) {
            if (t.ttc_s !== null && t.ttc_s !== undefined) {
                if (minTtc === null || t.ttc_s < minTtc) minTtc = t.ttc_s;
            }
            if (t.cpa !== null && t.cpa !== undefined) {
                if (minCpa === null || t.cpa < minCpa) minCpa = t.cpa;
            }
        }
        if (ttcEl) ttcEl.textContent = minTtc !== null ? `${Number(minTtc).toFixed(1)}s` : "—";
        if (cpaEl) cpaEl.textContent = minCpa !== null ? `${Number(minCpa).toFixed(2)}m` : "—";

        // 3. Camera HUD Bounding Boxes
        if (vdoManager) {
            vdoManager.drawBoundingBoxes(tracks, msg.frame_width || 640, msg.frame_height || 480, null, globalRisk);
        }

        // 4. Corridor Risk Boxes
        const cr = risk.corridor_risks || {};
        updateCorridorBox("left", typeof cr.left === "number" ? cr.left : 0.0);
        updateCorridorBox("center", typeof cr.center === "number" ? cr.center : 0.0);
        updateCorridorBox("right", typeof cr.right === "number" ? cr.right : 0.0);

        // 5. Guidance Text
        const saferLine = document.getElementById("home-safer-line");
        if (saferLine) {
            if (dir === "LEFT") {
                saferLine.innerHTML = `<span class="guidance-highlight">Move Left</span> — safer path is to your left.`;
            } else if (dir === "RIGHT") {
                saferLine.innerHTML = `<span class="guidance-highlight">Move Right</span> — safer path is to your right.`;
            } else if (dir === "STOP") {
                saferLine.innerHTML = state === "SAFE"
                    ? `Forward corridor is clear to navigate.`
                    : `<span style="color: #ef4444; font-weight: 700;">⚠ Forward blocked — Stop and reassess.</span>`;
            } else {
                saferLine.textContent = `Center corridor is clear.`;
            }
        }

        // 6. Quick Details
        QuickDetailsPage.update(tracks[0], risk);

        // 7. Nearby Count
        const nearbyBadge = document.getElementById("home-nearby-badge");
        if (nearbyBadge) {
            const count = tracks.length;
            nearbyBadge.textContent = count === 1 ? "1 Object" : `${count} Objects`;
            nearbyBadge.className = count > 0 ? "pill-badge yellow" : "pill-badge gray";
        }

        // 8. Recent Guidance (only real haptic events, deduped)
        updateRecentGuidance(haptic, state);
    }

    function updateCorridorBox(side, score) {
        const box = document.getElementById(`home-box-corr-${side}`);
        const txt = document.getElementById(`home-txt-corr-${side}`);
        const lbl = document.getElementById(`home-lbl-corr-${side}`);
        if (!box || !txt || !lbl) return;

        txt.textContent = Number(score).toFixed(2);

        let cls = "safe";
        let tag = "CLEAR";
        if (score > 0.60) { cls = "risky"; tag = "BLOCKED"; }
        else if (score > 0.30) { cls = "caution"; tag = "CAUTION"; }

        box.className = `corridor-box ${cls}`;
        lbl.textContent = tag;
    }

    function updateRecentGuidance(haptic, state) {
        // Only record meaningful haptic events (skip ALL_CLEAR / DEGRADED_WARN noise)
        if (!haptic || !haptic.pattern_id) return;
        const pattern = haptic.pattern_id;
        if (pattern === "ALL_CLEAR" || pattern === "DEGRADED_WARN") return;

        const dir = (haptic.direction || "STOP").toUpperCase();
        const timeStr = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

        // Human-readable description
        let description = "Guidance issued";
        if (dir === "LEFT") description = "← Move Left — obstacle on right";
        else if (dir === "RIGHT") description = "→ Move Right — obstacle on left";
        else if (dir === "STOP") description = "⏹ Stop — path blocked";

        const item = { time: timeStr, dir, description, state };

        // Deduplicate consecutive identical events
        if (recentHistory.length > 0) {
            const last = recentHistory[0];
            if (last.dir === dir && last.state === state) return;
        }

        recentHistory.unshift(item);
        if (recentHistory.length > 3) recentHistory.pop();

        const listEl = document.getElementById("home-recent-actions-list");
        if (listEl) {
            listEl.innerHTML = recentHistory.map(h => `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed #e2e8f0;">
                    <span style="font-size: 12px; font-weight: 600; color: var(--text-primary);">${h.description}</span>
                    <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">${h.time}</span>
                </div>
            `).join("");
        }
    }

    return {
        init: init,
        updateTelemetry: updateTelemetry,
        toggleAspect: () => vdoManager && vdoManager.toggleAspect(),
        toggleHud: () => vdoManager && vdoManager.toggleHud(),
    };
})();
