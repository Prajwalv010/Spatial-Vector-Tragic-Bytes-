/**
 * SpatialVector-HMI — Subpage: Prediction Inspector (Advanced Diagnostics)
 *
 * Relocated wholesale from the dev dashboard:
 * - Trajectory vector canvas (User path, FOE, object predicted path, collision intersection)
 * - Track carousel selector (< Track #i >)
 * - Object metrics (TTC, CPA, Intersect, Relative Velocity, Pred Confidence, State)
 * - Rule-based reasoning breakdown for selected obstacle
 */

window.PredictionInspectorPage = (function () {
    "use strict";

    let containerEl = null;
    let canvasEl = null;
    let activeTracks = [];
    let selectedTrackIndex = 0;
    let lastMsg = null;

    function init(container) {
        containerEl = container;
        render();
        canvasEl = document.getElementById("insp-sub-canvas");
        drawCanvas();
    }

    function render() {
        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="App.closeSubpage()">‹ Settings</button>
                <span class="subpage-title">Prediction Inspector</span>
            </div>

            <!-- Track Selector Bar -->
            <div class="track-selector-bar">
                <button class="track-nav-btn" onclick="PredictionInspectorPage.prevTrack()">‹</button>
                <span id="insp-track-label" class="track-select-title">No Tracks Available</span>
                <button class="track-nav-btn" onclick="PredictionInspectorPage.nextTrack()">›</button>
            </div>

            <!-- Vector Trajectory Canvas -->
            <div class="video-card" style="margin-bottom: 12px; background: #0f172a; height: 180px;">
                <canvas id="insp-sub-canvas" class="hud-canvas" width="400" height="180"></canvas>
            </div>

            <!-- Primary Metrics Table -->
            <div class="expand-panel open" style="margin-bottom: 12px;">
                <div class="expand-header" style="cursor: default;">
                    <span>Kinematics & Collision Prediction</span>
                    <span id="insp-badge-state" class="pill-badge gray">STANDBY</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div class="kv-row">
                        <span class="kv-key">Track ID</span>
                        <span id="insp-val-id" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Class Category</span>
                        <span id="insp-val-class" class="kv-val">None</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Detection Confidence</span>
                        <span id="insp-val-trackconf" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Time to Collision (TTC)</span>
                        <span id="insp-val-ttc" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Closest Point of Approach (CPA)</span>
                        <span id="insp-val-cpa" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Path Intersection</span>
                        <span id="insp-badge-intersect" class="pill-badge gray">NO</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Relative Velocity</span>
                        <span id="insp-val-vel" class="kv-val mono">—</span>
                    </div>
                    <div class="kv-row">
                        <span class="kv-key">Prediction Confidence</span>
                        <span id="insp-val-predconf" class="kv-val mono">—</span>
                    </div>
                </div>
            </div>

            <!-- Risk Reasoning Factors -->
            <div class="expand-panel open">
                <div class="expand-header" style="cursor: default;">
                    <span>Active Risk Reason Codes</span>
                </div>
                <div class="expand-body" style="display: block;">
                    <div id="insp-reasoning-list" class="reasoning-list">
                        <span style="font-size: 11px; color: var(--text-muted);">No active risk factors detected.</span>
                    </div>
                </div>
            </div>
        `;
    }

    function prevTrack() {
        if (!activeTracks.length) return;
        selectedTrackIndex = (selectedTrackIndex - 1 + activeTracks.length) % activeTracks.length;
        updateUI();
        drawCanvas();
    }

    function nextTrack() {
        if (!activeTracks.length) return;
        selectedTrackIndex = (selectedTrackIndex + 1) % activeTracks.length;
        updateUI();
        drawCanvas();
    }

    function updateTelemetry(msg) {
        lastMsg = msg;
        activeTracks = (msg && Array.isArray(msg.tracks)) ? msg.tracks : [];
        if (selectedTrackIndex >= activeTracks.length) {
            selectedTrackIndex = 0;
        }
        updateUI();
        drawCanvas();
    }

    function updateUI() {
        const lblTrack = document.getElementById("insp-track-label");
        const valId = document.getElementById("insp-val-id");
        const valClass = document.getElementById("insp-val-class");
        const valTrackConf = document.getElementById("insp-val-trackconf");
        const valTtc = document.getElementById("insp-val-ttc");
        const valCpa = document.getElementById("insp-val-cpa");
        const badgeIntersect = document.getElementById("insp-badge-intersect");
        const valVel = document.getElementById("insp-val-vel");
        const valPredConf = document.getElementById("insp-val-predconf");
        const badgeState = document.getElementById("insp-badge-state");
        const listReasoning = document.getElementById("insp-reasoning-list");

        if (!activeTracks.length) {
            if (lblTrack) lblTrack.textContent = "No Tracks Detected";
            if (valId) valId.textContent = "—";
            if (valClass) valClass.textContent = "None";
            if (valTrackConf) valTrackConf.textContent = "—";
            if (valTtc) valTtc.textContent = "—";
            if (valCpa) valCpa.textContent = "—";
            if (badgeIntersect) {
                badgeIntersect.textContent = "NO";
                badgeIntersect.className = "pill-badge gray";
            }
            if (valVel) valVel.textContent = "—";
            if (valPredConf) valPredConf.textContent = "—";
            if (badgeState) {
                badgeState.textContent = (lastMsg && lastMsg.risk_state && lastMsg.risk_state.state) || "STANDBY";
                badgeState.className = "pill-badge gray";
            }
            if (listReasoning) {
                listReasoning.innerHTML = `<span style="font-size: 11px; color: var(--text-muted);">No active risk factors.</span>`;
            }
            return;
        }

        const track = activeTracks[selectedTrackIndex] || activeTracks[0];
        const state = (lastMsg && lastMsg.risk_state && lastMsg.risk_state.state) || "SAFE";

        if (lblTrack) {
            lblTrack.textContent = `Track #${track.track_id} (${track.class_name || 'Object'}) [${selectedTrackIndex + 1}/${activeTracks.length}]`;
        }
        if (valId) valId.textContent = `#${track.track_id}`;
        if (valClass) valClass.textContent = track.class_name || "Unknown";
        if (valTrackConf) valTrackConf.textContent = (track.track_confidence !== undefined && track.track_confidence !== null) ? Number(track.track_confidence).toFixed(2) : "—";
        if (valTtc) valTtc.textContent = (track.ttc_s !== undefined && track.ttc_s !== null) ? `${Number(track.ttc_s).toFixed(1)}s` : "None";
        if (valCpa) valCpa.textContent = (track.cpa !== undefined && track.cpa !== null) ? `${Number(track.cpa).toFixed(2)}m` : "--";

        if (badgeIntersect) {
            const isInt = Boolean(track.intersect);
            badgeIntersect.textContent = isInt ? "YES (Hazard)" : "NO";
            badgeIntersect.className = isInt ? "pill-badge red" : "pill-badge green";
        }

        if (valVel) {
            if (Array.isArray(track.relative_velocity) && track.relative_velocity.length >= 2) {
                const speed = Math.hypot(track.relative_velocity[0], track.relative_velocity[1]);
                valVel.textContent = `${speed.toFixed(1)} px/s`;
            } else if (typeof track.relative_velocity === 'number') {
                valVel.textContent = `${track.relative_velocity.toFixed(1)} px/s`;
            } else {
                valVel.textContent = "—";
            }
        }

        if (valPredConf) valPredConf.textContent = (track.pred_conf !== undefined && track.pred_conf !== null) ? Number(track.pred_conf).toFixed(2) : "—";

        if (badgeState) {
            badgeState.textContent = state;
            const sLower = state.toLowerCase();
            badgeState.className = `pill-badge ${sLower === 'safe' ? 'green' : (sLower === 'caution' ? 'amber' : (sLower === 'warning' ? 'orange' : (sLower === 'critical' ? 'red' : 'purple')))}`;
        }

        if (listReasoning && lastMsg && lastMsg.risk_state) {
            const reasons = lastMsg.risk_state.reason_codes || [];
            if (!reasons.length) {
                listReasoning.innerHTML = `<span style="font-size: 11px; color: var(--text-muted);">No active risk factors detected.</span>`;
            } else {
                listReasoning.innerHTML = reasons.map((r, i) => `
                    <div style="font-size: 11px; margin-bottom: 4px; display: flex; gap: 6px;">
                        <span style="color: var(--primary); font-weight: 700;">${i + 1}.</span>
                        <span>${ReasoningFormatter.format(r)}</span>
                    </div>
                `).join("");
            }
        }
    }

    function drawCanvas() {
        if (!canvasEl) return;
        const ctx = canvasEl.getContext("2d");
        const w = canvasEl.width;
        const h = canvasEl.height;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, w, h);

        // Perspective grid lines
        ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
        ctx.lineWidth = 1;
        for (let x = 20; x < w; x += 30) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = 20; y < h; y += 30) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        const userX = w * 0.5;
        const userY = h * 0.88;
        const foeX = w * 0.5;
        const foeY = h * 0.28;

        // User Path (Dashed Blue Line)
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(userX, userY);
        ctx.lineTo(foeX, foeY);
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.restore();

        // FOE marker
        ctx.save();
        ctx.beginPath();
        ctx.arc(foeX, foeY, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#3b82f6";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        // User Current Position marker
        ctx.save();
        ctx.beginPath();
        ctx.arc(userX, userY, 7, 0, Math.PI * 2);
        ctx.fillStyle = "#2563eb";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText("Current position", userX - 70, userY + 4);
        ctx.restore();

        const track = activeTracks[selectedTrackIndex] || activeTracks[0];
        if (track) {
            const hasIntersect = Boolean(track.intersect);
            const objX = w * 0.50;
            const objY = h * 0.22;
            const interX = w * 0.50;
            const interY = h * 0.55;

            // Object Predicted Path (Dashed Red or Green Line)
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(objX, objY);
            ctx.lineTo(interX, interY);
            ctx.strokeStyle = hasIntersect ? "#ef4444" : "#10b981";
            ctx.lineWidth = 2.5;
            ctx.setLineDash([5, 3]);
            ctx.stroke();
            ctx.restore();

            // Object graphic box representation
            ctx.save();
            ctx.fillStyle = hasIntersect ? "#ef4444" : "#10b981";
            ctx.beginPath();
            ctx.roundRect(objX - 22, objY - 12, 44, 24, 4);
            ctx.fill();
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 8px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText(String(track.class_name || "OBJ").toUpperCase().slice(0, 7), objX, objY + 4);
            ctx.restore();

            if (hasIntersect) {
                // Collision Intersection Point
                ctx.save();
                ctx.beginPath();
                ctx.arc(interX, interY, 6, 0, Math.PI * 2);
                ctx.fillStyle = "#ef4444";
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = "rgba(239, 68, 68, 0.95)";
                ctx.beginPath();
                ctx.roundRect(interX + 10, interY - 9, 74, 18, 4);
                ctx.fill();
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 9px 'JetBrains Mono', monospace";
                ctx.textAlign = "left";
                const ttcTxt = track.ttc_s ? `${Number(track.ttc_s).toFixed(1)}s hazard` : "Hazard point";
                ctx.fillText(ttcTxt, interX + 14, interY + 4);
                ctx.restore();
            }
        }

        // Legend overlay
        ctx.save();
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.beginPath();
        ctx.roundRect(w - 105, 8, 98, 64, 6);
        ctx.fill();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
        ctx.stroke();

        ctx.font = "8px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#94a3b8";

        // User path line
        ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2; ctx.setLineDash([4, 2]);
        ctx.beginPath(); ctx.moveTo(w - 95, 20); ctx.lineTo(w - 75, 20); ctx.stroke();
        ctx.fillText("User path", w - 70, 23);

        // Object path line
        ctx.strokeStyle = "#ef4444"; ctx.beginPath(); ctx.moveTo(w - 95, 34); ctx.lineTo(w - 75, 34); ctx.stroke();
        ctx.fillText("Object path", w - 70, 37);

        // Intersection dot
        ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(w - 85, 48, 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#94a3b8"; ctx.fillText("Intersection", w - 70, 51);

        ctx.restore();
    }

    return {
        init,
        prevTrack,
        nextTrack,
        updateTelemetry,
    };
})();
