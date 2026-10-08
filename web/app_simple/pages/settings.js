/**
 * SpatialVector-HMI — Tab 3: Settings Page
 * Redesigned: Removed VDO.Ninja button, clean section headers, functional API settings,
 * session info replaced with human-readable system status.
 */

window.SettingsPage = (function () {
    "use strict";

    let containerEl = null;
    let liveConfig = {
        weight_ttc: 0.50,
        weight_miss_distance: 0.30,
        weight_intersection_confidence: 0.20,
        state_thresholds: { caution: 0.30, warning: 0.60, critical: 0.85 },
        hysteresis_frames_up: 3,
        hysteresis_frames_down: 5,
        degraded_confidence_threshold: 0.25,
        horizon_s: 5.0,
    };

    function init(container) {
        containerEl = container;
        render();
        fetchLiveThresholds();
    }

    function render() {
        containerEl.innerHTML = `
            <div class="settings-menu-list">

                <!-- Page Title -->
                <div style="margin-bottom: 4px;">
                    <div style="font-size: 18px; font-weight: 800; color: var(--text-primary);">Settings</div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">System configuration & preferences</div>
                </div>

                <!-- 1. Social Assist -->
                <div class="settings-menu-item" onclick="App.openSubpage('view-social-assist')">
                    <div class="menu-item-left">
                        <span class="menu-item-title">👥  Social Assist</span>
                        <span class="menu-item-desc">Known contacts & optional recognition</span>
                    </div>
                    <span class="expand-chevron">›</span>
                </div>

                <!-- 2. Safety Preferences (Live Sliders — fully functional) -->
                <div class="expand-panel open" id="panel-safety-prefs">
                    <button class="expand-header" onclick="SettingsPage.togglePanel('panel-safety-prefs')">
                        <div class="menu-item-left" style="text-align: left;">
                            <span class="menu-item-title">⚙️  Safety Thresholds</span>
                            <span class="menu-item-desc">Adjust collision warning sensitivity</span>
                        </div>
                        <span class="expand-chevron">▼</span>
                    </button>
                    <div class="expand-body" style="display: block;">
                        <!-- Warning Threshold Slider -->
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Warning Sensitivity</span>
                                <span id="val-pref-warning" style="font-family: var(--font-mono); font-weight: 700; color: #f97316;">0.60</span>
                            </div>
                            <p style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px;">Lower = warns earlier. Higher = warns only at closer risk.</p>
                            <input id="range-pref-warning" type="range" class="slider-ctrl" min="0.40" max="0.80" step="0.05" value="0.60" oninput="SettingsPage.onSliderChange()">
                        </div>

                        <!-- Critical Threshold Slider -->
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Critical Stop Sensitivity</span>
                                <span id="val-pref-critical" style="font-family: var(--font-mono); font-weight: 700; color: #ef4444;">0.85</span>
                            </div>
                            <p style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px;">Risk score above this triggers emergency stop signal.</p>
                            <input id="range-pref-critical" type="range" class="slider-ctrl" min="0.75" max="0.95" step="0.05" value="0.85" oninput="SettingsPage.onSliderChange()">
                        </div>

                        <!-- TTC Weight Slider -->
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Time Sensitivity Weight</span>
                                <span id="val-pref-w-ttc" style="font-family: var(--font-mono); font-weight: 700;">0.50</span>
                            </div>
                            <p style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px;">How much weight the system places on time-to-collision vs. distance.</p>
                            <input id="range-pref-w-ttc" type="range" class="slider-ctrl" min="0.30" max="0.70" step="0.05" value="0.50" oninput="SettingsPage.onSliderChange()">
                        </div>

                        <!-- Action Buttons -->
                        <div style="display: flex; gap: 8px; margin-top: 10px;">
                            <button class="action-btn" onclick="SettingsPage.applyThresholds(false)">Apply Now</button>
                            <button class="action-btn secondary" onclick="SettingsPage.applyThresholds(true)">Save as Default</button>
                        </div>
                        <span id="settings-status-msg" style="font-size: 11px; color: var(--brand-primary); display: block; margin-top: 6px;"></span>
                    </div>
                </div>

                <!-- 3. Privacy Notice -->
                <div class="expand-panel" id="panel-privacy">
                    <button class="expand-header" onclick="SettingsPage.togglePanel('panel-privacy')">
                        <div class="menu-item-left" style="text-align: left;">
                            <span class="menu-item-title">🔒  Privacy & Data</span>
                            <span class="menu-item-desc">On-device processing, no cloud sync</span>
                        </div>
                        <span class="expand-chevron">▼</span>
                    </button>
                    <div class="expand-body">
                        <div class="kv-row">
                            <div>
                                <span class="kv-key" style="font-weight: 600;">Edge-Only Processing</span>
                                <div style="font-size: 10px; color: var(--text-muted);">All inference runs on your local device</div>
                            </div>
                            <span class="pill-badge green">ENFORCED</span>
                        </div>
                        <div class="kv-row">
                            <div>
                                <span class="kv-key" style="font-weight: 600;">Cloud Sync</span>
                                <div style="font-size: 10px; color: var(--text-muted);">No telemetry or images leave device</div>
                            </div>
                            <span class="pill-badge gray">OFF</span>
                        </div>
                        <div class="kv-row">
                            <div>
                                <span class="kv-key" style="font-weight: 600;">Social Assist Sandbox</span>
                                <div style="font-size: 10px; color: var(--text-muted);">Optional — has zero effect on safety logic</div>
                            </div>
                            <span class="pill-badge green">ISOLATED</span>
                        </div>
                    </div>
                </div>

                <!-- 4. Advanced Diagnostics -->
                <div class="expand-panel" id="panel-advanced">
                    <button class="expand-header" onclick="SettingsPage.togglePanel('panel-advanced')">
                        <div class="menu-item-left" style="text-align: left;">
                            <span class="menu-item-title">🔬  Advanced Diagnostics</span>
                            <span class="menu-item-desc">Prediction inspector, test scenarios, system health</span>
                        </div>
                        <span class="expand-chevron">▼</span>
                    </button>
                    <div class="expand-body">
                        <div class="settings-menu-list">
                            <div class="settings-menu-item" style="padding: 10px 12px;" onclick="App.openSubpage('view-adv-inspector')">
                                <div>
                                    <span style="font-weight: 600; font-size: 12px;">Prediction Inspector</span>
                                    <div style="font-size: 10px; color: var(--text-muted);">Deep view of obstacle trajectory analysis</div>
                                </div>
                                <span class="expand-chevron">›</span>
                            </div>
                            <div class="settings-menu-item" style="padding: 10px 12px;" onclick="App.openSubpage('view-adv-replay')">
                                <div>
                                    <span style="font-weight: 600; font-size: 12px;">Test Scenarios</span>
                                    <div style="font-size: 10px; color: var(--text-muted);">Run S1–S6 benchmark safety scenarios</div>
                                </div>
                                <span class="expand-chevron">›</span>
                            </div>
                            <div class="settings-menu-item" style="padding: 10px 12px;" onclick="App.openSubpage('view-adv-health')">
                                <div>
                                    <span style="font-weight: 600; font-size: 12px;">System Health</span>
                                    <div style="font-size: 10px; color: var(--text-muted);">Camera, Arduino & pipeline status</div>
                                </div>
                                <span class="expand-chevron">›</span>
                            </div>
                            <div class="settings-menu-item" style="padding: 10px 12px;" onclick="App.openSubpage('view-adv-export')">
                                <div>
                                    <span style="font-weight: 600; font-size: 12px;">Session Logs & Export</span>
                                    <div style="font-size: 10px; color: var(--text-muted);">Download event logs for review</div>
                                </div>
                                <span class="expand-chevron">›</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 5. About -->
                <div class="expand-panel" id="panel-about">
                    <button class="expand-header" onclick="SettingsPage.togglePanel('panel-about')">
                        <div class="menu-item-left" style="text-align: left;">
                            <span class="menu-item-title">ℹ️  About This Device</span>
                            <span class="menu-item-desc">Software version & hardware info</span>
                        </div>
                        <span class="expand-chevron">▼</span>
                    </button>
                    <div class="expand-body">
                        <div class="kv-row">
                            <span class="kv-key">System</span>
                            <span class="kv-val">SpatialVector-HMI</span>
                        </div>
                        <div class="kv-row">
                            <span class="kv-key">Version</span>
                            <span class="kv-val">v2.1 (Prototype)</span>
                        </div>
                        <div class="kv-row">
                            <span class="kv-key">Controller</span>
                            <span class="kv-val">Arduino Uno Q</span>
                        </div>
                        <div class="kv-row">
                            <span class="kv-key">Haptic Motors</span>
                            <span class="kv-val">2 (Left + Right)</span>
                        </div>
                        <div class="kv-row">
                            <span class="kv-key">Camera</span>
                            <span class="kv-val">USB Webcam (Port 2)</span>
                        </div>
                        <div class="kv-row">
                            <span class="kv-key">Pipeline</span>
                            <span class="kv-val">YOLOv8 + ByteTrack</span>
                        </div>
                    </div>
                </div>

            </div>
        `;
    }

    function togglePanel(id) {
        const p = document.getElementById(id);
        if (p) p.classList.toggle("open");
    }

    async function fetchLiveThresholds() {
        try {
            const res = await fetch("/api/settings/risk-thresholds");
            if (res.ok) {
                const data = await res.json();
                if (data.config) {
                    liveConfig = data.config;
                    syncSlidersFromConfig();
                }
            }
        } catch (e) {
            console.warn("[Settings] Could not fetch thresholds:", e);
        }
    }

    function syncSlidersFromConfig() {
        const warn = (liveConfig.state_thresholds && liveConfig.state_thresholds.warning) || 0.60;
        const crit = (liveConfig.state_thresholds && liveConfig.state_thresholds.critical) || 0.85;
        const wTtc = liveConfig.weight_ttc || 0.50;

        const rangeWarn = document.getElementById("range-pref-warning");
        const rangeCrit = document.getElementById("range-pref-critical");
        const rangeTtc = document.getElementById("range-pref-w-ttc");

        if (rangeWarn) rangeWarn.value = warn;
        if (rangeCrit) rangeCrit.value = crit;
        if (rangeTtc) rangeTtc.value = wTtc;

        onSliderChange();
    }

    function onSliderChange() {
        const rangeWarn = document.getElementById("range-pref-warning");
        const rangeCrit = document.getElementById("range-pref-critical");
        const rangeTtc = document.getElementById("range-pref-w-ttc");
        const valWarn = document.getElementById("val-pref-warning");
        const valCrit = document.getElementById("val-pref-critical");
        const valTtc = document.getElementById("val-pref-w-ttc");

        if (rangeWarn && valWarn) valWarn.textContent = Number(rangeWarn.value).toFixed(2);
        if (rangeCrit && valCrit) valCrit.textContent = Number(rangeCrit.value).toFixed(2);
        if (rangeTtc && valTtc) valTtc.textContent = Number(rangeTtc.value).toFixed(2);
    }

    async function applyThresholds(persist = false) {
        const rangeWarn = document.getElementById("range-pref-warning");
        const rangeCrit = document.getElementById("range-pref-critical");
        const rangeTtc = document.getElementById("range-pref-w-ttc");
        const msgEl = document.getElementById("settings-status-msg");

        const warn = parseFloat(rangeWarn.value);
        const crit = parseFloat(rangeCrit.value);
        const wTtc = parseFloat(rangeTtc.value);

        const remaining = Math.max(0.1, 1.0 - wTtc);
        const wMiss = parseFloat((remaining * 0.6).toFixed(2));
        const wInt = parseFloat((remaining * 0.4).toFixed(2));

        const payload = {
            weight_ttc: wTtc,
            weight_miss_distance: wMiss,
            weight_intersection_confidence: wInt,
            state_thresholds: { caution: 0.30, warning: warn, critical: crit },
            hysteresis_frames_up: liveConfig.hysteresis_frames_up || 3,
            hysteresis_frames_down: liveConfig.hysteresis_frames_down || 5,
            degraded_confidence_threshold: liveConfig.degraded_confidence_threshold || 0.25,
            horizon_s: liveConfig.horizon_s || 5.0,
            persist: persist,
        };

        try {
            if (msgEl) msgEl.textContent = "Applying...";
            const res = await fetch("/api/settings/risk-thresholds", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (res.ok) {
                const data = await res.json();
                liveConfig = data.config;
                if (msgEl) {
                    msgEl.style.color = "#10b981";
                    msgEl.textContent = persist
                        ? "✓ Saved as default!"
                        : "✓ Applied to live session!";
                    setTimeout(() => { if (msgEl) msgEl.textContent = ""; }, 3000);
                }
            } else {
                const err = await res.json();
                if (msgEl) {
                    msgEl.style.color = "#ef4444";
                    msgEl.textContent = `Error: ${err.detail || "Validation failed"}`;
                }
            }
        } catch (e) {
            if (msgEl) {
                msgEl.style.color = "#ef4444";
                msgEl.textContent = `Network error: ${e}`;
            }
        }
    }

    return {
        init: init,
        togglePanel: togglePanel,
        onSliderChange: onSliderChange,
        applyThresholds: applyThresholds,
    };
})();
