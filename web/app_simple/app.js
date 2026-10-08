/**
 * SpatialVector-HMI — Simplified 3-Tab Alternate Dashboard Main Controller
 *
 * Coordinates:
 * - 3-Tab Navigation (Home, Haptics, Settings)
 * - Subpage Router (Social Assist, Prediction Inspector, Test Replay, System Health, Logs Export)
 * - WebSocket Telemetry Ingestion (/ws/telemetry) with auto-reconnection
 * - Staleness Watchdog (1500ms heartbeat monitor)
 * - Shared Telemetry Dispatcher across all loaded page modules
 */

/* ============================================================
   PROXIMITY BUZZER  — Web Audio API
   Generates real beep tones when an obstacle is too close.
   No external files needed.
   ============================================================ */
window.ProximityBuzzer = (function () {
    "use strict";

    let audioCtx = null;
    let enabled  = true;
    let buzzing  = false;          // true while critical-rate beep loop is running
    let warnedOnce = false;        // one-shot for caution beep
    let beepTimer = null;
    let lastTtc = Infinity;
    const CRITICAL_TTC = 2.0;      // seconds — urgent rapid beep
    const CAUTION_TTC  = 4.0;      // seconds — single warning beep

    function ensureCtx() {
        if (!audioCtx) {
            try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch(e) {}
        }
        // Browsers may suspend context until a user gesture
        if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
        return !!audioCtx;
    }

    /**
     * Play a single beep tone.
     * @param {number} freq   Frequency in Hz (440 = warning, 880 = critical)
     * @param {number} dur    Duration in seconds
     * @param {number} vol    Volume 0–1
     */
    function beep(freq, dur, vol) {
        if (!ensureCtx()) return;
        try {
            const osc   = audioCtx.createOscillator();
            const gain  = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.type      = 'sine';
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(vol, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + dur);
        } catch(e) {}
    }

    /** Start the rapid critical beep loop (rate depends on TTC). */
    function startCriticalLoop(ttc) {
        if (buzzing) return;
        buzzing = true;
        function loop() {
            if (!buzzing || !enabled) return;
            const currentTtc = lastTtc;
            if (currentTtc >= CRITICAL_TTC) { stopCriticalLoop(); return; }
            // The closer the object, the faster and higher the beep
            const urgency  = Math.max(0, 1 - currentTtc / CRITICAL_TTC); // 0–1
            const freq     = 660 + urgency * 660;                          // 660–1320 Hz
            const interval = 600 - urgency * 480;                          // 600–120 ms
            beep(freq, 0.12, 0.7);
            beepTimer = setTimeout(loop, interval);
        }
        loop();
    }

    function stopCriticalLoop() {
        buzzing = false;
        if (beepTimer) { clearTimeout(beepTimer); beepTimer = null; }
    }

    /**
     * Call this every time new telemetry arrives.
     * @param {number} ttc  Time-to-collision in seconds (Infinity if no threat)
     */
    function onTelemetry(ttc) {
        if (!enabled) return;
        lastTtc = (ttc === null || ttc === undefined) ? Infinity : ttc;

        if (lastTtc < CRITICAL_TTC) {
            warnedOnce = false;          // reset caution flag when critical
            startCriticalLoop(lastTtc);
        } else {
            stopCriticalLoop();
            if (lastTtc < CAUTION_TTC && !warnedOnce) {
                warnedOnce = true;
                beep(440, 0.18, 0.5);    // single mid-tone caution beep
            } else if (lastTtc >= CAUTION_TTC) {
                warnedOnce = false;      // reset when clear
            }
        }
    }

    /** Manual test — play three increasing beeps. */
    function testBeep() {
        ensureCtx();
        beep(440, 0.15, 0.6);
        setTimeout(() => beep(660, 0.15, 0.7), 250);
        setTimeout(() => beep(880, 0.2,  0.8), 500);
    }

    function setEnabled(val) {
        enabled = val;
        if (!val) stopCriticalLoop();
    }
    function isEnabled() { return enabled; }

    return { onTelemetry, testBeep, setEnabled, isEnabled };
})();


window.App = (function () {
    "use strict";

    let activeTab = "home";
    let activeSubpage = null;
    let ws = null;
    let reconnectDelay = 1000;
    let watchdog = null;
    let lastTelemetry = null;

    function init() {
        console.log("[SpatialVector] Initializing 3-Tab Simplified Dashboard...");

        // 1. Initialize Page Modules
        const viewHome = document.getElementById("view-home");
        const viewHaptics = document.getElementById("view-haptics");
        const viewSettings = document.getElementById("view-settings");
        const viewSocial = document.getElementById("view-social-assist");
        const viewInsp = document.getElementById("view-adv-inspector");
        const viewReplay = document.getElementById("view-adv-replay");
        const viewHealth = document.getElementById("view-adv-health");
        const viewExport = document.getElementById("view-adv-export");

        if (window.HomePage && viewHome) HomePage.init(viewHome);
        if (window.HapticsPage && viewHaptics) HapticsPage.init(viewHaptics);
        if (window.SettingsPage && viewSettings) SettingsPage.init(viewSettings);
        if (window.SocialAssistPage && viewSocial) SocialAssistPage.init(viewSocial);
        if (window.PredictionInspectorPage && viewInsp) PredictionInspectorPage.init(viewInsp);
        if (window.TestReplayPage && viewReplay) TestReplayPage.init(viewReplay);
        if (window.SystemHealthPage && viewHealth) SystemHealthPage.init(viewHealth);
        if (window.LogsExportPage && viewExport) LogsExportPage.init(viewExport);

        // 2. Setup Watchdog
        const bannerEl = document.getElementById("staleness-banner");
        const connPill = document.getElementById("conn-status-badge");
        const connTxt = document.getElementById("conn-status-txt");

        if (window.StalenessWatchdog) {
            watchdog = StalenessWatchdog.createWatchdog({
                thresholdMs: 1500,
                onStaleChange: (isStale, durationMs) => {
                    if (isStale) {
                        if (bannerEl) {
                            bannerEl.style.display = "block";
                            bannerEl.textContent = durationMs >= 1000000000
                                ? "⚠️ PIPELINE OFFLINE — Waiting for telemetry connection..."
                                : `⚠️ PIPELINE OFFLINE — No telemetry for ${(durationMs / 1000).toFixed(1)}s`;
                        }
                        if (connPill) {
                            connPill.className = "conn-pill disconnected";
                        }
                        if (connTxt) connTxt.textContent = "Offline";
                    } else {
                        if (bannerEl) bannerEl.style.display = "none";
                        if (connPill) {
                            connPill.className = "conn-pill connected";
                        }
                        if (connTxt) connTxt.textContent = "Online";
                    }
                }
            });
            watchdog.start();
        }

        // 3. Connect WebSocket Telemetry
        connectWs();

        // 4. Start Clock
        startClock();

        // 5. Default Tab
        switchTab("home");
    }

    function switchTab(tab) {
        activeTab = tab;
        activeSubpage = null;

        // Hide all subpages
        document.querySelectorAll(".page-view.subpage").forEach(el => {
            el.classList.remove("active");
            el.style.display = "none";
        });

        // Hide all main page views
        document.querySelectorAll(".page-view:not(.subpage)").forEach(el => {
            el.classList.remove("active");
            el.style.display = "none";
        });

        // Show target view
        const targetView = document.getElementById(`view-${tab}`);
        if (targetView) {
            targetView.style.display = "block";
            targetView.classList.add("active");
        }

        // Update Bottom Nav
        document.querySelectorAll(".nav-item").forEach(btn => {
            if (btn.getAttribute("data-tab") === tab) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        // Scroll to top
        const mainScroll = document.getElementById("main-content-scroll");
        if (mainScroll) mainScroll.scrollTop = 0;
    }

    function openSubpage(subpageId) {
        activeSubpage = subpageId;

        // Hide main settings view
        const viewSettings = document.getElementById("view-settings");
        if (viewSettings) {
            viewSettings.classList.remove("active");
            viewSettings.style.display = "none";
        }

        // Hide any other subpages
        document.querySelectorAll(".page-view.subpage").forEach(el => {
            el.classList.remove("active");
            el.style.display = "none";
        });

        // Show target subpage
        const sub = document.getElementById(subpageId);
        if (sub) {
            sub.style.display = "block";
            sub.classList.add("active");
        }

        // Scroll to top
        const mainScroll = document.getElementById("main-content-scroll");
        if (mainScroll) mainScroll.scrollTop = 0;
    }

    function closeSubpage() {
        activeSubpage = null;

        // Hide all subpages
        document.querySelectorAll(".page-view.subpage").forEach(el => {
            el.classList.remove("active");
            el.style.display = "none";
        });

        // Return to settings tab
        const viewSettings = document.getElementById("view-settings");
        if (viewSettings) {
            viewSettings.style.display = "block";
            viewSettings.classList.add("active");
        }

        const mainScroll = document.getElementById("main-content-scroll");
        if (mainScroll) mainScroll.scrollTop = 0;
    }

    function connectWs() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

        try {
            ws = new WebSocket(wsUrl);

            ws.onopen = function () {
                console.log("[WS] Connected to telemetry backend:", wsUrl);
                reconnectDelay = 1000;
                if (watchdog) watchdog.onMessageReceived();
            };

            ws.onmessage = function (event) {
                try {
                    const msg = JSON.parse(event.data);
                    lastTelemetry = msg;
                    if (watchdog) watchdog.onMessageReceived();
                    dispatchTelemetry(msg);
                } catch (e) {
                    console.warn("[WS] Error parsing telemetry packet:", e);
                }
            };

            ws.onclose = function () {
                console.warn(`[WS] Disconnected. Reconnecting in ${reconnectDelay}ms...`);
                setTimeout(connectWs, reconnectDelay);
                reconnectDelay = Math.min(reconnectDelay * 1.5, 5000);
            };

            ws.onerror = function (err) {
                console.error("[WS] WebSocket error:", err);
                ws.close();
            };
        } catch (err) {
            console.error("[WS] Initialization exception:", err);
            setTimeout(connectWs, 2000);
        }
    }

    function dispatchTelemetry(msg) {
        // ── Proximity Buzzer ──────────────────────────────────────────────
        if (window.ProximityBuzzer) {
            // Accept TTC from various telemetry shapes
            const ttc = (msg.ttc !== undefined)    ? msg.ttc
                      : (msg.threat && msg.threat.ttc !== undefined) ? msg.threat.ttc
                      : (msg.risk  && msg.risk.ttc  !== undefined)   ? msg.risk.ttc
                      : Infinity;
            ProximityBuzzer.onTelemetry(ttc);
        }

        // ── Page modules ─────────────────────────────────────────────────
        if (window.HomePage && typeof HomePage.updateTelemetry === "function") {
            HomePage.updateTelemetry(msg);
        }
        if (window.HapticsPage && typeof HapticsPage.updateTelemetry === "function") {
            HapticsPage.updateTelemetry(msg);
        }
        if (window.PredictionInspectorPage && typeof PredictionInspectorPage.updateTelemetry === "function") {
            PredictionInspectorPage.updateTelemetry(msg);
        }
        if (window.SystemHealthPage && typeof SystemHealthPage.updateTelemetry === "function") {
            SystemHealthPage.updateTelemetry(msg);
        }
        if (window.TestReplayPage && typeof TestReplayPage.updateTelemetry === "function") {
            TestReplayPage.updateTelemetry(msg);
        }
    }

    function startClock() {
        const clockEl = document.getElementById("header-clock");
        function tick() {
            if (clockEl) {
                const now = new Date();
                clockEl.textContent = now.toTimeString().split(" ")[0];
            }
        }
        tick();
        setInterval(tick, 1000);
    }

    return {
        init,
        switchTab,
        openSubpage,
        closeSubpage,
        getLastTelemetry: () => lastTelemetry,
    };
})();

// Bootstrap when DOM ready
document.addEventListener("DOMContentLoaded", function () {
    window.App.init();
});
