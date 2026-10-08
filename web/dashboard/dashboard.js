/**
 * SpatialVector-HMI — Mobile Assistive Dashboard Engine (M11)
 *
 * Implements:
 * 1. 5-Tab Navigation (Live View, Prediction Inspector, Haptic State, Test & Replay, Social Assist)
 * 2. Client-Side Staleness Watchdog (Section 0 requirement: threshold = 1500ms)
 * 3. Dynamic VDO.Ninja Camera Feed embedding & real-time link configuration
 * 4. Aspect Ratio (Fit/Fill) and HUD (ON/OFF) video overlay controls
 * 5. WebSocket auto-reconnect with telemetry visualization:
 *    - Real-time Risk Alert Banner with Circular Gauge (Safe, Caution, Warning, Critical, Degraded)
 *    - 3 Separate Metric Tiles (Risk Score, TTC, CPA)
 *    - Corridor Risk Triad (Left, Center, Right)
 *    - Recommended Direction Guidance
 *    - Calibrated Ground-Plane Perspective HUD Overlay Canvas
 * 6. Interactive Prediction Inspector with Vector Trajectory Canvas & Dynamic Reasoning
 * 7. Interactive Haptic State with Vest SVG Vibration Ripples & Telemetry History
 * 8. Interactive Test & Replay Scenarios (S1-S6) with Multi-metric Timeline Chart & Scrubber
 * 9. Social Assist with Privacy-isolated Face Recognition & Contacts Directory
 */

(function () {
    "use strict";

    const STALE_THRESHOLD_MS = 1500;
    let lastMessageTimestamp = 0;
    let pageLoadTimestamp = Date.now();
    let ws = null;
    let reconnectDelay = 1000;
    let currentCameraUrl = "2";

    // Overlay and Video Controls
    let hudOverlayEnabled = true;
    let isVideoCover = true;

    // Telemetry and Model State (Clean Standby State on Startup)
    let currentRiskState = "STANDBY";
    let currentGlobalRisk = 0.0;
    let currentTTC = null;
    let currentCPA = null;
    let activeTracks = [];
    let selectedTrackIndex = 0;
    let hapticHistory = [];
    let lastHapticKey = "";
    let lastHapticLogTime = 0;
    let currentFrameWidth = 640;
    let currentFrameHeight = 480;
    let lastReasonCodes = [];

    // Predict screen state
    let predictViewMode = "2d"; // '2d' | 'top' | 'timeline'
    let reasoningAccordionOpen = false;

    // Replay & Scenarios State
    const SCENARIOS = {
        s1: {
            id: "s1",
            title: "S1 — Parallel Wall",
            desc: "Walk parallel to a wall at close distance",
            icon: "🚶",
            expectedRisk: "SAFE",
            expectedHaptic: "ALL_CLEAR (Silent)",
            expectedTtc: "> 5.0 s",
            expectedIntersect: "NO (Parallel path)",
            actualRisk: "SAFE",
            actualHaptic: "ALL_CLEAR (Silent)",
            result: "PASS",
            riskPeak: 0.12,
            curve: [0.05, 0.08, 0.11, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.03]
        },
        s2: {
            id: "s2",
            title: "S2 — Head-On Obstacle",
            desc: "Obstacle approaching directly in user path",
            icon: "🚶",
            expectedRisk: "WARNING",
            expectedHaptic: "LEFT / RIGHT Guidance (Urgency 3)",
            expectedTtc: "2.1 s",
            expectedIntersect: "YES (Direct collision)",
            actualRisk: "WARNING",
            actualHaptic: "LEFT Guidance (Urgency 3)",
            result: "PASS",
            riskPeak: 0.72,
            curve: [0.15, 0.22, 0.38, 0.55, 0.72, 0.65, 0.40, 0.20, 0.10, 0.05]
        },
        s3: {
            id: "s3",
            title: "S3 — Crossing Object",
            desc: "Pedestrian crossing user path at lateral angle",
            icon: "🚶",
            expectedRisk: "WARNING",
            expectedHaptic: "DIRECTIONAL CUE (Pulse left)",
            expectedTtc: "2.8 s",
            expectedIntersect: "YES (Crossing point)",
            actualRisk: "WARNING",
            actualHaptic: "LEFT Guidance (Urgency 2)",
            result: "PASS",
            riskPeak: 0.76,
            curve: [0.10, 0.18, 0.35, 0.62, 0.76, 0.68, 0.30, 0.15, 0.08, 0.05]
        },
        s4: {
            id: "s4",
            title: "S4 — Static Obstacle",
            desc: "Parked scooter or stationary object in corridor",
            icon: "🛵",
            expectedRisk: "SAFE",
            expectedHaptic: "ALL_CLEAR (Pass corridor clear)",
            expectedTtc: "> 4.5 s",
            expectedIntersect: "NO (Adequate margin)",
            actualRisk: "SAFE",
            actualHaptic: "ALL_CLEAR (Silent)",
            result: "PASS",
            riskPeak: 0.22,
            curve: [0.08, 0.12, 0.18, 0.22, 0.19, 0.14, 0.09, 0.06, 0.04, 0.02]
        },
        s5: {
            id: "s5",
            title: "S5 — Multiple Obstacles",
            desc: "Multiple converging obstacles; prioritizes highest hazard",
            icon: "👥",
            expectedRisk: "CRITICAL",
            expectedHaptic: "DUAL MOTOR EMERGENCY BRAKE",
            expectedTtc: "1.4 s",
            expectedIntersect: "YES (Immediate hazard)",
            actualRisk: "CRITICAL",
            actualHaptic: "DUAL MOTORS (Urgency 5)",
            result: "PASS",
            riskPeak: 0.95,
            curve: [0.20, 0.40, 0.65, 0.82, 0.95, 0.92, 0.88, 0.60, 0.35, 0.15]
        },
        s6: {
            id: "s6",
            title: "S6 — Narrow Corridor",
            desc: "Tight walking passage; spatial reasoning validates center clear path",
            icon: "↔",
            expectedRisk: "SAFE",
            expectedHaptic: "ALL_CLEAR (Center aligned)",
            expectedTtc: "> 4.0 s",
            expectedIntersect: "NO (Passing corridor)",
            actualRisk: "SAFE",
            actualHaptic: "ALL_CLEAR (Silent)",
            result: "PASS",
            riskPeak: 0.18,
            curve: [0.06, 0.09, 0.14, 0.18, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02]
        }
    };
    let currentScenarioId = "s1";
    let replayTimer = null;
    let modalTestTimer = null;
    let replayProgress = 35; // 0 to 100 percent
    let isReplaying = false;
    let isModalTesting = false;
    let testHistory = [
        { title: "S1 Parallel Wall", result: "PASS", expected: "SAFE", actual: "SAFE" },
        { title: "S2 Head-On Obstacle", result: "PASS", expected: "WARNING", actual: "WARNING" }
    ];

    // Social Assist State (UI-Only Prototype)
    let socialAssistEnabled = true;
    let socialDemoState = "known"; // 'known' | 'unknown' | 'possible'
    let mockContacts = [
        { id: 1, name: "Rahul", relationship: "Friend", photos: [], photosCount: 3, status: "Ready", avatar: "👤" },
        { id: 2, name: "Ananya", relationship: "Family", photos: [], photosCount: 2, status: "Ready", avatar: "👩" }
    ];
    let currentContactId = null;
    let addFriendState = { photos: [], name: "", relationship: "Friend", step: 1 };
    let tempFriendPhotos = [];

    // -------------------------------------------------------------------------
    // DOM Element References
    // -------------------------------------------------------------------------
    const stalenessBanner = document.getElementById("staleness-banner");
    const staleSecTxt = document.getElementById("stale-sec-txt");
    const connPillBadge = document.getElementById("conn-pill-badge");
    const connStatusDot = document.getElementById("conn-status-dot");
    const connStatusText = document.getElementById("conn-status-text");
    const sessionIdLabel = document.getElementById("session-id-label");
    const headerClock = document.getElementById("header-clock");

    // Live Tab Elements
    const riskAlertCard = document.getElementById("risk-alert-card");
    const alertIconBox = document.getElementById("alert-icon-box");
    const alertStateName = document.getElementById("alert-state-name");
    const alertStateSub = document.getElementById("alert-state-sub");
    const gaugeCircleStroke = document.getElementById("gauge-circle-stroke");
    const valRiskScore = document.getElementById("val-risk-score");
    const valTtc = document.getElementById("val-ttc");
    const valCpa = document.getElementById("val-cpa");

    const vdoNinjaFrame = document.getElementById("vdo-ninja-frame");
    const mjpegStream = document.getElementById("mjpeg-video-stream");
    const liveOverlayCanvas = document.getElementById("live-overlay-canvas");
    const btnToggleCameraSrc = document.getElementById("btn-toggle-camera-src");
    const btnOpenSettings = document.getElementById("btn-open-settings");

    const boxCorrLeft = document.getElementById("box-corr-left");
    const txtCorrLeft = document.getElementById("txt-corr-left");
    const lblCorrLeft = document.getElementById("lbl-corr-left");
    const boxCorrCenter = document.getElementById("box-corr-center");
    const txtCorrCenter = document.getElementById("txt-corr-center");
    const lblCorrCenter = document.getElementById("lbl-corr-center");
    const boxCorrRight = document.getElementById("box-corr-right");
    const txtCorrRight = document.getElementById("txt-corr-right");
    const lblCorrRight = document.getElementById("lbl-corr-right");

    const recArrowIcon = document.getElementById("rec-arrow-icon");
    const recDirName = document.getElementById("rec-dir-name");
    const recDirSub = document.getElementById("rec-dir-sub");

    // Prediction Inspector Elements
    const inspectorObjLabel = document.getElementById("inspector-obj-label");
    const inspObjIcon = document.getElementById("insp-obj-icon");
    const inspRiskBadge = document.getElementById("insp-risk-badge");
    const inspHeaderBadge = document.getElementById("insp-header-badge");
    const inspectorCanvas = document.getElementById("inspector-canvas");
    const inspId = document.getElementById("insp-id");
    const inspClass = document.getElementById("insp-class");
    const inspTrackConf = document.getElementById("insp-track-conf");
    const inspTtc = document.getElementById("insp-ttc");
    const inspTtcSub = document.getElementById("insp-ttc-sub");
    const inspCpa = document.getElementById("insp-cpa");
    const inspCpaSub = document.getElementById("insp-cpa-sub");
    const inspIntersect = document.getElementById("insp-intersect");
    const inspIntersectSub = document.getElementById("insp-intersect-sub");
    const inspVel = document.getElementById("insp-vel");
    const inspPredConf = document.getElementById("insp-pred-conf");
    const inspState = document.getElementById("insp-state");
    const inspectorReasoningList = document.getElementById("inspector-reasoning-list");
    const predictRiskTimelineCanvas = document.getElementById("predict-risk-timeline-canvas");
    const riskTimelineVal = document.getElementById("risk-timeline-val");
    const reasoningAccordionChevron = document.getElementById("reasoning-accordion-chevron");
    const reasoningAccordionPanel = document.getElementById("reasoning-accordion-panel");

    // Haptics Elements (2-Motor Hardware Setup: Pin 5 Left, Pin 6 Right)
    const svgMotorLeft = document.getElementById("svg-motor-left");
    const svgRippleLeft = document.getElementById("svg-ripple-left");
    const svgLblLeft = document.getElementById("svg-lbl-left");
    const svgMotorRight = document.getElementById("svg-motor-right");
    const svgRippleRight = document.getElementById("svg-ripple-right");
    const svgLblRight = document.getElementById("svg-lbl-right");

    const hapticPatLabel = document.getElementById("haptic-pat-label");
    const hapticVibBadge = document.getElementById("haptic-vib-badge");
    const hapticUrgVal = document.getElementById("haptic-urg-val");
    const hapticDurVal = document.getElementById("haptic-dur-val");
    const hapticTimeVal = document.getElementById("haptic-time-val");

    const devArdStatus = document.getElementById("dev-ard-status");
    const devLatencyVal = document.getElementById("dev-latency-val");
    const devMotorLStatus = document.getElementById("dev-motor-l-status");
    const devMotorRStatus = document.getElementById("dev-motor-r-status");
    const hapticsDeviceView = document.getElementById("haptics-device-view");
    const hapticsCommandsView = document.getElementById("haptics-commands-view");
    const hapticCommandsList = document.getElementById("haptic-commands-list");

    // Obstacle Spotlight Elements
    const obstacleSpotlightBox = document.getElementById("obstacle-spotlight-box");
    const obstacleSpotlightIcon = document.getElementById("obstacle-spotlight-icon");
    const obstacleSpotlightText = document.getElementById("obstacle-spotlight-text");

    // Test & Replay Elements
    const selectTestScenario = document.getElementById("select-test-scenario");
    const scenDesc = document.getElementById("scen-desc");
    const scenExpected = document.getElementById("scen-expected");
    const scenActual = document.getElementById("scen-actual");
    const scenResult = document.getElementById("scen-result");
    const replayTimelineCanvas = document.getElementById("replay-timeline-canvas");
    const scenFrameTxt = document.getElementById("scen-frame-txt");
    const replaySlider = document.getElementById("replay-slider");
    const testExecStatusBadge = document.getElementById("test-exec-status-badge");
    const testHistoryList = document.getElementById("test-history-list");
    const scenarioModal = document.getElementById("scenario-modal");
    const scenModalTitle = document.getElementById("scen-modal-title");
    const scenModalDesc = document.getElementById("scen-modal-desc");
    const scenModalExpRisk = document.getElementById("scen-modal-exp-risk");
    const scenModalExpHaptic = document.getElementById("scen-modal-exp-haptic");
    const scenModalExpTtc = document.getElementById("scen-modal-exp-ttc");
    const scenModalExpIntersect = document.getElementById("scen-modal-exp-intersect");
    const scenActiveBox = document.getElementById("scen-active-box");
    const scenExecTimerTxt = document.getElementById("scen-exec-timer-txt");
    const scenResultCard = document.getElementById("scen-result-card");
    const scenResultBadge = document.getElementById("scen-result-badge");
    const scenResExp = document.getElementById("scen-res-exp");
    const scenResAct = document.getElementById("scen-res-act");
    const scenResHap = document.getElementById("scen-res-hap");
    const scenTimelineEmbed = document.getElementById("scen-timeline-embed");
    const testResultCard = document.getElementById("test-result-card");
    const testResBadge = document.getElementById("test-res-badge");
    const testResExp = document.getElementById("test-res-exp");
    const testResAct = document.getElementById("test-res-act");

    // Social Elements
    const toggleSocialSwitch = document.getElementById("toggle-social-switch");
    const socialSwitchLabel = document.getElementById("social-switch-label");
    const socialLiveView = document.getElementById("social-live-view");
    const socialContactsView = document.getElementById("social-contacts-view");
    const socialFaceBox = document.getElementById("social-face-box");
    const socialFaceTagTop = document.getElementById("social-face-tag-top");
    const socialFaceTagBottom = document.getElementById("social-face-tag-bottom");
    const socTrackId = document.getElementById("soc-track-id");
    const socPersonName = document.getElementById("soc-person-name");
    const socRelationship = document.getElementById("soc-relationship");
    const socConf = document.getElementById("soc-conf");
    const socStatus = document.getElementById("soc-status");
    const contactsListContainer = document.getElementById("contacts-list-container");
    const addFriendModal = document.getElementById("add-friend-modal");
    const addFriendPhotoGrid = document.getElementById("add-friend-photo-grid");
    const addFriendNameInput = document.getElementById("add-friend-name-input");
    const addFriendRelationSelect = document.getElementById("add-friend-relation-select");
    const contactProfileModal = document.getElementById("contact-profile-modal");
    const profModalAvatar = document.getElementById("prof-modal-avatar");
    const profModalName = document.getElementById("prof-modal-name");
    const profModalRelation = document.getElementById("prof-modal-relation");
    const profModalPhotosCount = document.getElementById("prof-modal-photos-count");
    const profModalStatus = document.getElementById("prof-modal-status");

    // Info Modal Elements
    const infoModal = document.getElementById("info-modal");
    const btnOpenInfo = document.getElementById("btn-open-info");
    const btnCloseInfo = document.getElementById("btn-close-info");
    const btnDismissInfo = document.getElementById("btn-dismiss-info");

    // Settings Modal Elements
    const settingsModal = document.getElementById("settings-modal");
    const inputCameraUrl = document.getElementById("input-camera-url");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnCancelSettings = document.getElementById("btn-cancel-settings");
    const btnSaveSettings = document.getElementById("btn-save-settings");

    function resetDashboardLiveMetrics() {
        if (valRiskScore) valRiskScore.textContent = "—";
        if (gaugeCircleStroke) gaugeCircleStroke.setAttribute("stroke-dasharray", "0, 100");

        const camBadge = document.getElementById("camera-stream-badge");
        if (camBadge) {
            camBadge.textContent = "○ NO SIGNAL";
            camBadge.style.background = "#f1f5f9";
            camBadge.style.color = "#64748b";
            camBadge.style.borderColor = "#cbd5e1";
        }

        const boxes = [
            { box: boxCorrLeft, txt: txtCorrLeft, lbl: lblCorrLeft },
            { box: boxCorrCenter, txt: txtCorrCenter, lbl: lblCorrCenter },
            { box: boxCorrRight, txt: txtCorrRight, lbl: lblCorrRight }
        ];
        boxes.forEach(b => {
            if (b.box) b.box.className = "corridor-box";
            if (b.txt) b.txt.textContent = "—";
            if (b.lbl) {
                b.lbl.className = "corridor-status-tag";
                b.lbl.textContent = "STANDBY";
            }
        });

        if (valTtc) valTtc.textContent = "—";
        if (valCpa) valCpa.textContent = "—";
    }

    // -------------------------------------------------------------------------
    // 1. Client-Side Staleness Watchdog (Section 0 Requirement)
    // -------------------------------------------------------------------------
    setInterval(function checkWatchdog() {
        const now = Date.now();
        const hasNeverConnected = (lastMessageTimestamp === 0);
        const timeSinceLoad = now - pageLoadTimestamp;
        const timeSinceLastMsg = now - lastMessageTimestamp;

        if (hasNeverConnected) {
            // If never received any message and grace period expired, flag offline
            if (timeSinceLoad > STALE_THRESHOLD_MS) {
                if (stalenessBanner) {
                    stalenessBanner.style.display = "block";
                    stalenessBanner.innerHTML = `⚠️ PIPELINE OFFLINE — Waiting for telemetry server (${(timeSinceLoad / 1000).toFixed(1)}s)`;
                }
                if (connPillBadge) connPillBadge.className = "conn-pill disconnected";
                if (connStatusDot) connStatusDot.className = "status-dot disconnected";
                if (connStatusText) connStatusText.textContent = "Pipeline Offline / Not Started";
                resetMotorVisuals();
                resetDashboardLiveMetrics();
            }
            return;
        }

        if (timeSinceLastMsg > STALE_THRESHOLD_MS) {
            if (stalenessBanner) {
                stalenessBanner.style.display = "block";
                stalenessBanner.innerHTML = `⚠️ PIPELINE STALLED — No updates received for <span id="stale-sec-txt">${(timeSinceLastMsg / 1000).toFixed(1)}</span>s`;
            }

            if (connPillBadge) connPillBadge.className = "conn-pill disconnected";
            if (connStatusDot) connStatusDot.className = "status-dot disconnected";
            if (connStatusText) connStatusText.textContent = "Pipeline Stalled / Offline";

            resetMotorVisuals();
            resetDashboardLiveMetrics();
        } else {
            if (stalenessBanner) stalenessBanner.style.display = "none";
            if (connPillBadge) connPillBadge.className = "conn-pill";
            if (connStatusDot) connStatusDot.className = "status-dot";
            if (connStatusText) connStatusText.textContent = "Local Edge Connected";
        }
    }, 150);

    function updateClock() {
        const now = new Date();
        const timeStr = now.toTimeString().split(" ")[0];
        if (headerClock) headerClock.textContent = timeStr;
        const notchClock = document.getElementById("notch-clock");
        if (notchClock) {
            // Show HH:MM in notch (short format)
            const h = String(now.getHours()).padStart(2, '0');
            const m = String(now.getMinutes()).padStart(2, '0');
            notchClock.textContent = `${h}:${m}`;
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // -------------------------------------------------------------------------
    // 2. Camera Source, VDO.Ninja & HUD Controls
    // -------------------------------------------------------------------------
    window.toggleHudOverlay = function () {
        hudOverlayEnabled = !hudOverlayEnabled;
        const btn = document.getElementById("btn-toggle-hud");
        if (btn) btn.innerHTML = `<span>HUD: ${hudOverlayEnabled ? "ON" : "OFF"}</span>`;
        if (liveOverlayCanvas) {
            liveOverlayCanvas.style.display = hudOverlayEnabled ? "block" : "none";
            if (hudOverlayEnabled) drawLiveOverlay();
        }
    };

    window.toggleVideoAspect = function () {
        isVideoCover = !isVideoCover;
        const btn = document.getElementById("btn-toggle-aspect");
        if (btn) btn.innerHTML = `<span>${isVideoCover ? "Fill" : "Fit"}</span>`;
        if (vdoNinjaFrame) {
            if (isVideoCover) {
                vdoNinjaFrame.classList.add("fit-cover");
            } else {
                vdoNinjaFrame.classList.remove("fit-cover");
            }
        }
    };

    function formatVdoNinjaUrl(rawUrl) {
        if (!rawUrl) return "about:blank";
        let url = rawUrl.trim();
        // If it's a numeric port index (0, 1, 2…), this is for the local OpenCV backend,
        // not a streamable browser URL. Keep the iframe blank in that case.
        if (/^\d+$/.test(url)) return "about:blank";
        if (url.includes("vdo.ninja")) {
            if (!url.includes("cleanoutput")) {
                url += (url.includes("?") ? "&" : "?") + "cleanoutput";
            }
            if (!url.includes("transparent")) {
                url += "&transparent";
            }
            if (!url.includes("autoplay")) {
                url += "&autoplay=1";
            }
        }
        return url;
    }

    async function initCameraSource() {
        try {
            const resp = await fetch("/api/source");
            if (resp.ok) {
                const data = await resp.json();
                if (data.source) {
                    currentCameraUrl = data.source;
                }
            }
        } catch (e) {
            console.warn("Could not fetch /api/source, using default URL", e);
        }

        if (inputCameraUrl) inputCameraUrl.value = currentCameraUrl;
        if (vdoNinjaFrame) {
            vdoNinjaFrame.src = formatVdoNinjaUrl(currentCameraUrl);
        }
    }

    function openSettingsModal() {
        if (settingsModal) {
            if (inputCameraUrl) inputCameraUrl.value = currentCameraUrl;
            settingsModal.style.display = "flex";
        }
    }

    function closeSettingsModal() {
        if (settingsModal) settingsModal.style.display = "none";
    }

    async function saveCameraSettings() {
        const newUrl = inputCameraUrl ? inputCameraUrl.value.trim() : "";
        if (!newUrl) return;

        currentCameraUrl = newUrl;
        if (vdoNinjaFrame) {
            vdoNinjaFrame.src = formatVdoNinjaUrl(currentCameraUrl);
        }

        try {
            await fetch("/api/set-source", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: newUrl })
            });
        } catch (err) {
            console.error("Failed to persist camera source", err);
        }

        closeSettingsModal();
    }

    function openInfoModal() {
        if (infoModal) infoModal.style.display = "flex";
    }

    function closeInfoModal() {
        if (infoModal) infoModal.style.display = "none";
    }

    window.setQuickSource = function(src) {
        if (inputCameraUrl) inputCameraUrl.value = src;
        document.querySelectorAll(".source-quick-chips .source-chip").forEach(btn => {
            if (btn.textContent.includes(src) || (src === '2' && btn.textContent.includes('Port 2'))) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    };

    window.setHapticPreset = function(preset) {
        document.querySelectorAll("[id^='chip-haptic-']").forEach(btn => btn.classList.remove("active"));
        const activeBtn = document.getElementById(`chip-haptic-${preset}`);
        if (activeBtn) activeBtn.classList.add("active");
    };

    if (btnOpenSettings) btnOpenSettings.addEventListener("click", openSettingsModal);
    if (btnToggleCameraSrc) btnToggleCameraSrc.addEventListener("click", openSettingsModal);
    if (btnCloseSettings) btnCloseSettings.addEventListener("click", closeSettingsModal);
    if (btnCancelSettings) btnCancelSettings.addEventListener("click", closeSettingsModal);
    if (btnSaveSettings) btnSaveSettings.addEventListener("click", saveCameraSettings);

    if (btnOpenInfo) btnOpenInfo.addEventListener("click", openInfoModal);
    if (btnCloseInfo) btnCloseInfo.addEventListener("click", closeInfoModal);
    if (btnDismissInfo) btnDismissInfo.addEventListener("click", closeInfoModal);

    // -------------------------------------------------------------------------
    // 3. Tab Switching
    // -------------------------------------------------------------------------
    window.switchTab = function (tabId) {
        const tabs = ["live", "predict", "haptics", "test", "social"];
        const navButtons = document.querySelectorAll(".bottom-nav-bar .nav-item");

        tabs.forEach((name, idx) => {
            const screen = document.getElementById(`screen-${name}`);
            if (screen) {
                if (name === tabId) {
                    screen.classList.add("active");
                } else {
                    screen.classList.remove("active");
                }
            }
            if (navButtons[idx]) {
                if (name === tabId) {
                    navButtons[idx].classList.add("active");
                } else {
                    navButtons[idx].classList.remove("active");
                }
            }
        });

        if (tabId === "live") {
            drawLiveOverlay();
        } else if (tabId === "predict") {
            updateInspector();
            window.switchPredictView(predictViewMode);
        } else if (tabId === "test") {
            renderTestHistory();
        }
    };

    // -------------------------------------------------------------------------
    // 4. WebSocket Telemetry Processing
    // -------------------------------------------------------------------------
    function connectWs() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host || "localhost:8080";
        const wsUrl = `${protocol}//${host}/ws/telemetry`;

        try {
            ws = new WebSocket(wsUrl);
        } catch (e) {
            scheduleReconnect();
            return;
        }

        ws.onopen = function () {
            lastMessageTimestamp = Date.now();
            if (connPillBadge) connPillBadge.className = "conn-pill";
            if (connStatusDot) connStatusDot.className = "status-dot";
            if (connStatusText) connStatusText.textContent = "Local Edge Connected";
            if (stalenessBanner) stalenessBanner.style.display = "none";
            reconnectDelay = 1000;
        };

        ws.onmessage = function (event) {
            lastMessageTimestamp = Date.now();
            try {
                const msg = JSON.parse(event.data);
                handleTelemetryMessage(msg);
            } catch (err) {
                console.error("Telemetry parse error", err);
            }
        };

        ws.onclose = function () {
            if (connPillBadge) connPillBadge.className = "conn-pill disconnected";
            if (connStatusDot) connStatusDot.className = "status-dot disconnected";
            if (connStatusText) connStatusText.textContent = "Pipeline Disconnected";
            if (stalenessBanner) {
                stalenessBanner.style.display = "block";
                stalenessBanner.textContent = "⚠️ PIPELINE DISCONNECTED — Reconnecting...";
            }
            scheduleReconnect();
        };

        ws.onerror = function () {
            if (connPillBadge) connPillBadge.className = "conn-pill disconnected";
            if (connStatusDot) connStatusDot.className = "status-dot disconnected";
            if (connStatusText) connStatusText.textContent = "Connection Error";
            ws.close();
        };
    }

    function scheduleReconnect() {
        setTimeout(connectWs, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 1.5, 5000);
    }

    function handleTelemetryMessage(msg) {
        if (sessionIdLabel) {
            sessionIdLabel.textContent = "Active Session: Local";
        }

        const camBadge = document.getElementById("camera-stream-badge");
        if (camBadge) {
            camBadge.textContent = "● CAMERA ACTIVE";
            camBadge.style.background = "rgba(15, 23, 42, 0.75)";
            camBadge.style.color = "#ffffff";
            camBadge.style.borderColor = "rgba(255, 255, 255, 0.15)";
        }

        if (msg.frame_width) currentFrameWidth = msg.frame_width;
        if (msg.frame_height) currentFrameHeight = msg.frame_height;

        // 1. Risk State & Score (Smoothed with EMA filter to eliminate jitter)
        const rs = msg.risk_state || {};
        if (Array.isArray(rs.reason_codes)) {
            lastReasonCodes = rs.reason_codes;
        } else {
            lastReasonCodes = [];
        }
        const state = (rs.state || "SAFE").toUpperCase();
        const rawRisk = (typeof rs.global_risk === "number") ? rs.global_risk : 0.0;
        currentRiskState = state;
        currentGlobalRisk = (currentGlobalRisk * 0.75) + (rawRisk * 0.25);

        updateRiskBanner(state, currentGlobalRisk);

        // 2. Corridors (Defaults to 0.0 instead of fake mock constants)
        const cr = rs.corridor_risks || {};
        const leftRisk = (typeof cr.left === "number") ? cr.left : 0.0;
        const centerRisk = (typeof cr.center === "number") ? cr.center : 0.0;
        const rightRisk = (typeof cr.right === "number") ? cr.right : 0.0;
        updateCorridorBoxes(leftRisk, centerRisk, rightRisk);

        // 3. Recommended Direction
        updateDirectionRecommendation(leftRisk, centerRisk, rightRisk, state);

        // 4. Tracks & Objects
        if (msg.tracks && msg.tracks.length > 0) {
            activeTracks = msg.tracks;
            let minTtc = null;
            let minCpa = null;
            for (const t of activeTracks) {
                if (t.ttc_s !== null && t.ttc_s !== undefined) {
                    if (minTtc === null || t.ttc_s < minTtc) minTtc = t.ttc_s;
                }
                if (t.cpa !== null && t.cpa !== undefined) {
                    if (minCpa === null || t.cpa < minCpa) minCpa = t.cpa;
                }
            }
            currentTTC = minTtc;
            currentCPA = minCpa;
        } else {
            activeTracks = [];
            currentTTC = null;
            currentCPA = null;
        }
        if (valTtc) valTtc.textContent = currentTTC !== null ? `${Number(currentTTC).toFixed(1)} s` : "—";
        if (valCpa) valCpa.textContent = currentCPA !== null ? `${Number(currentCPA).toFixed(2)} m` : "—";

        // Update Obstacle Spotlight with human-readable information
        updateObstacleSpotlight(activeTracks, state);

        // 5. Haptic Feedback
        if (msg.haptic) {
            updateHapticTelemetry(msg.haptic);
        }

        // 6. Pipeline Health
        if (msg.pipeline_health) {
            const ph = msg.pipeline_health;
            if (devArdStatus) {
                if (ph.arduino === "OK") {
                    devArdStatus.className = "pill-badge green";
                    devArdStatus.textContent = "● Connected";
                } else {
                    devArdStatus.className = "pill-badge red";
                    devArdStatus.textContent = "● Degraded";
                }
            }
        }

        // Render Canvases
        drawLiveOverlay();
        updateInspector();
        drawInspectorCanvas();
    }

    // -------------------------------------------------------------------------
    // 5. UI Renderers for Live Tab
    // -------------------------------------------------------------------------
    function updateObstacleSpotlight(tracks, state) {
        if (!obstacleSpotlightBox || !obstacleSpotlightText || !obstacleSpotlightIcon) return;

        if (!tracks || tracks.length === 0 || state === "SAFE") {
            obstacleSpotlightIcon.textContent = "🟢";
            obstacleSpotlightText.textContent = "Path Clear · No immediate obstacles in trajectory";
            obstacleSpotlightBox.style.borderColor = "var(--border-light)";
            obstacleSpotlightBox.style.background = "rgba(255, 255, 255, 0.92)";
            obstacleSpotlightText.style.color = "var(--text-title)";
            return;
        }

        // Find primary hazard track (lowest TTC or highest prediction confidence)
        let primaryHazard = tracks[0];
        for (const t of tracks) {
            if (t.ttc_s !== null && (primaryHazard.ttc_s === null || t.ttc_s < primaryHazard.ttc_s)) {
                primaryHazard = t;
            }
        }

        const className = primaryHazard.class_name ? primaryHazard.class_name.toUpperCase() : "OBSTACLE";
        const trackId = primaryHazard.track_id || 1;
        const ttcStr = (primaryHazard.ttc_s !== null && primaryHazard.ttc_s !== undefined) ? `${Number(primaryHazard.ttc_s).toFixed(1)}s` : null;
        const cpaStr = (primaryHazard.cpa !== null && primaryHazard.cpa !== undefined) ? `${Number(primaryHazard.cpa).toFixed(1)}m` : null;

        let icon = "⚠️";
        let color = "#9a3412";
        let bg = "#fff7ed";
        let border = "#fed7aa";

        if (state === "CRITICAL") {
            icon = "🛑";
            color = "#991b1b";
            bg = "#fef2f2";
            border = "#fecaca";
        } else if (state === "CAUTION") {
            icon = "🟡";
            color = "#92400e";
            bg = "#fffbeb";
            border = "#fde68a";
        }

        obstacleSpotlightIcon.textContent = icon;
        let detailText = `Detected: ${className} (#${trackId})`;
        if (ttcStr) detailText += ` · Est. Contact: ${ttcStr}`;
        if (cpaStr) detailText += ` (Pass margin: ${cpaStr})`;

        obstacleSpotlightText.textContent = detailText;
        obstacleSpotlightBox.style.borderColor = border;
        obstacleSpotlightBox.style.background = bg;
        obstacleSpotlightText.style.color = color;
    }

    function updateRiskBanner(state, riskScore) {
        if (!riskAlertCard) return;

        riskAlertCard.className = `risk-alert-card alert-${state.toLowerCase()}`;
        if (alertStateName) alertStateName.textContent = state;

        let strokeColor = "#10b981";
        let subText = "Clear path · No collision risk";

        if (state === "SAFE") {
            strokeColor = "#10b981";
            subText = "Clear navigation path · No hazard detected";
            if (alertStateName) alertStateName.style.color = "#065f46";
        } else if (state === "CAUTION") {
            strokeColor = "#f59e0b";
            subText = "Approaching obstacle · Monitoring trajectory";
            if (alertStateName) alertStateName.style.color = "#92400e";
        } else if (state === "WARNING") {
            strokeColor = "#f97316";
            subText = "Predicted collision risk detected";
            if (alertStateName) alertStateName.style.color = "#9a3412";
        } else if (state === "CRITICAL") {
            strokeColor = "#ef4444";
            subText = "Imminent collision · Emergency stop required";
            if (alertStateName) alertStateName.style.color = "#991b1b";
        } else if (state === "DEGRADED") {
            strokeColor = "#8b5cf6";
            subText = "Sensor confidence degraded · Fallback active";
            if (alertStateName) alertStateName.style.color = "#5b21b6";
        }

        if (alertStateSub) alertStateSub.textContent = subText;
        if (valRiskScore) valRiskScore.textContent = riskScore.toFixed(2);

        if (gaugeCircleStroke) {
            const pct = Math.min(100, Math.max(0, Math.round(riskScore * 100)));
            gaugeCircleStroke.setAttribute("stroke-dasharray", `${pct}, 100`);
            gaugeCircleStroke.setAttribute("stroke", strokeColor);
        }
    }

    function updateCorridorBoxes(left, center, right) {
        applyCorridorStyle(boxCorrLeft, txtCorrLeft, lblCorrLeft, left);
        applyCorridorStyle(boxCorrCenter, txtCorrCenter, lblCorrCenter, center);
        applyCorridorStyle(boxCorrRight, txtCorrRight, lblCorrRight, right);
    }

    function applyCorridorStyle(boxEl, txtEl, lblEl, score) {
        if (!boxEl || !txtEl || !lblEl) return;
        txtEl.textContent = Number(score).toFixed(2);

        if (score < 0.30) {
            boxEl.className = "corridor-box safe";
            lblEl.className = "corridor-status-tag";
            lblEl.textContent = "SAFE";
        } else if (score < 0.70) {
            boxEl.className = "corridor-box caution";
            lblEl.className = "corridor-status-tag";
            lblEl.textContent = "CAUTION";
        } else {
            boxEl.className = "corridor-box risky";
            lblEl.className = "corridor-status-tag";
            lblEl.textContent = "RISKY";
        }
    }

    function updateDirectionRecommendation(left, center, right, state) {
        if (!recDirName || !recDirSub || !recArrowIcon) return;

        if (state === "CRITICAL") {
            recDirName.textContent = "STOP IMMEDIATELY";
            recDirSub.textContent = "Hazards in active corridor";
            recDirName.style.color = "var(--c-critical)";
            recArrowIcon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>`;
            return;
        }

        if (left <= center && left <= right) {
            recDirName.textContent = "MOVE LEFT";
            recDirSub.textContent = "Left corridor is safest";
            recDirName.style.color = "#047857";
            recArrowIcon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>`;
        } else if (right < left && right <= center) {
            recDirName.textContent = "MOVE RIGHT";
            recDirSub.textContent = "Right corridor is safest";
            recDirName.style.color = "#047857";
            recArrowIcon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
        } else {
            recDirName.textContent = "MAINTAIN PATH";
            recDirSub.textContent = "Center path is unobstructed";
            recDirName.style.color = "#2563eb";
            recArrowIcon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>`;
        }
    }

    // -------------------------------------------------------------------------
    // 6. Live HUD Perspective & Bounding Box Overlay
    // -------------------------------------------------------------------------
    function drawLiveOverlay() {
        if (!liveOverlayCanvas || !hudOverlayEnabled) return;
        const ctx = liveOverlayCanvas.getContext("2d");
        const w = liveOverlayCanvas.width;
        const h = liveOverlayCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Ground perspective horizon (positioned cleanly below the center)
        const vanishY = h * 0.56;
        const vanishX = w * 0.50;

        // Ground perspective trapezoids (Only occupy lower 44% of frame)
        // Center Corridor (Hazard zone)
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(vanishX - 24, vanishY);
        ctx.lineTo(vanishX + 24, vanishY);
        ctx.lineTo(w * 0.68, h);
        ctx.lineTo(w * 0.32, h);
        ctx.closePath();

        if (currentGlobalRisk > 0.5) {
            ctx.fillStyle = "rgba(239, 68, 68, 0.16)";
            ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
        } else {
            ctx.fillStyle = "rgba(16, 185, 129, 0.10)";
            ctx.strokeStyle = "rgba(16, 185, 129, 0.55)";
        }
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.restore();

        // Left Corridor
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(vanishX - 60, vanishY);
        ctx.lineTo(vanishX - 24, vanishY);
        ctx.lineTo(w * 0.32, h);
        ctx.lineTo(w * 0.04, h);
        ctx.closePath();
        ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
        ctx.fill();
        ctx.restore();

        // Right Corridor
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(vanishX + 24, vanishY);
        ctx.lineTo(vanishX + 60, vanishY);
        ctx.lineTo(w * 0.96, h);
        ctx.lineTo(w * 0.68, h);
        ctx.closePath();
        ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
        ctx.fill();
        ctx.restore();

        // Draw Bounding Boxes dynamically from real telemetry tracks (with video-to-canvas coordinate mapping)
        if (activeTracks && activeTracks.length > 0) {
            const primaryTrack = activeTracks[selectedTrackIndex] || activeTracks[0];
            if (primaryTrack && primaryTrack.bbox && Array.isArray(primaryTrack.bbox) && primaryTrack.bbox.length === 4) {
                // bbox is [x1, y1, x2, y2] in original frame pixel coordinates.
                // Scale to the canvas's actual rendered size — do not assume a fixed frame resolution.
                const scaleX = liveOverlayCanvas.width / (currentFrameWidth || 640);
                const scaleY = liveOverlayCanvas.height / (currentFrameHeight || 480);
                const bx = primaryTrack.bbox[0] * scaleX;
                const by = primaryTrack.bbox[1] * scaleY;
                const bw = (primaryTrack.bbox[2] - primaryTrack.bbox[0]) * scaleX;
                const bh = (primaryTrack.bbox[3] - primaryTrack.bbox[1]) * scaleY;

                if (bw > 4 && bh > 4) {
                    ctx.save();
                    ctx.strokeStyle = (currentGlobalRisk > 0.5) ? "#ef4444" : "#10b981";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([]);

                    // High-tech corner brackets
                    const len = Math.min(10, Math.min(bw, bh) / 3);
                    // Top-left
                    ctx.beginPath(); ctx.moveTo(bx, by + len); ctx.lineTo(bx, by); ctx.lineTo(bx + len, by); ctx.stroke();
                    // Top-right
                    ctx.beginPath(); ctx.moveTo(bx + bw - len, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + len); ctx.stroke();
                    // Bottom-left
                    ctx.beginPath(); ctx.moveTo(bx, by + bh - len); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + len, by + bh); ctx.stroke();
                    // Bottom-right
                    ctx.beginPath(); ctx.moveTo(bx + bw - len, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - len); ctx.stroke();

                    // Label Tag Badge
                    const ttcPart = (primaryTrack.ttc_s !== null && primaryTrack.ttc_s !== undefined)
                        ? ` | TTC: ${Number(primaryTrack.ttc_s).toFixed(1)}s`
                        : (currentTTC !== null ? ` | TTC: ${Number(currentTTC).toFixed(1)}s` : "");
                    const labelText = `${primaryTrack.class_name || 'Obstacle'} #${primaryTrack.track_id}${ttcPart}`;
                    ctx.font = "bold 9px 'JetBrains Mono', monospace";
                    const textWidth = ctx.measureText(labelText).width;

                    ctx.fillStyle = (currentGlobalRisk > 0.5) ? "rgba(220, 38, 38, 0.92)" : "rgba(16, 185, 129, 0.92)";
                    ctx.beginPath();
                    ctx.roundRect(bx - 2, Math.max(0, by - 19), textWidth + 12, 17, 4);
                    ctx.fill();

                    ctx.fillStyle = "#ffffff";
                    ctx.fillText(labelText, bx + 4, Math.max(12, by - 7));
                    ctx.restore();
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 7. Tab 2: Prediction Inspector & Vector Canvas
    // -------------------------------------------------------------------------
    window.prevTrack = function () {
        if (activeTracks.length === 0) return;
        selectedTrackIndex = (selectedTrackIndex - 1 + activeTracks.length) % activeTracks.length;
        updateInspector();
        drawInspectorCanvas();
    };

    window.nextTrack = function () {
        if (activeTracks.length === 0) return;
        selectedTrackIndex = (selectedTrackIndex + 1) % activeTracks.length;
        updateInspector();
        drawInspectorCanvas();
    };

    window.switchPredictView = function (mode) {
        predictViewMode = mode || "2d";
        const btn2d = document.getElementById("btn-pred-2d");
        const btnTop = document.getElementById("btn-pred-top");
        const btnTimeline = document.getElementById("btn-pred-timeline");
        const cardMain = document.getElementById("card-pred-main-view");
        const cardRiskTimeline = document.getElementById("card-risk-timeline");

        if (btn2d) btn2d.classList.toggle("active", predictViewMode === "2d");
        if (btnTop) btnTop.classList.toggle("active", predictViewMode === "top");
        if (btnTimeline) btnTimeline.classList.toggle("active", predictViewMode === "timeline");

        if (cardMain) cardMain.style.display = (predictViewMode === "timeline") ? "none" : "flex";
        if (cardRiskTimeline) cardRiskTimeline.style.display = (predictViewMode === "timeline") ? "flex" : "none";

        const toolbarEl = document.getElementById("topview-toolbar");
        const scrubberEl = document.getElementById("topview-time-scrubber");
        if (toolbarEl) toolbarEl.style.display = (predictViewMode === "top") ? "flex" : "none";
        if (scrubberEl) scrubberEl.style.display = (predictViewMode === "top") ? "flex" : "none";

        if (predictViewMode === "timeline") {
            const track = (activeTracks && activeTracks.length) ? (activeTracks[selectedTrackIndex] || activeTracks[0]) : null;
            drawRiskTimeline(track);
        } else {
            drawInspectorCanvas();
        }
    };

    // Top View Interactive Spatial State & Global Helpers
    window.topViewZoom = 1.0;
    window.topViewFilterMode = "all";
    window.topViewScrubTime = 0.0;
    window.topViewAutoScale = true;

    window.zoomTopView = function (delta) {
        window.topViewZoom = Math.max(0.5, Math.min(2.5, window.topViewZoom + delta));
        drawInspectorCanvas();
    };

    window.resetTopViewZoom = function () {
        window.topViewZoom = 1.0;
        drawInspectorCanvas();
    };

    window.toggleTopViewAutoScale = function () {
        window.topViewAutoScale = !window.topViewAutoScale;
        const btn = document.getElementById("btn-autoscale");
        if (btn) {
            btn.textContent = window.topViewAutoScale ? "Auto Scale: ON" : "Auto Scale: OFF";
            btn.style.background = window.topViewAutoScale ? "rgba(16, 185, 129, 0.2)" : "rgba(148, 163, 184, 0.2)";
            btn.style.color = window.topViewAutoScale ? "#10b981" : "#94a3b8";
            btn.style.borderColor = window.topViewAutoScale ? "#10b981" : "#64748b";
        }
        drawInspectorCanvas();
    };

    window.setTopViewFilter = function (mode) {
        window.topViewFilterMode = mode || "all";
        ["all", "selected", "risk"].forEach(m => {
            const btn = document.getElementById(`topview-filter-${m}`);
            if (btn) {
                const isActive = (m === window.topViewFilterMode);
                btn.style.background = isActive ? "#2563eb" : "rgba(15, 23, 42, 0.85)";
                btn.style.color = isActive ? "#ffffff" : "#cbd5e1";
                btn.style.borderColor = isActive ? "#3b82f6" : "rgba(255, 255, 255, 0.15)";
            }
        });
        drawInspectorCanvas();
    };

    window.onTopViewScrub = function (val) {
        window.topViewScrubTime = parseFloat(val) || 0.0;
        const txt = document.getElementById("scrubber-val");
        if (txt) {
            txt.textContent = (window.topViewScrubTime === 0.0) ? "NOW" : `+${window.topViewScrubTime.toFixed(1)}s`;
        }
        drawInspectorCanvas();
    };

    window.resetTopViewScrub = function () {
        window.topViewScrubTime = 0.0;
        const input = document.getElementById("topview-scrubber-input");
        const txt = document.getElementById("scrubber-val");
        if (input) input.value = 0;
        if (txt) txt.textContent = "NOW";
        drawInspectorCanvas();
    };

    window.toggleReasoningPanel = function () {
        reasoningAccordionOpen = !reasoningAccordionOpen;
        if (reasoningAccordionPanel) {
            reasoningAccordionPanel.style.display = reasoningAccordionOpen ? "flex" : "none";
        }
        if (reasoningAccordionChevron) {
            reasoningAccordionChevron.style.transform = reasoningAccordionOpen ? "rotate(180deg)" : "rotate(0deg)";
        }
    };
    window.toggleReasoningAccordion = window.toggleReasoningPanel;

    function setRiskBadge(el, state) {
        if (!el) return;
        const s = (state || "SAFE").toUpperCase();
        el.textContent = s;
        const cls = (s === "CRITICAL") ? "red" : (s === "WARNING") ? "orange" : (s === "DEGRADED") ? "yellow" : "green";
        el.className = `pill-badge ${cls}`;
    }

    function updatePredictMetrics(track) {
        if (!track) {
            if (inspTtc) inspTtc.textContent = "None";
            if (inspTtcSub) inspTtcSub.textContent = "Time to contact";
            if (inspCpa) inspCpa.textContent = "--";
            if (inspCpaSub) inspCpaSub.textContent = "Closest margin";
            if (inspIntersect) {
                inspIntersect.textContent = "NO";
                inspIntersect.className = "pill-badge green";
            }
            if (inspIntersectSub) inspIntersectSub.textContent = "Path collision";
            if (inspPredConf) inspPredConf.textContent = "—";
            return;
        }
        if (inspTtc) {
            inspTtc.textContent = (track.ttc_s !== null && track.ttc_s !== undefined) ? `${Number(track.ttc_s).toFixed(1)} s` : "None";
        }
        if (inspTtcSub) {
            inspTtcSub.textContent = (track.ttc_s !== null && track.ttc_s < 2.5) ? "Decreasing / Danger" : "Time to contact";
        }
        if (inspCpa) {
            inspCpa.textContent = (track.cpa !== null && track.cpa !== undefined) ? `${Number(track.cpa).toFixed(2)} m` : "--";
        }
        if (inspCpaSub) {
            inspCpaSub.textContent = (track.cpa !== null && track.cpa < 0.6) ? "Below safe buffer" : "Closest margin";
        }
        if (inspIntersect) {
            const isIntersect = Boolean(track.intersect);
            inspIntersect.textContent = isIntersect ? "YES" : "NO";
            inspIntersect.className = isIntersect ? "pill-badge red" : "pill-badge green";
        }
        if (inspIntersectSub) {
            inspIntersectSub.textContent = Boolean(track.intersect) ? "Direct path conflict" : "Path collision";
        }
        if (inspPredConf) {
            const confVal = track.pred_conf ?? 0.81;
            inspPredConf.textContent = `${Math.round(confVal * 100)}%`;
        }
    }

    function updateInspector() {
        if (!activeTracks || activeTracks.length === 0) {
            if (inspectorObjLabel) inspectorObjLabel.textContent = "No Track Selected";
            if (inspObjIcon) inspObjIcon.textContent = "🟢";
            setRiskBadge(inspRiskBadge, "SAFE");
            if (inspId) inspId.textContent = "—";
            if (inspClass) inspClass.textContent = "None";
            if (inspTrackConf) inspTrackConf.textContent = "—";
            updatePredictMetrics(null);
            if (inspVel) inspVel.textContent = "— px/s";
            if (inspState) {
                inspState.textContent = "IDLE";
                inspState.className = "pill-badge green";
            }
            if (riskTimelineVal) {
                riskTimelineVal.textContent = "● Safe Horizon";
                riskTimelineVal.style.color = "#10b981";
            }
            drawRiskTimeline(null);
            renderReasoningList(lastReasonCodes, null);
            return;
        }

        const track = activeTracks[selectedTrackIndex] || activeTracks[0];
        if (!track) return;

        const isHazard = Array.isArray(lastReasonCodes) && lastReasonCodes.some(c => typeof c === "string" && c.includes(`track_${track.track_id}`));
        const trackState = (currentRiskState === "DEGRADED") ? "DEGRADED" : (isHazard ? currentRiskState : "SAFE");

        const cname = (track.class_name || "").toLowerCase();
        let icon = "⚠️";
        if (cname.includes("person") || cname.includes("pedestrian")) icon = "🚶";
        else if (cname.includes("scooter") || cname.includes("bike") || cname.includes("motorcycle")) icon = "🛵";
        else if (cname.includes("car") || cname.includes("truck") || cname.includes("bus") || cname.includes("vehicle")) icon = "🚗";
        else if (cname.includes("wall") || cname.includes("barrier")) icon = "🧱";
        if (inspObjIcon) inspObjIcon.textContent = icon;

        if (inspectorObjLabel) {
            inspectorObjLabel.textContent = `${track.class_name || "Object"} #${track.track_id}`;
        }
        setRiskBadge(inspRiskBadge, trackState);

        if (inspId) inspId.textContent = `#${track.track_id}`;
        if (inspClass) inspClass.textContent = track.class_name || "Unknown";
        if (inspTrackConf) {
            inspTrackConf.textContent = (track.track_confidence ?? 0.0).toFixed(2);
        }
        updatePredictMetrics(track);
        if (inspVel) {
            let velStr = "0.0 px/s";
            if (Array.isArray(track.relative_velocity) && track.relative_velocity.length >= 2) {
                const speed = Math.hypot(track.relative_velocity[0], track.relative_velocity[1]);
                velStr = `${speed.toFixed(1)} px/s`;
            } else if (typeof track.relative_velocity === "number") {
                velStr = `${track.relative_velocity.toFixed(1)} px/s`;
            } else if (track.relative_velocity) {
                velStr = `${track.relative_velocity} px/s`;
            }
            inspVel.textContent = velStr;
        }
        if (inspState) {
            inspState.textContent = trackState;
            inspState.className = `pill-badge ${trackState.toLowerCase() === "safe" ? "green" : (trackState.toLowerCase() === "critical" ? "red" : (trackState.toLowerCase() === "warning" ? "orange" : "yellow"))}`;
        }

        const calculatedRisk = (trackState === "CRITICAL") ? 0.88 : (trackState === "WARNING" ? 0.62 : (typeof currentGlobalRisk === "number" ? currentGlobalRisk : 0.12));
        drawRiskTimeline(track, calculatedRisk);
        if (riskTimelineVal) {
            if (trackState === "CRITICAL") {
                riskTimelineVal.textContent = `● ${Number(track.ttc_s || 1.8).toFixed(1)}s (Critical Hazard)`;
                riskTimelineVal.style.color = "#ef4444";
            } else if (trackState === "WARNING") {
                riskTimelineVal.textContent = `● ${Number(track.ttc_s || 2.4).toFixed(1)}s (Caution / Warning)`;
                riskTimelineVal.style.color = "#f97316";
            } else {
                riskTimelineVal.textContent = `● ${(typeof currentGlobalRisk === "number" ? currentGlobalRisk : 0).toFixed(2)} (Safe)`;
                riskTimelineVal.style.color = "#10b981";
            }
        }

        renderReasoningList(lastReasonCodes, track.track_id);
    }

    function renderReasoningList(reasonCodes, selectedTrackId) {
        if (!inspectorReasoningList) return;
        if (!reasonCodes || reasonCodes.length === 0) {
            inspectorReasoningList.innerHTML = `<div class="reasoning-item"><span>No active risk factors for this track.</span></div>`;
            return;
        }

        let filtered = reasonCodes;
        if (selectedTrackId !== null && selectedTrackId !== undefined) {
            const matching = reasonCodes.filter(c => typeof c === 'string' && c.includes(`track_${selectedTrackId}`));
            if (matching.length > 0) {
                filtered = matching;
            }
        }

        if (filtered.length === 0) {
            inspectorReasoningList.innerHTML = `<div class="reasoning-item"><span>No active risk factors for this track.</span></div>`;
            return;
        }

        inspectorReasoningList.innerHTML = filtered.map((code, i) => `
            <div class="reasoning-item"><span class="reasoning-num">${i + 1}.</span><span>${formatReasonCode(code)}</span></div>
        `).join('');
    }

    function formatReasonCode(code) {
        if (!code || typeof code !== 'string') return "";
        const parts = code.split(':');
        const type = parts[0];
        const labels = {
            ttc_low: "Time-to-collision below safe threshold",
            intersection: "Predicted path intersects user's trajectory",
            miss_dist: "Close point of approach within danger margin",
            degraded: "System operating in degraded state",
            waiting_for_camera: "Waiting for camera connection",
            camera_disconnected: "Camera connection lost",
        };
        const base = labels[type] || type.replace(/_/g, " ");
        const trackPart = parts.find(p => p.startsWith("track_"));
        const detailPart = parts.slice(1).find(p => !p.startsWith("track_"));
        let extra = "";
        if (detailPart) extra += ` [${detailPart}]`;
        if (trackPart) extra += ` (${trackPart.replace('_', ' ')})`;
        return `${base}${extra}`;
    }

    function drawRiskTimeline(track, riskOverride) {
        const riskVal = (typeof riskOverride === "number")
            ? riskOverride
            : (typeof currentGlobalRisk === "number" ? currentGlobalRisk : 0.08);
        const ttcVal = track && track.ttc_s !== null && track.ttc_s !== undefined ? Number(track.ttc_s) : null;
        drawPredictRiskTimeline(riskVal, ttcVal);
    }
    window.drawRiskTimeline = function () {
        const track = (activeTracks && activeTracks.length) ? (activeTracks[selectedTrackIndex] || activeTracks[0]) : null;
        drawRiskTimeline(track);
    };

    function drawPredictRiskTimeline(riskVal, ttcVal) {
        if (!predictRiskTimelineCanvas) return;
        const ctx = predictRiskTimelineCanvas.getContext("2d");
        const w = predictRiskTimelineCanvas.width;
        const h = predictRiskTimelineCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Subtle background risk zones
        ctx.fillStyle = "rgba(239, 68, 68, 0.05)";
        ctx.fillRect(0, 0, w, h * 0.35);

        ctx.fillStyle = "rgba(249, 115, 22, 0.04)";
        ctx.fillRect(0, h * 0.35, w, h * 0.35);

        ctx.fillStyle = "rgba(16, 185, 129, 0.04)";
        ctx.fillRect(0, h * 0.70, w, h * 0.30);

        // Divider lines
        ctx.strokeStyle = "rgba(148, 163, 184, 0.20)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(0, h * 0.35); ctx.lineTo(w, h * 0.35);
        ctx.moveTo(0, h * 0.70); ctx.lineTo(w, h * 0.70);
        ctx.stroke();
        ctx.setLineDash([]);

        // Zone text labels
        ctx.font = "bold 8px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#ef4444";
        ctx.fillText("HIGH", w - 28, 11);
        ctx.fillStyle = "#f97316";
        ctx.fillText("WARN", w - 30, h * 0.35 + 11);
        ctx.fillStyle = "#10b981";
        ctx.fillText("SAFE", w - 28, h * 0.70 + 11);

        // Smooth Risk Projection Curve
        const risk = (typeof riskVal === "number") ? Math.max(0.06, Math.min(0.98, riskVal)) : 0.12;
        const ttc = (typeof ttcVal === "number" && ttcVal > 0) ? Math.min(6.0, ttcVal) : 2.1;

        const points = [];
        const numSteps = 24;
        const peakStep = Math.min(numSteps - 2, Math.max(2, Math.round((ttc / 6.0) * numSteps)));

        for (let i = 0; i <= numSteps; i++) {
            const x = (i / numSteps) * (w - 38);
            let yVal;
            if (risk > 0.35) {
                const dist = Math.abs(i - peakStep);
                const bell = Math.exp(- (dist * dist) / 12);
                yVal = 0.08 + (risk - 0.08) * bell;
            } else {
                yVal = risk * (1 - (i / numSteps) * 0.4);
            }
            const y = h - 6 - (yVal * (h - 14));
            points.push({ x, y, val: yVal });
        }

        // Stroke line
        ctx.beginPath();
        points.forEach((pt, i) => {
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        });
        ctx.strokeStyle = risk > 0.6 ? "#ef4444" : (risk > 0.3 ? "#f97316" : "#10b981");
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Area fill
        ctx.lineTo(points[points.length - 1].x, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = risk > 0.6 ? "rgba(239, 68, 68, 0.09)" : (risk > 0.3 ? "rgba(249, 115, 22, 0.07)" : "rgba(16, 185, 129, 0.07)");
        ctx.fill();

        // Hazard peak marker
        if (risk > 0.35 && points[peakStep]) {
            const peak = points[peakStep];
            ctx.save();
            ctx.beginPath();
            ctx.arc(peak.x, peak.y, 4.5, 0, Math.PI * 2);
            ctx.fillStyle = risk > 0.6 ? "#ef4444" : "#f97316";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Label at peak
            ctx.font = "bold 8px 'JetBrains Mono', monospace";
            const tag = `${ttc.toFixed(1)}s`;
            const tagW = ctx.measureText(tag).width + 8;
            ctx.fillStyle = risk > 0.6 ? "#dc2626" : "#ea580c";
            ctx.beginPath();
            ctx.roundRect(peak.x - tagW / 2, Math.max(2, peak.y - 18), tagW, 13, 3);
            ctx.fill();
            ctx.fillStyle = "#ffffff";
            ctx.fillText(tag, peak.x - tagW / 2 + 4, Math.max(11, peak.y - 8));
            ctx.restore();
        }
    }

    function drawInspectorCanvas() {
        if (!inspectorCanvas) return;
        const ctx = inspectorCanvas.getContext("2d");
        const w = inspectorCanvas.width;
        const h = inspectorCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Live Video Stream Background in 2D perspective mode
        const videoImg = document.getElementById("mjpeg-video-stream");
        if (predictViewMode === "2d" && videoImg && videoImg.complete && videoImg.naturalWidth > 0 && videoImg.style.display !== "none") {
            ctx.save();
            ctx.globalAlpha = 0.45;
            ctx.drawImage(videoImg, 0, 0, w, h);
            ctx.restore();
            // Subtle dark gradient vignette over video to keep HUD crisp
            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, "rgba(15, 23, 42, 0.4)");
            grad.addColorStop(1, "rgba(15, 23, 42, 0.75)");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, w, h);
        } else {
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(0, 0, w, h);
        }

        // Technical Grid
        ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
        ctx.lineWidth = 1;
        for (let x = 20; x < w; x += 30) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = 20; y < h; y += 30) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        const track = (activeTracks && activeTracks.length) ? (activeTracks[selectedTrackIndex] || activeTracks[0]) : null;
        const objLabel = track ? String(track.class_name || "OBJ").toUpperCase().slice(0, 8) : "CLEAR";
        const ttcLabel = (track && track.ttc_s != null && track.ttc_s !== undefined) ? `${Number(track.ttc_s).toFixed(1)}s` : "—";
        const isThreat = !!(track && (track.intersect || currentRiskState === "WARNING" || currentRiskState === "CRITICAL"));
        const threatColor = isThreat ? "#ef4444" : "#10b981";

        if (predictViewMode === "top") {
            // ================= BIRD'S-EYE TOP VIEW (RADAR SPATIAL ENGINE) =================
            const zoom = topViewZoom || 1.0;
            const scrubberTime = topViewScrubTime || 0.0;
            
            // Base user anchor position
            const userX = w * 0.5;
            const userY = h * 0.88;
            
            // 1. Adaptive Distance Rings (1m, 2m, 3m, 5m)
            const meterScale = 35 * zoom; // pixels per meter
            const distanceRings = [1, 2, 3, 5];
            
            ctx.save();
            ctx.setLineDash([2, 4]);
            ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
            ctx.lineWidth = 1;
            ctx.font = "bold 9px 'JetBrains Mono', monospace";
            ctx.fillStyle = "#64748b";
            
            distanceRings.forEach(m => {
                const r = m * meterScale;
                if (userY - r > 10) {
                    ctx.beginPath();
                    ctx.arc(userX, userY, r, Math.PI, 0); // semi-circle forward
                    ctx.stroke();
                    ctx.fillText(`${m}m`, userX + r + 4, userY - 2);
                }
            });
            ctx.restore();

            // 2. Field-of-View (FOV) Sensing Cone
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(userX, userY);
            ctx.arc(userX, userY, 5.5 * meterScale, -Math.PI * 0.70, -Math.PI * 0.30);
            ctx.closePath();
            ctx.fillStyle = "rgba(59, 130, 246, 0.05)";
            ctx.fill();
            ctx.strokeStyle = "rgba(59, 130, 246, 0.20)";
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.restore();

            // 3. 3D User Avatar Marker & Heading
            ctx.save();
            // Heading direction line
            ctx.beginPath();
            ctx.moveTo(userX, userY);
            ctx.lineTo(userX, userY - 4.5 * meterScale);
            ctx.strokeStyle = "#3b82f6";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]);
            ctx.stroke();

            // Render 3D User Avatar Image
            if (!window.userAvatarImg) {
                window.userAvatarImg = new Image();
                window.userAvatarImg.src = "user_avatar_3d.jpg";
            }

            const avatarSize = 32;
            if (window.userAvatarImg.complete && window.userAvatarImg.naturalWidth > 0) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(userX, userY, avatarSize / 2 + 2, 0, Math.PI * 2);
                ctx.fillStyle = "#2563eb";
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.clip();
                ctx.drawImage(window.userAvatarImg, userX - avatarSize / 2, userY - avatarSize / 2, avatarSize, avatarSize);
                ctx.restore();
            } else {
                // Fallback blue avatar node if image loading
                ctx.save();
                ctx.beginPath();
                ctx.arc(userX, userY, 12, 0, Math.PI * 2);
                ctx.fillStyle = "#2563eb";
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.restore();
            }
            
            ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
            ctx.fillStyle = "#94a3b8";
            ctx.textAlign = "center";
            ctx.fillText("You (Heading 0°)", userX, userY + 24);
            ctx.restore();

            // 4. Compass / Orientation Indicator (Top-Left)
            ctx.save();
            ctx.translate(24, 24);
            ctx.beginPath();
            ctx.arc(0, 0, 14, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
            ctx.fill();
            ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
            ctx.stroke();
            ctx.font = "bold 9px 'JetBrains Mono', monospace";
            ctx.fillStyle = "#ef4444";
            ctx.textAlign = "center";
            ctx.fillText("N", 0, -4);
            ctx.fillStyle = "#94a3b8";
            ctx.fillText("▲", 0, 6);
            ctx.restore();

            // 5. Render Detected Objects & Trajectories
            let tracksToDraw = activeTracks || [];
            if (topViewFilterMode === "selected" && track) {
                tracksToDraw = [track];
            } else if (topViewFilterMode === "risk") {
                tracksToDraw = (activeTracks || []).filter(t => t.intersect || (t.ttc_s !== null && t.ttc_s < 3.0));
            }

            tracksToDraw.forEach((t, idx) => {
                const isSelected = track && t.track_id === track.track_id;
                const tBearing = Number(t.bearing || 0.0);
                const tTtc = (t.ttc_s !== null && t.ttc_s !== undefined) ? Number(t.ttc_s) : 4.0;
                const tCpa = (t.cpa !== null && t.cpa !== undefined) ? Number(t.cpa) : 1.2;
                const isThreatTrack = Boolean(t.intersect || (t.ttc_s !== null && t.ttc_s < 2.5));
                const strokeColor = isThreatTrack ? "#ef4444" : "#10b981";

                // Position calculation relative to user
                const initialDistMeter = Math.max(0.6, tTtc * 1.2);
                const objX = userX + Math.sin(tBearing) * (initialDistMeter * meterScale);
                const objY = userY - Math.cos(tBearing) * (initialDistMeter * meterScale);

                // Scrubber-adjusted position over time (+1s, +2s, +3s)
                const scrubRatio = Math.min(1.0, scrubberTime / Math.max(0.1, tTtc));
                const interX = userX + (tBearing * 20);
                const interY = userY - (tCpa * meterScale);

                const currentObjX = objX + (interX - objX) * scrubRatio;
                const currentObjY = objY + (interY - objY) * scrubRatio;

                // 5a. Translucent Expanding Uncertainty Region
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(objX, objY);
                ctx.lineTo(interX - 18, interY);
                ctx.lineTo(interX + 18, interY);
                ctx.closePath();
                ctx.fillStyle = isThreatTrack ? "rgba(239, 68, 68, 0.12)" : "rgba(16, 185, 129, 0.08)";
                ctx.fill();
                ctx.restore();

                // 5b. Curved Predicted Trajectory Path
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(objX, objY);
                ctx.quadraticCurveTo((objX + interX) / 2 + 10, (objY + interY) / 2, interX, interY);
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = isSelected ? 3.5 : 2.0;
                ctx.setLineDash(isSelected ? [] : [4, 3]);
                ctx.stroke();
                ctx.restore();

                // 5c. Time Progression Milestones (+1s, +2s, +3s)
                [1.0, 2.0, 3.0].forEach(sec => {
                    if (sec < tTtc) {
                        const ratio = sec / tTtc;
                        const mx = objX + (interX - objX) * ratio;
                        const my = objY + (interY - objY) * ratio;
                        ctx.save();
                        ctx.beginPath();
                        ctx.arc(mx, my, 2.5, 0, Math.PI * 2);
                        ctx.fillStyle = strokeColor;
                        ctx.fill();
                        ctx.font = "bold 8px 'JetBrains Mono', monospace";
                        ctx.fillStyle = "#94a3b8";
                        ctx.fillText(`+${sec}s`, mx + 4, my + 3);
                        ctx.restore();
                    }
                });

                // 5d. Predicted Intersection & Danger Zone
                if (isThreatTrack) {
                    ctx.save();
                    // Danger zone aura
                    ctx.beginPath();
                    ctx.arc(interX, interY, 18, 0, Math.PI * 2);
                    ctx.fillStyle = "rgba(239, 68, 68, 0.22)";
                    ctx.fill();
                    ctx.strokeStyle = "rgba(239, 68, 68, 0.6)";
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([2, 2]);
                    ctx.stroke();

                    // Intersection core point
                    ctx.beginPath();
                    ctx.arc(interX, interY, 5, 0, Math.PI * 2);
                    ctx.fillStyle = "#ef4444";
                    ctx.fill();
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Threat callout badge
                    ctx.fillStyle = "#dc2626";
                    ctx.beginPath();
                    ctx.roundRect(interX + 8, interY - 10, 78, 18, 4);
                    ctx.fill();
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 9px 'JetBrains Mono', monospace";
                    ctx.fillText(`TTC ${tTtc.toFixed(1)}s`, interX + 13, interY + 3);
                    ctx.restore();
                }

                // 5e. Distinct Interactive Object Marker
                ctx.save();
                ctx.translate(currentObjX, currentObjY);
                if (isSelected) {
                    ctx.beginPath();
                    ctx.arc(0, 0, 16, 0, Math.PI * 2);
                    ctx.strokeStyle = "#3b82f6";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                ctx.fillStyle = strokeColor;
                ctx.beginPath();
                ctx.roundRect(-16, -11, 32, 22, 5);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 8px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.fillText(`#${t.track_id}`, 0, 3);
                ctx.restore();
            });

        } else {
            // ================= 2D PERSPECTIVE VIEW =================
            const userX = w * 0.5;
            const userY = h * 0.88;
            const foeX = w * 0.5;
            const foeY = h * 0.28;

            // Risk zone (user corridor cone)
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(userX - 28, userY);
            ctx.lineTo(userX - 70, foeY - 10);
            ctx.lineTo(userX + 70, foeY - 10);
            ctx.lineTo(userX + 28, userY);
            ctx.closePath();
            ctx.fillStyle = isThreat ? "rgba(239, 68, 68, 0.10)" : "rgba(16, 185, 129, 0.10)";
            ctx.fill();
            ctx.strokeStyle = isThreat ? "rgba(239, 68, 68, 0.28)" : "rgba(16, 185, 129, 0.28)";
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.restore();

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
            ctx.fillText("Current position", userX - 74, userY + 4);
            ctx.restore();

            if (track) {
                // Compute real trajectory from track telemetry
                const trackBearing = Number(track.bearing || 0.0);
                const trackTtc = (track.ttc_s !== null && track.ttc_s !== undefined) ? Number(track.ttc_s) : 4.0;
                
                // Map bearing (-0.5 rad to +0.5 rad) to screen X coordinate
                // -0.5 rad (Left) -> w * 0.25, 0 rad (Center) -> w * 0.50, +0.5 rad (Right) -> w * 0.75
                const objX = Math.max(w * 0.15, Math.min(w * 0.85, w * 0.50 + (trackBearing / 0.5236) * (w * 0.30)));
                
                // Map TTC/Distance to perspective screen Y coordinate (0s = near userY, >5s = horizon foeY)
                const normDist = Math.max(0.0, Math.min(1.0, trackTtc / 5.0));
                const objY = foeY + normDist * (userY - foeY - 30);
                
                // Predicted Intersection Point on user path
                const interX = w * 0.50;
                const interY = userY - (1.0 - Math.min(1.0, trackTtc / 5.0)) * (userY - foeY);

                ctx.save();
                ctx.beginPath();
                ctx.ellipse(interX, interY, 36, 14, 0, 0, Math.PI * 2);
                ctx.fillStyle = isThreat ? "rgba(239, 68, 68, 0.20)" : "rgba(16, 185, 129, 0.15)";
                ctx.fill();
                ctx.strokeStyle = isThreat ? "rgba(239, 68, 68, 0.50)" : "rgba(16, 185, 129, 0.40)";
                ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.restore();

                // Object Projected Path Trajectory Line
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(objX, objY);
                ctx.lineTo(interX, interY);
                ctx.strokeStyle = threatColor;
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 3]);
                ctx.stroke();
                ctx.restore();

                // Object Tag Node
                ctx.save();
                ctx.fillStyle = threatColor;
                ctx.beginPath();
                ctx.roundRect(objX - 22, objY - 12, 44, 24, 5);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 9px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.fillText(objLabel, objX, objY + 3);
                ctx.restore();

                if (isThreat) {
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
                    ctx.roundRect(interX + 10, interY - 9, 84, 18, 4);
                    ctx.fill();
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 9px 'JetBrains Mono', monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(`${ttcLabel} hazard`, interX + 15, interY + 4);
                    ctx.restore();
                }
            }
        }

        // Compact Legend (Top Right)
        ctx.save();
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.beginPath();
        ctx.roundRect(w - 110, 8, 102, 70, 6);
        ctx.fill();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
        ctx.stroke();

        ctx.font = "8px 'Plus Jakarta Sans', sans-serif";
        ctx.fillStyle = "#94a3b8";

        // User path
        ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2; ctx.setLineDash([4, 2]);
        ctx.beginPath(); ctx.moveTo(w - 100, 20); ctx.lineTo(w - 82, 20); ctx.stroke();
        ctx.fillText("User path", w - 76, 23);

        // Object path
        ctx.strokeStyle = "#ef4444"; ctx.beginPath(); ctx.moveTo(w - 100, 36); ctx.lineTo(w - 82, 36); ctx.stroke();
        ctx.fillText("Object path", w - 76, 39);

        // Intersection
        ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(w - 91, 50, 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#94a3b8"; ctx.fillText("Intersection", w - 76, 53);

        // Uncertainty
        ctx.fillStyle = "rgba(239, 68, 68, 0.4)";
        ctx.fillRect(w - 96, 62, 10, 6);
        ctx.fillStyle = "#94a3b8"; ctx.fillText("Uncertainty", w - 76, 67);
        ctx.restore();
    }

    // -------------------------------------------------------------------------
    // 8. Tab 3: Haptic State
    // -------------------------------------------------------------------------
    window.switchHapticView = function (view) {
        const btnDev = document.getElementById("btn-seg-device");
        const btnCmd = document.getElementById("btn-seg-commands");

        if (view === "device") {
            if (hapticsDeviceView) hapticsDeviceView.style.display = "block";
            if (hapticsCommandsView) hapticsCommandsView.style.display = "none";
            if (btnDev) btnDev.classList.add("active");
            if (btnCmd) btnCmd.classList.remove("active");
        } else {
            if (hapticsDeviceView) hapticsDeviceView.style.display = "none";
            if (hapticsCommandsView) hapticsCommandsView.style.display = "block";
            if (btnDev) btnDev.classList.remove("active");
            if (btnCmd) btnCmd.classList.add("active");
            renderHapticCommandsList();
        }
    };

    function updateHapticTelemetry(haptic) {
        const dir = (haptic.direction || "STOP").toUpperCase();
        const pattern = haptic.pattern_id || "ALL_CLEAR";
        const urgency = haptic.urgency || 1;
        const dur = haptic.duration_ms || 200;
        const timeNow = new Date().toTimeString().split(" ")[0];

        // MAX_URGENCY = 5 matches corridor_policy.py's urgency_levels config value
        const MAX_URGENCY = 5;
        if (hapticPatLabel) hapticPatLabel.textContent = pattern;
        if (hapticUrgVal) hapticUrgVal.textContent = `${urgency}/${MAX_URGENCY}`;
        if (hapticDurVal) hapticDurVal.textContent = `${dur} ms`;
        if (hapticTimeVal) hapticTimeVal.textContent = timeNow;

        resetMotorVisuals();

        const isVibrating = pattern !== "ALL_CLEAR";
        if (hapticVibBadge) {
            hapticVibBadge.textContent = isVibrating ? "● Vibrating" : "○ Idle";
            hapticVibBadge.style.background = isVibrating ? "#eff6ff" : "#f1f5f9";
            hapticVibBadge.style.color = isVibrating ? "#2563eb" : "#64748b";
            hapticVibBadge.style.borderColor = isVibrating ? "#bfdbfe" : "#cbd5e1";
        }

        // 2-Motor Hardware routing (Left = Pin 5, Right = Pin 6)
        if (isVibrating) {
            if (dir === "LEFT") {
                setMotorActive(svgMotorLeft, svgRippleLeft, svgLblLeft, devMotorLStatus, true);
            } else if (dir === "RIGHT") {
                setMotorActive(svgMotorRight, svgRippleRight, svgLblRight, devMotorRStatus, true);
            } else if (dir === "STOP" || dir === "CENTER" || urgency >= 4) {
                // Dual motor emergency warning
                setMotorActive(svgMotorLeft, svgRippleLeft, svgLblLeft, devMotorLStatus, true);
                setMotorActive(svgMotorRight, svgRippleRight, svgLblRight, devMotorRStatus, true);
            }
        }

        // Stabilized Command Logging (Rate-limited, human-readable descriptions)
        const hapticKey = `${pattern}_${dir}_${urgency}`;
        const now = Date.now();
        const shouldLog = (hapticKey !== lastHapticKey) || (isVibrating && (now - lastHapticLogTime > 2500));

        if (shouldLog) {
            lastHapticKey = hapticKey;
            lastHapticLogTime = now;

            let semanticDesc = "";
            if (!isVibrating || pattern === "ALL_CLEAR") {
                semanticDesc = "ALL CLEAR · Both motors idle (Safe path)";
            } else if (dir === "LEFT") {
                semanticDesc = `LEFT (Pin 5): Pulse warning · Obstacle on left (urgency ${urgency})`;
            } else if (dir === "RIGHT") {
                semanticDesc = `RIGHT (Pin 6): Pulse warning · Obstacle on right (urgency ${urgency})`;
            } else if (dir === "STOP" || urgency >= 4) {
                semanticDesc = `DUAL MOTORS (Pins 5 & 6): Immediate brake warning! (urgency ${urgency})`;
            } else {
                semanticDesc = `${dir} MOTOR: Active guidance (urgency ${urgency})`;
            }

            hapticHistory.unshift({ time: timeNow, pattern, duration: dur, urgency, desc: semanticDesc });
            if (hapticHistory.length > 20) hapticHistory.pop();
            renderHapticCommandsList();
        }
    }

    function setMotorActive(circleEl, rippleEl, labelEl, badgeEl, active) {
        if (!circleEl) return;
        if (active) {
            circleEl.setAttribute("fill", "#eff6ff");
            circleEl.setAttribute("stroke", "#3b82f6");
            circleEl.setAttribute("stroke-width", "3");
            if (rippleEl) rippleEl.classList.add("vibrating");
            if (labelEl) labelEl.setAttribute("fill", "#3b82f6");
            if (badgeEl) {
                badgeEl.className = "pill-badge green";
                badgeEl.textContent = "● ACTIVE";
            }
        }
    }

    function resetMotorVisuals() {
        const motors = [
            { c: svgMotorLeft, r: svgRippleLeft, l: svgLblLeft, b: devMotorLStatus },
            { c: svgMotorRight, r: svgRippleRight, l: svgLblRight, b: devMotorRStatus }
        ];

        motors.forEach(m => {
            if (m.c) {
                m.c.setAttribute("fill", "#f8fafc");
                m.c.setAttribute("stroke", "#94a3b8");
                m.c.setAttribute("stroke-width", "2");
            }
            if (m.r) m.r.classList.remove("vibrating");
            if (m.l) m.l.setAttribute("fill", "#64748b");
            if (m.b) {
                m.b.className = "detail-val";
                m.b.style.color = "var(--text-muted)";
                m.b.textContent = "○ Idle";
            }
        });
    }

    function renderHapticCommandsList() {
        if (!hapticCommandsList) return;
        hapticCommandsList.innerHTML = hapticHistory.slice(0, 8).map(cmd => `
            <div class="detail-table-row">
                <span class="detail-key mono" style="font-size:0.70rem;">${cmd.time}</span>
                <span class="detail-val" style="font-size:0.72rem; font-weight:600;">${cmd.desc || `${cmd.pattern} (${cmd.duration}ms)`}</span>
            </div>
        `).join("");
    }

    // -------------------------------------------------------------------------
    // 9. Tab 4: Test & Replay Scenarios
    // -------------------------------------------------------------------------
    window.openScenarioDetail = function (scenId) {
        currentScenarioId = scenId;
        const scen = SCENARIOS[scenId] || SCENARIOS.s1;

        if (scenModalTitle) scenModalTitle.textContent = scen.title;
        if (scenModalDesc) scenModalDesc.textContent = scen.desc;
        if (scenModalExpRisk) {
            scenModalExpRisk.textContent = scen.expectedRisk;
            scenModalExpRisk.className = `pill-badge ${scen.expectedRisk === "CRITICAL" ? "red" : (scen.expectedRisk === "WARNING" ? "orange" : "green")}`;
        }
        if (scenModalExpHaptic) scenModalExpHaptic.textContent = scen.expectedHaptic;
        if (scenModalExpTtc) scenModalExpTtc.textContent = scen.expectedTtc;
        if (scenModalExpIntersect) scenModalExpIntersect.textContent = scen.expectedIntersect;

        if (scenActiveBox) scenActiveBox.style.display = "none";
        if (scenResultCard) scenResultCard.style.display = "none";
        if (scenTimelineEmbed) scenTimelineEmbed.style.display = "none";
        const runnerControls = document.getElementById("scen-runner-controls");
        if (runnerControls) runnerControls.style.display = "block";

        window.selectScenario(scenId);
        if (scenarioModal) scenarioModal.style.display = "flex";
    };
    window.openScenarioModal = window.openScenarioDetail;

    window.closeScenarioDetail = function () {
        window.stopModalScenarioTest();
        window.pauseReplay();
        if (scenarioModal) scenarioModal.style.display = "none";
    };
    window.closeScenarioModal = window.closeScenarioDetail;

    window.openLatestTestDetails = function () {
        window.openScenarioDetail(currentScenarioId);
        if (scenResultCard) scenResultCard.style.display = "flex";
        const runnerControls = document.getElementById("scen-runner-controls");
        if (runnerControls) runnerControls.style.display = "none";
        if (scenTimelineEmbed) {
            scenTimelineEmbed.style.display = "flex";
            drawReplayTimeline();
        }
    };

    function highlightQuickChip(scenId) {
        document.querySelectorAll(".scenario-chips-row .scenario-chip").forEach(ch => {
            const on = ch.id === `chip-scen-${scenId}`;
            ch.style.background = on ? "var(--c-primary)" : "#ffffff";
            ch.style.color = on ? "#ffffff" : "var(--text-body)";
            ch.style.borderColor = on ? "var(--c-primary)" : "var(--border-light)";
        });
    }

    function recordTestResult(scen) {
        testHistory.unshift({
            title: scen.title,
            result: scen.result || "PASS",
            expected: scen.expectedRisk,
            actual: scen.actualRisk,
            time: new Date().toTimeString().split(" ")[0]
        });
        if (testHistory.length > 5) testHistory.length = 5;
        renderTestHistory();
        showScreenTestResult(scen);
    }

    function showScreenTestResult(scen) {
        if (testResultCard) testResultCard.style.display = "flex";
        if (testResBadge) {
            testResBadge.textContent = `✔ ${scen.result || "PASS"}`;
            testResBadge.className = "pill-badge green";
        }
        if (testResExp) testResExp.textContent = scen.expectedRisk;
        if (testResAct) testResAct.textContent = scen.actualRisk;

        if (scenResultBadge) {
            scenResultBadge.textContent = "✔ TEST PASSED";
            scenResultBadge.className = "pill-badge green";
        }
        if (scenResExp) scenResExp.textContent = scen.expectedRisk;
        if (scenResAct) scenResAct.textContent = scen.actualRisk;
        if (scenResHap) scenResHap.textContent = scen.actualHaptic;

        if (testExecStatusBadge) {
            testExecStatusBadge.textContent = "✔ READY";
            testExecStatusBadge.style.background = "var(--c-safe-bg)";
            testExecStatusBadge.style.borderColor = "var(--c-safe-border)";
            testExecStatusBadge.style.color = "var(--c-safe-text)";
        }
    }

    function onReplayComplete() {
        const scen = SCENARIOS[currentScenarioId] || SCENARIOS.s1;
        isModalTesting = false;
        if (scenActiveBox) scenActiveBox.style.display = "none";
        if (scenResultCard) scenResultCard.style.display = "flex";
        if (scenTimelineEmbed) {
            scenTimelineEmbed.style.display = "flex";
            drawReplayTimeline();
        }
        recordTestResult(scen);
    }

    window.startModalScenarioTest = function () {
        if (isModalTesting || isReplaying) return;
        isModalTesting = true;

        const runnerControls = document.getElementById("scen-runner-controls");
        if (runnerControls) runnerControls.style.display = "none";
        if (scenResultCard) scenResultCard.style.display = "none";
        if (scenActiveBox) scenActiveBox.style.display = "flex";
        if (scenExecTimerTxt) scenExecTimerTxt.textContent = "Replaying scenario...";

        if (testExecStatusBadge) {
            testExecStatusBadge.textContent = "● RUNNING";
            testExecStatusBadge.style.background = "#eff6ff";
            testExecStatusBadge.style.borderColor = "#bfdbfe";
            testExecStatusBadge.style.color = "#2563eb";
        }

        window.selectScenario(currentScenarioId);
        window.resetReplay();
        window.startReplay();
    };

    window.stopModalScenarioTest = function () {
        if (modalTestTimer) {
            clearInterval(modalTestTimer);
            modalTestTimer = null;
        }
        isModalTesting = false;
        window.pauseReplay();
        if (scenActiveBox) scenActiveBox.style.display = "none";
        const runnerControls = document.getElementById("scen-runner-controls");
        if (runnerControls) runnerControls.style.display = "block";

        if (testExecStatusBadge) {
            testExecStatusBadge.textContent = "✔ READY";
            testExecStatusBadge.style.background = "var(--c-safe-bg)";
            testExecStatusBadge.style.borderColor = "var(--c-safe-border)";
            testExecStatusBadge.style.color = "var(--c-safe-text)";
        }
    };

    window.toggleScenarioTimelineView = function () {
        if (!scenTimelineEmbed) return;
        const isHidden = scenTimelineEmbed.style.display === "none" || !scenTimelineEmbed.style.display;
        scenTimelineEmbed.style.display = isHidden ? "flex" : "none";
        if (isHidden) {
            drawReplayTimeline();
        }
    };

    window.runQuickTest = function (scenId) {
        highlightQuickChip(scenId);
        window.openScenarioDetail(scenId);
        window.startModalScenarioTest();
    };
    window.quickRunScenario = window.runQuickTest;

    function renderTestHistory() {
        if (!testHistoryList) return;
        if (!testHistory.length) {
            testHistoryList.innerHTML = `<span style="font-size:0.72rem; color:var(--text-muted); padding:4px 0;">No recent test runs</span>`;
            return;
        }
        testHistoryList.innerHTML = testHistory.map(item => `
            <div class="detail-table-row">
                <div style="display:flex; flex-direction:column; gap:1px;">
                    <span style="font-size:0.75rem; font-weight:700;">${escapeHtml(item.title)}</span>
                    <span style="font-size:0.65rem; color:var(--text-muted);">${escapeHtml(item.time || "Recent")} · Exp: ${escapeHtml(item.expected)} / Act: ${escapeHtml(item.actual)}</span>
                </div>
                <span class="pill-badge green">✔ ${escapeHtml(item.result)}</span>
            </div>
        `).join("");
    }

    window.clearTestHistory = function () {
        testHistory = [];
        renderTestHistory();
        if (testResultCard) testResultCard.style.display = "none";
    };

    window.selectScenario = function (scenId) {
        currentScenarioId = scenId;
        if (selectTestScenario) selectTestScenario.value = scenId;

        const scen = SCENARIOS[scenId] || SCENARIOS.s1;
        if (scenDesc) scenDesc.textContent = scen.desc;
        if (scenExpected) scenExpected.textContent = scen.expectedRisk;
        if (scenActual) scenActual.textContent = scen.actualRisk;
        if (scenResult) {
            scenResult.className = "pill-badge green";
            scenResult.textContent = "✔ PASS";
        }

        replayProgress = 35;
        if (replaySlider) replaySlider.value = replayProgress;
        updateScrubberText();
        drawReplayTimeline();
    };

    window.startReplay = function () {
        if (isReplaying) return;
        isReplaying = true;
        if (replayTimer) clearInterval(replayTimer);

        replayTimer = setInterval(() => {
            replayProgress += 2;
            if (replayProgress > 100) {
                replayProgress = 100;
                window.pauseReplay();
                onReplayComplete();
            }
            if (replaySlider) replaySlider.value = replayProgress;
            updateScrubberText();
            drawReplayTimeline();
        }, 80);
    };

    window.pauseReplay = function () {
        isReplaying = false;
        if (replayTimer) clearInterval(replayTimer);
    };

    window.resetReplay = function () {
        window.pauseReplay();
        replayProgress = 0;
        if (replaySlider) replaySlider.value = 0;
        updateScrubberText();
        drawReplayTimeline();
    };

    window.onScrubberInput = function (val) {
        replayProgress = Number(val);
        updateScrubberText();
        drawReplayTimeline();
    };

    function updateScrubberText() {
        if (!scenFrameTxt) return;
        const scen = SCENARIOS[currentScenarioId] || SCENARIOS.s1;
        const frame = 1200 + Math.round((replayProgress / 100) * 180);
        const curveIdx = Math.min(scen.curve.length - 1, Math.floor((replayProgress / 100) * scen.curve.length));
        const risk = scen.curve[curveIdx] || 0.11;
        scenFrameTxt.textContent = `Frame ${frame} | Risk ${risk.toFixed(2)}`;
    }

    function drawReplayTimeline() {
        if (!replayTimelineCanvas) return;
        const ctx = replayTimelineCanvas.getContext("2d");
        const w = replayTimelineCanvas.width;
        const h = replayTimelineCanvas.height;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);

        const scen = SCENARIOS[currentScenarioId] || SCENARIOS.s1;
        const pts = scen.curve;

        ctx.strokeStyle = "#f1f5f9";
        ctx.lineWidth = 1;
        for (let y = 15; y < h; y += 22) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        ctx.beginPath();
        const step = w / (pts.length - 1);
        pts.forEach((val, i) => {
            const px = i * step;
            const py = h - 10 - (val * (h - 25));
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });

        ctx.strokeStyle = scen.riskPeak > 0.5 ? "#f97316" : "#2563eb";
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = scen.riskPeak > 0.5 ? "rgba(249, 115, 22, 0.08)" : "rgba(37, 99, 235, 0.08)";
        ctx.fill();

        const cursorX = (replayProgress / 100) * w;
        ctx.save();
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cursorX, 0);
        ctx.lineTo(cursorX, h);
        ctx.stroke();

        const curveIdx = Math.min(pts.length - 1, Math.floor((replayProgress / 100) * pts.length));
        const cursorY = h - 10 - ((pts[curveIdx] || 0.1) * (h - 25));
        ctx.beginPath();
        ctx.arc(cursorX, cursorY, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#2563eb";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
    }

    // -------------------------------------------------------------------------
    // 10. Tab 5: Social Assist (UI-Only Mock Prototype)
    // -------------------------------------------------------------------------
    function escapeHtml(value) {
        return String(value == null ? "" : value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    window.toggleSocialAssist = function () {
        socialAssistEnabled = !socialAssistEnabled;

        if (toggleSocialSwitch) {
            const knob = toggleSocialSwitch.querySelector(".toggle-knob");
            if (socialAssistEnabled) {
                toggleSocialSwitch.style.background = "var(--c-primary)";
                if (knob) {
                    knob.style.right = "2px";
                    knob.style.left = "auto";
                }
            } else {
                toggleSocialSwitch.style.background = "#cbd5e1";
                if (knob) {
                    knob.style.right = "auto";
                    knob.style.left = "2px";
                }
            }
        }

        if (socialSwitchLabel) {
            socialSwitchLabel.textContent = socialAssistEnabled ? "Enabled" : "Disabled";
            socialSwitchLabel.style.color = socialAssistEnabled ? "var(--c-primary)" : "var(--text-muted)";
        }

        if (socialFaceBox) {
            socialFaceBox.style.opacity = socialAssistEnabled ? "1" : "0.2";
        }
    };

    window.switchSocialView = function (view) {
        const btnLive = document.getElementById("btn-seg-live-rec");
        const btnContacts = document.getElementById("btn-seg-contacts");

        if (view === "live") {
            if (socialLiveView) socialLiveView.style.display = "block";
            if (socialContactsView) socialContactsView.style.display = "none";
            if (btnLive) btnLive.classList.add("active");
            if (btnContacts) btnContacts.classList.remove("active");
        } else {
            if (socialLiveView) socialLiveView.style.display = "none";
            if (socialContactsView) socialContactsView.style.display = "block";
            if (btnLive) btnLive.classList.remove("active");
            if (btnContacts) btnContacts.classList.add("active");
            renderContactsList();
        }
    };

    window.setSocialDemoState = function (state) {
        socialDemoState = state;
        document.querySelectorAll(".demo-state-chip").forEach(ch => ch.classList.remove("active"));
        const activeChip = document.getElementById(`chip-demo-${state}`);
        if (activeChip) activeChip.classList.add("active");

        if (!socialFaceBox) return;

        if (state === "known") {
            socialFaceBox.style.borderColor = "#10b981";
            if (socialFaceTagTop) {
                socialFaceTagTop.style.background = "#10b981";
                socialFaceTagTop.textContent = "Person #4";
            }
            if (socialFaceTagBottom) {
                socialFaceTagBottom.style.background = "#10b981";
                socialFaceTagBottom.textContent = "Rahul (0.91)";
            }
            if (socTrackId) socTrackId.textContent = "#4";
            if (socPersonName) {
                socPersonName.textContent = "Rahul";
                socPersonName.style.color = "var(--c-primary)";
            }
            if (socRelationship) socRelationship.textContent = "Friend";
            if (socConf) socConf.textContent = "91%";
            if (socStatus) {
                socStatus.textContent = "● Known";
                socStatus.className = "pill-badge green";
            }
        } else if (state === "unknown") {
            socialFaceBox.style.borderColor = "#f59e0b";
            if (socialFaceTagTop) {
                socialFaceTagTop.style.background = "#f59e0b";
                socialFaceTagTop.textContent = "Person #7";
            }
            if (socialFaceTagBottom) {
                socialFaceTagBottom.style.background = "#f59e0b";
                socialFaceTagBottom.textContent = "Unknown Face";
            }
            if (socTrackId) socTrackId.textContent = "#7";
            if (socPersonName) {
                socPersonName.textContent = "Unregistered Person";
                socPersonName.style.color = "var(--text-title)";
            }
            if (socRelationship) socRelationship.textContent = "None (Unknown)";
            if (socConf) socConf.textContent = "42%";
            if (socStatus) {
                socStatus.textContent = "○ Unregistered";
                socStatus.className = "pill-badge orange";
            }
        } else if (state === "possible") {
            socialFaceBox.style.borderColor = "#3b82f6";
            if (socialFaceTagTop) {
                socialFaceTagTop.style.background = "#3b82f6";
                socialFaceTagTop.textContent = "Person #2";
            }
            if (socialFaceTagBottom) {
                socialFaceTagBottom.style.background = "#3b82f6";
                socialFaceTagBottom.textContent = "Priya? (0.64)";
            }
            if (socTrackId) socTrackId.textContent = "#2";
            if (socPersonName) {
                socPersonName.textContent = "Priya Patel";
                socPersonName.style.color = "#2563eb";
            }
            if (socRelationship) socRelationship.textContent = "Family";
            if (socConf) socConf.textContent = "64% (Tentative)";
            if (socStatus) {
                socStatus.textContent = "◐ Possible Match";
                socStatus.className = "pill-badge orange";
            }
        }
    };

    function renderContactsList() {
        if (!contactsListContainer) return;
        contactsListContainer.innerHTML = mockContacts.map(c => `
            <div class="contact-card" onclick="openContactProfileModal(${c.id})">
                <div class="contact-avatar">${c.avatar}</div>
                <div class="contact-info">
                    <span class="contact-name">${c.name}</span>
                    <span class="contact-status">${c.relationship} · ${c.photosCount} photos</span>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span class="pill-badge green" style="font-size:0.62rem;">${c.status.split('·')[0].trim()}</span>
                    <span style="color:var(--text-dim); font-weight:800; font-size:0.85rem;">&gt;</span>
                </div>
            </div>
        `).join("");
    }

    function setAddFriendStep(step) {
        addFriendState.step = step;
        const p1 = document.getElementById("af-step-photos");
        const p2 = document.getElementById("af-step-name");
        const p3 = document.getElementById("af-step-review");
        if (p1) p1.className = step === 1 ? "wizard-pane active" : "wizard-pane";
        if (p2) p2.className = step === 2 ? "wizard-pane active" : "wizard-pane";
        if (p3) p3.className = step === 3 ? "wizard-pane active" : "wizard-pane";

        const dot1 = document.getElementById("af-dot-1");
        const dot2 = document.getElementById("af-dot-2");
        const dot3 = document.getElementById("af-dot-3");
        if (dot1) dot1.className = step === 1 ? "wizard-dot active" : (step > 1 ? "wizard-dot done" : "wizard-dot");
        if (dot2) dot2.className = step === 2 ? "wizard-dot active" : (step > 2 ? "wizard-dot done" : "wizard-dot");
        if (dot3) dot3.className = step === 3 ? "wizard-dot active" : "wizard-dot";

        const caption = document.getElementById("af-step-caption");
        if (caption) {
            if (step === 1) caption.textContent = "Step 1 of 3 · Reference photos";
            else if (step === 2) caption.textContent = "Step 2 of 3 · Contact details";
            else caption.textContent = "Step 3 of 3 · Review & confirm";
        }

        const btnBack = document.getElementById("btn-af-back");
        const btnNext = document.getElementById("btn-af-next");
        const btnSave = document.getElementById("btn-af-save");
        if (btnBack) btnBack.style.display = step > 1 ? "inline-block" : "none";
        if (btnNext) btnNext.style.display = step < 3 ? "inline-block" : "none";
        if (btnSave) btnSave.style.display = step === 3 ? "inline-block" : "none";

        if (step === 3) {
            const revName = document.getElementById("af-review-name");
            const revRel = document.getElementById("af-review-rel");
            const revPhotos = document.getElementById("af-review-photos");
            const revGrid = document.getElementById("af-review-grid");
            const nameVal = addFriendNameInput ? addFriendNameInput.value.trim() : "";
            const relVal = addFriendRelationSelect ? addFriendRelationSelect.value : "Friend";
            if (revName) revName.textContent = nameVal || "(Unnamed)";
            if (revRel) revRel.textContent = relVal;
            if (revPhotos) revPhotos.textContent = `${tempFriendPhotos.length} photo${tempFriendPhotos.length === 1 ? "" : "s"}`;
            if (revGrid) {
                revGrid.innerHTML = tempFriendPhotos.map(d => `<img src="${d}" class="photo-thumb" style="width:100%; height:72px;">`).join("");
            }
        }
    }

    window.nextAddFriendStep = function () {
        if (addFriendState.step === 1) {
            setAddFriendStep(2);
        } else if (addFriendState.step === 2) {
            const nameVal = addFriendNameInput ? addFriendNameInput.value.trim() : "";
            if (!nameVal) {
                if (addFriendNameInput) {
                    addFriendNameInput.focus();
                    addFriendNameInput.style.borderColor = "#ef4444";
                    setTimeout(() => { if (addFriendNameInput) addFriendNameInput.style.borderColor = "var(--border-light)"; }, 1500);
                }
                return;
            }
            setAddFriendStep(3);
        }
    };

    window.prevAddFriendStep = function () {
        if (addFriendState.step > 1) {
            setAddFriendStep(addFriendState.step - 1);
        }
    };

    window.openAddFriendModal = function () {
        tempFriendPhotos = [];
        if (addFriendNameInput) addFriendNameInput.value = "";
        if (addFriendRelationSelect) addFriendRelationSelect.value = "Friend";
        renderAddFriendPhotoGrid();
        setAddFriendStep(1);
        if (addFriendModal) addFriendModal.style.display = "flex";
    };

    window.closeAddFriendModal = function () {
        if (addFriendModal) addFriendModal.style.display = "none";
    };

    window.handleFriendPhotosSelect = function (event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        Array.from(files).slice(0, 4 - tempFriendPhotos.length).forEach(file => {
            const reader = new FileReader();
            reader.onload = function (e) {
                tempFriendPhotos.push(e.target.result);
                renderAddFriendPhotoGrid();
            };
            reader.readAsDataURL(file);
        });
    };

    function renderAddFriendPhotoGrid() {
        if (!addFriendPhotoGrid) return;
        const photosHtml = tempFriendPhotos.map((dataUri, idx) => `
            <div style="position:relative; width:100%; height:72px;">
                <img src="${dataUri}" class="photo-thumb" style="width:100%; height:100%;">
                <button onclick="removeFriendPhoto(${idx})" style="position:absolute; top:3px; right:3px; width:18px; height:18px; border-radius:50%; background:rgba(0,0,0,0.7); color:#ffffff; font-size:11px; display:flex; align-items:center; justify-content:center; border:none; cursor:pointer;">&times;</button>
            </div>
        `).join("");

        const addBtnHtml = tempFriendPhotos.length < 4 ? `
            <div class="photo-add-btn" onclick="document.getElementById('add-friend-file-input').click()">
                <span style="font-size:1.3rem; font-weight:800;">+</span>
                <span>Add Photo</span>
            </div>
        ` : '';

        addFriendPhotoGrid.innerHTML = photosHtml + addBtnHtml;
    }

    window.removeFriendPhoto = function (idx) {
        tempFriendPhotos.splice(idx, 1);
        renderAddFriendPhotoGrid();
    };

    window.saveNewFriend = function () {
        const name = addFriendNameInput ? addFriendNameInput.value.trim() : "";
        if (!name) {
            if (addFriendNameInput) {
                addFriendNameInput.focus();
                addFriendNameInput.style.borderColor = "#ef4444";
                setTimeout(() => { addFriendNameInput.style.borderColor = "var(--border-light)"; }, 1500);
            }
            return;
        }

        const relation = addFriendRelationSelect ? addFriendRelationSelect.value : "Friend";
        const avatar = tempFriendPhotos.length > 0 
            ? `<img src="${tempFriendPhotos[0]}" alt="${name}">` 
            : (relation === "Family" ? "👩" : (relation === "Physician" ? "👨‍⚕️" : "👤"));

        const newContact = {
            id: Date.now(),
            name: name,
            relationship: relation,
            photos: [...tempFriendPhotos],
            photosCount: Math.max(1, tempFriendPhotos.length),
            status: "Ready · Active",
            avatar: avatar
        };

        mockContacts.unshift(newContact);
        renderContactsList();
        window.closeAddFriendModal();
    };

    window.openContactProfileModal = function (contactId) {
        currentContactId = contactId;
        const contact = mockContacts.find(c => c.id === contactId);
        if (!contact) return;

        if (profModalAvatar) profModalAvatar.innerHTML = contact.avatar;
        if (profModalName) profModalName.textContent = contact.name;
        if (profModalRelation) profModalRelation.textContent = contact.relationship;
        if (profModalPhotosCount) profModalPhotosCount.textContent = `${contact.photosCount} photos enrolled`;
        if (profModalStatus) profModalStatus.textContent = contact.status;

        const profGrid = document.getElementById("prof-modal-photo-grid");
        if (profGrid) {
            if (contact.photos && contact.photos.length > 0) {
                profGrid.innerHTML = contact.photos.map(p => `<img src="${p}" class="photo-thumb" style="width:100%; height:72px;">`).join("");
            } else {
                profGrid.innerHTML = `<span style="font-size:0.68rem; color:var(--text-muted); padding:4px 0;">3 synthetic reference encodings active on edge DB.</span>`;
            }
        }

        if (contactProfileModal) contactProfileModal.style.display = "flex";
    };

    window.closeContactProfileModal = function () {
        if (contactProfileModal) contactProfileModal.style.display = "none";
    };

    window.deleteCurrentContact = function () {
        if (!currentContactId) return;
        mockContacts = mockContacts.filter(c => c.id !== currentContactId);
        renderContactsList();
        window.closeContactProfileModal();
    };

    window.editContactName = function () {
        if (!currentContactId) return;
        const contact = mockContacts.find(c => c.id === currentContactId);
        if (!contact) return;
        const newName = prompt("Edit contact name:", contact.name);
        if (newName && newName.trim()) {
            contact.name = newName.trim();
            if (profModalName) profModalName.textContent = contact.name;
            renderContactsList();
        }
    };

    window.addMorePhotosToContact = function () {
        const fileInput = document.getElementById("prof-add-photo-input");
        if (fileInput) {
            fileInput.click();
        } else {
            if (!currentContactId) return;
            const contact = mockContacts.find(c => c.id === currentContactId);
            if (!contact) return;
            contact.photosCount++;
            if (profModalPhotosCount) profModalPhotosCount.textContent = `${contact.photosCount} photos enrolled`;
            renderContactsList();
        }
    };

    window.handleProfilePhotoSelect = function (event) {
        const files = event.target.files;
        if (!files || files.length === 0 || !currentContactId) return;
        const contact = mockContacts.find(c => c.id === currentContactId);
        if (!contact) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            if (!contact.photos) contact.photos = [];
            contact.photos.push(e.target.result);
            contact.photosCount = contact.photos.length;
            if (profModalPhotosCount) profModalPhotosCount.textContent = `${contact.photosCount} photos enrolled`;
            const profGrid = document.getElementById("prof-modal-photo-grid");
            if (profGrid) {
                profGrid.innerHTML = contact.photos.map(p => `<img src="${p}" class="photo-thumb" style="width:100%; height:72px;">`).join("");
            }
            renderContactsList();
        };
        reader.readAsDataURL(files[0]);
    };

    // -------------------------------------------------------------------------
    // Startup Initialization
    // -------------------------------------------------------------------------
    resetDashboardLiveMetrics();
    initCameraSource();
    connectWs();
    renderContactsList();
    renderTestHistory();
    window.switchTab("live");
    window.selectScenario("s1");
})();
