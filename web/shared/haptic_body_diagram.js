/**
 * SpatialVector-HMI — Shared Haptic Body Diagram & Motor Controller.
 *
 * Visualizes 3-motor haptic belt/vest (LEFT, CENTER, RIGHT) with vibration
 * ripples, urgency badges, and pattern interpretation.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.HapticBodyDiagram = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    "use strict";

    const PATTERN_DESCRIPTIONS = {
        ALL_CLEAR: "Safe navigation path; all motors idle.",
        DEGRADED_WARN: "Sensor dropout warning pulse; caution recommended.",
        STOP_CRITICAL: "Emergency Stop! Collision hazard detected in all forward paths.",
        LEFT_SLOW: "Low-urgency prompt: veer slightly left.",
        LEFT_MED: "Active guidance: move into the left corridor.",
        LEFT_FAST: "High-urgency alert: obstacle detected; steer left immediately.",
        CENTER_SLOW: "Low-urgency prompt: forward corridor clear.",
        CENTER_MED: "Active guidance: maintain centered heading.",
        CENTER_FAST: "High-urgency alert: obstacles flanking sides; steer center.",
        RIGHT_SLOW: "Low-urgency prompt: veer slightly right.",
        RIGHT_MED: "Active guidance: move into the right corridor.",
        RIGHT_FAST: "High-urgency alert: obstacle detected; steer right immediately.",
    };

    function getPatternDescription(patternId) {
        if (!patternId) return "All clear — corridor safe.";
        return PATTERN_DESCRIPTIONS[patternId] || `Active pattern: ${patternId}`;
    }

    class DiagramController {
        constructor(containerEl, options = {}) {
            this.containerEl = containerEl;
            this.maxUrgency = options.maxUrgency || 5;
            this.motorElements = {};
            this.render();
        }

        render() {
            if (!this.containerEl) return;
            this.containerEl.innerHTML = `
                <div class="haptic-body-wrap" style="position: relative; width: 100%; height: 180px; max-width: 320px; margin: 0 auto; overflow: hidden; border-radius: 12px; background: #f8fafc;">
                    <img src="haptic_vest_3d.jpg" alt="3D Human Haptic Vest" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.85; z-index: 1;" />
                    <svg viewBox="0 0 280 200" class="haptic-vest-svg" style="position: relative; z-index: 2; width: 100%; height: 100%;">
                        <!-- Left Motor (Pin 5) -->
                        <g id="haptic-motor-left" transform="translate(68, 115)">
                            <circle class="motor-ripple" r="26" fill="none" stroke="#3b82f6" stroke-width="2" opacity="0" />
                            <circle class="motor-base" r="18" fill="rgba(255, 255, 255, 0.9)" stroke="#3b82f6" stroke-width="3" />
                            <circle class="motor-core" r="9" fill="#2563eb" />
                            <text x="0" y="32" text-anchor="middle" font-size="10" font-weight="800" fill="#1d4ed8" font-family="'JetBrains Mono', monospace">LEFT (P5)</text>
                        </g>

                        <!-- Center Motor (Spine) -->
                        <g id="haptic-motor-center" transform="translate(140, 75)" style="display:none;">
                            <circle class="motor-ripple" r="26" fill="none" stroke="#3b82f6" stroke-width="2" opacity="0" />
                            <circle class="motor-base" r="18" fill="rgba(255, 255, 255, 0.9)" stroke="#94a3b8" stroke-width="2.5" />
                            <circle class="motor-core" r="9" fill="#cbd5e1" />
                        </g>

                        <!-- Right Motor (Pin 6) -->
                        <g id="haptic-motor-right" transform="translate(212, 115)">
                            <circle class="motor-ripple" r="26" fill="none" stroke="#3b82f6" stroke-width="2" opacity="0" />
                            <circle class="motor-base" r="18" fill="rgba(255, 255, 255, 0.9)" stroke="#94a3b8" stroke-width="2.5" />
                            <circle class="motor-core" r="9" fill="#cbd5e1" />
                            <text x="0" y="32" text-anchor="middle" font-size="10" font-weight="800" fill="#475569" font-family="'JetBrains Mono', monospace">RIGHT (P6)</text>
                        </g>
                    </svg>
                </div>
            `;

            this.motorElements = {
                LEFT: this.containerEl.querySelector("#haptic-motor-left"),
                CENTER: this.containerEl.querySelector("#haptic-motor-center"),
                RIGHT: this.containerEl.querySelector("#haptic-motor-right"),
            };
        }

        update(hapticCommand) {
            this.reset();
            if (!hapticCommand) return;

            const dir = (hapticCommand.direction || "STOP").toUpperCase();
            const pattern = hapticCommand.pattern_id || "ALL_CLEAR";
            const urgency = hapticCommand.urgency || 1;
            const isVibrating = pattern !== "ALL_CLEAR";

            if (!isVibrating) return;

            const activeMotors = [];
            if (dir === "LEFT") activeMotors.push(this.motorElements.LEFT);
            else if (dir === "CENTER") activeMotors.push(this.motorElements.CENTER);
            else if (dir === "RIGHT") activeMotors.push(this.motorElements.RIGHT);
            else if (dir === "STOP") {
                activeMotors.push(this.motorElements.LEFT, this.motorElements.CENTER, this.motorElements.RIGHT);
            }

            activeMotors.forEach(m => {
                if (!m) return;
                const base = m.querySelector(".motor-base");
                const core = m.querySelector(".motor-core");
                const ripple = m.querySelector(".motor-ripple");

                const activeColor = (urgency >= 4 || dir === "STOP") ? "#ef4444" : "#2563eb";

                if (base) {
                    base.setAttribute("stroke", activeColor);
                    base.setAttribute("fill", (urgency >= 4 || dir === "STOP") ? "#fef2f2" : "#eff6ff");
                }
                if (core) {
                    core.setAttribute("fill", activeColor);
                }
                if (ripple) {
                    ripple.setAttribute("stroke", activeColor);
                    ripple.style.animation = `motorRipple ${Math.max(0.3, 1.2 - urgency * 0.18)}s infinite cubic-bezier(0, 0.2, 0.8, 1)`;
                    ripple.setAttribute("opacity", "0.85");
                }
            });
        }

        reset() {
            Object.values(this.motorElements).forEach(m => {
                if (!m) return;
                const base = m.querySelector(".motor-base");
                const core = m.querySelector(".motor-core");
                const ripple = m.querySelector(".motor-ripple");

                if (base) {
                    base.setAttribute("stroke", "#94a3b8");
                    base.setAttribute("fill", "#ffffff");
                }
                if (core) {
                    core.setAttribute("fill", "#cbd5e1");
                }
                if (ripple) {
                    ripple.style.animation = "none";
                    ripple.setAttribute("opacity", "0");
                }
            });
        }
    }

    return {
        create: (el, opts) => new DiagramController(el, opts),
        getDescription: getPatternDescription,
    };
}));
