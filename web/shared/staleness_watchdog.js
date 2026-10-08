/**
 * SpatialVector-HMI — Shared Client-Side Staleness Watchdog.
 *
 * Enforces the M11 watchdog rule: flags pipeline stall if no telemetry packet
 * is received within threshold_ms (default 1500ms). Also handles the cold-start
 * "never connected" case.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.StalenessWatchdog = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    "use strict";

    class Watchdog {
        constructor(options = {}) {
            this.thresholdMs = options.thresholdMs || 1500;
            this.lastMessageTimestamp = 0;
            this.pageLoadTimestamp = Date.now();
            this.intervalId = null;

            this.onStale = options.onStale || null;
            this.onOffline = options.onOffline || null;
            this.onHealthy = options.onHealthy || null;
            this.onStaleChange = options.onStaleChange || null;
            this.statusBannerEl = options.statusBannerEl || null;
            this.statusPillEl = options.statusPillEl || null;
            this.statusTextEl = options.statusTextEl || null;
        }

        start(checkIntervalMs = 250) {
            if (this.intervalId) clearInterval(this.intervalId);
            this.intervalId = setInterval(() => this.check(), checkIntervalMs);
        }

        stop() {
            if (this.intervalId) {
                clearInterval(this.intervalId);
                this.intervalId = null;
            }
        }

        heartbeat() {
            this.lastMessageTimestamp = Date.now();
            if (this.onStaleChange) this.onStaleChange(false, 0);
        }

        onMessageReceived() {
            this.heartbeat();
        }

        check() {
            const now = Date.now();
            const hasNeverConnected = (this.lastMessageTimestamp === 0);
            const timeSinceLoad = now - this.pageLoadTimestamp;
            const timeSinceLastMsg = now - this.lastMessageTimestamp;

            if (hasNeverConnected) {
                if (timeSinceLoad > this.thresholdMs) {
                    this._triggerOffline(timeSinceLoad);
                }
                return;
            }

            if (timeSinceLastMsg > this.thresholdMs) {
                this._triggerStale(timeSinceLastMsg);
            } else {
                this._triggerHealthy();
            }
        }

        _triggerOffline(elapsedMs) {
            const sec = (elapsedMs / 1000).toFixed(1);
            if (this.statusBannerEl) {
                this.statusBannerEl.style.display = "block";
                this.statusBannerEl.textContent = `⚠️ PIPELINE OFFLINE — Waiting for telemetry server (${sec}s)`;
            }
            if (this.statusPillEl) this.statusPillEl.className = "conn-pill disconnected";
            if (this.statusTextEl) this.statusTextEl.textContent = "Pipeline Offline";
            if (this.onOffline) this.onOffline(elapsedMs);
            if (this.onStaleChange) this.onStaleChange(true, elapsedMs);
        }

        _triggerStale(elapsedMs) {
            const sec = (elapsedMs / 1000).toFixed(1);
            if (this.statusBannerEl) {
                this.statusBannerEl.style.display = "block";
                this.statusBannerEl.textContent = `⚠️ PIPELINE STALLED — No updates received for ${sec}s`;
            }
            if (this.statusPillEl) this.statusPillEl.className = "conn-pill disconnected";
            if (this.statusTextEl) this.statusTextEl.textContent = "Pipeline Stalled";
            if (this.onStale) this.onStale(elapsedMs);
            if (this.onStaleChange) this.onStaleChange(true, elapsedMs);
        }

        _triggerHealthy() {
            if (this.statusBannerEl) this.statusBannerEl.style.display = "none";
            if (this.statusPillEl) this.statusPillEl.className = "conn-pill connected";
            if (this.statusTextEl) this.statusTextEl.textContent = "Connected (15 FPS)";
            if (this.onHealthy) this.onHealthy();
            if (this.onStaleChange) this.onStaleChange(false, 0);
        }
    }

    return {
        create: (opts) => new Watchdog(opts),
        createWatchdog: (opts) => new Watchdog(opts),
    };
}));
