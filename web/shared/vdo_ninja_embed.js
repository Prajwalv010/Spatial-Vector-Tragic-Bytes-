/**
 * SpatialVector-HMI — Shared VDO.Ninja Camera Feed & Bounding Box Overlay Module.
 *
 * Reusable across both web/dashboard/ and web/app_simple/.
 * Provides:
 * 1. Safe VDO.Ninja URL formatting and iframe mounting with clean-output parameters.
 * 2. Automatic backend camera source synchronization (/api/source & /api/set-source).
 * 3. Aspect Ratio toggling (Fit vs Fill).
 * 4. Calibrated bounding box drawing with high-tech corner brackets and telemetry tag badges.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.VdoNinjaEmbed = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    "use strict";

    function formatVdoNinjaUrl(rawUrl) {
        let url = (rawUrl || "").trim();
        if (!url) return "https://vdo.ninja/?view=spatialvector_demo";

        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = `https://vdo.ninja/?view=${encodeURIComponent(url)}`;
        }

        const glue = url.includes("?") ? "&" : "?";
        const params = "cleanoutput&transparent=1&autoplay=1&autostart=1&noheader=1";
        if (!url.includes("cleanoutput")) {
            url += `${glue}${params}`;
        }
        return url;
    }

    class VdoNinjaManager {
        constructor(options = {}) {
            this.iframeEl = options.iframeEl || null;
            this.canvasEl = options.canvasEl || null;
            this.onSourceResolved = options.onSourceResolved || null;
            this.currentUrl = "https://vdo.ninja/?view=spatialvector_demo";
            this.isCover = true;
            this.hudEnabled = true;
        }

        init() {
            this.fetchBackendSource();
            window.addEventListener("resize", () => this.resizeCanvas());
            this.resizeCanvas();
        }

        async fetchBackendSource() {
            try {
                const res = await fetch("/api/source");
                if (res.ok) {
                    const data = await res.json();
                    if (data.source && data.source !== "0") {
                        this.setSource(data.source, false);
                    }
                }
            } catch (err) {
                console.warn("[VdoNinja] Failed to fetch /api/source:", err);
            }
        }

        async setSource(newSource, persist = true) {
            this.currentUrl = formatVdoNinjaUrl(newSource);
            if (this.iframeEl) {
                this.iframeEl.src = this.currentUrl;
            }
            if (persist) {
                try {
                    await fetch("/api/set-source", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ source: newSource })
                    });
                } catch (e) {
                    console.error("[VdoNinja] Failed to save source:", e);
                }
            }
            if (this.onSourceResolved) {
                this.onSourceResolved(this.currentUrl);
            }
        }

        resizeCanvas() {
            if (!this.canvasEl) return;
            const rect = this.canvasEl.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                this.canvasEl.width = Math.round(rect.width);
                this.canvasEl.height = Math.round(rect.height);
            }
        }

        toggleAspect() {
            this.isCover = !this.isCover;
            if (this.iframeEl) {
                this.iframeEl.style.objectFit = this.isCover ? "cover" : "contain";
            }
            return this.isCover;
        }

        toggleHud() {
            this.hudEnabled = !this.hudEnabled;
            if (this.canvasEl) {
                this.canvasEl.style.display = this.hudEnabled ? "block" : "none";
            }
            return this.hudEnabled;
        }

        drawBoundingBoxes(tracks, frameW = 640, frameH = 480, selectedTrackId = null, globalRisk = 0.0) {
            if (!this.canvasEl || !this.hudEnabled) return;

            // Sync canvas coordinate resolution with rendered CSS display dimensions
            const clientW = this.canvasEl.clientWidth || 400;
            const clientH = this.canvasEl.clientHeight || 224;
            if (this.canvasEl.width !== clientW || this.canvasEl.height !== clientH) {
                this.canvasEl.width = clientW;
                this.canvasEl.height = clientH;
            }

            const ctx = this.canvasEl.getContext("2d");
            const w = this.canvasEl.width;
            const h = this.canvasEl.height;

            ctx.clearRect(0, 0, w, h);

            // Ground Perspective Corridor Trapezoids (lower 44% of frame)
            const vanishY = h * 0.56;
            const vanishX = w * 0.50;

            // Center Corridor (Hazard corridor)
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(vanishX - 24, vanishY);
            ctx.lineTo(vanishX + 24, vanishY);
            ctx.lineTo(w * 0.68, h);
            ctx.lineTo(w * 0.32, h);
            ctx.closePath();

            if (globalRisk > 0.5) {
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

            if (!tracks || tracks.length === 0) return;

            const scaleX = w / (frameW || 640);
            const scaleY = h / (frameH || 480);

            tracks.forEach(track => {
                if (!track.bbox || !Array.isArray(track.bbox) || track.bbox.length !== 4) return;
                const bx = track.bbox[0] * scaleX;
                const by = track.bbox[1] * scaleY;
                const bw = (track.bbox[2] - track.bbox[0]) * scaleX;
                const bh = (track.bbox[3] - track.bbox[1]) * scaleY;

                if (bw < 4 || bh < 4) return;

                const isSelected = selectedTrackId === null || track.track_id === selectedTrackId;
                const isThreat = track.intersect || (globalRisk > 0.5 && isSelected);
                const strokeColor = isThreat ? "#ef4444" : "#10b981";

                ctx.save();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = isSelected ? 2.5 : 1.5;
                ctx.setLineDash([]);

                // Technical corner brackets
                const len = Math.min(12, Math.min(bw, bh) / 3);
                // Top-left
                ctx.beginPath(); ctx.moveTo(bx, by + len); ctx.lineTo(bx, by); ctx.lineTo(bx + len, by); ctx.stroke();
                // Top-right
                ctx.beginPath(); ctx.moveTo(bx + bw - len, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + len); ctx.stroke();
                // Bottom-left
                ctx.beginPath(); ctx.moveTo(bx, by + bh - len); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + len, by + bh); ctx.stroke();
                // Bottom-right
                ctx.beginPath(); ctx.moveTo(bx + bw - len, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - len); ctx.stroke();

                // Telemetry Badge Tag
                const ttcTxt = (track.ttc_s !== null && track.ttc_s !== undefined)
                    ? ` | TTC: ${Number(track.ttc_s).toFixed(1)}s`
                    : "";
                const tag = `${track.class_name || 'Obstacle'} #${track.track_id}${ttcTxt}`;
                ctx.font = "bold 10px 'JetBrains Mono', monospace";
                const tagW = ctx.measureText(tag).width;

                ctx.fillStyle = isThreat ? "rgba(220, 38, 38, 0.95)" : "rgba(16, 185, 129, 0.95)";
                ctx.beginPath();
                ctx.roundRect(bx - 2, Math.max(0, by - 20), tagW + 12, 18, 4);
                ctx.fill();

                ctx.fillStyle = "#ffffff";
                ctx.fillText(tag, bx + 4, Math.max(13, by - 7));
                ctx.restore();
            });
        }
    }

    return {
        formatUrl: formatVdoNinjaUrl,
        createManager: (opts) => new VdoNinjaManager(opts),
    };
}));
