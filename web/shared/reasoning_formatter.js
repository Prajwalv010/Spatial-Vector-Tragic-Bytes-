/**
 * SpatialVector-HMI — Shared Reasoning Code Formatter.
 *
 * Translates M08/M11 raw machine reason codes into human-readable explanations.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.ReasoningFormatter = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    "use strict";

    const REASON_DICTIONARY = {
        ttc_low: "Time-to-collision below safe reaction threshold",
        intersection: "Predicted trajectory intersects future path",
        miss_dist: "Closest point of approach within danger corridor",
        degraded: "Sensor confidence degraded; running fallback mode",
        waiting_for_camera: "Waiting for camera connection",
        camera_disconnected: "Camera stream interrupted",
        center_corridor_blocked: "Forward navigation corridor is blocked",
        fallback_active: "Upstream sensor fallback mode active",
    };

    function formatCode(code) {
        if (!code || typeof code !== "string") return "";
        const parts = code.split(":");
        const type = parts[0];

        const base = REASON_DICTIONARY[type] || type.replace(/_/g, " ");
        const trackPart = parts.find(p => p.startsWith("track_"));
        const detailPart = parts.slice(1).find(p => !p.startsWith("track_"));

        let extra = "";
        if (detailPart) extra += ` [${detailPart}]`;
        if (trackPart) extra += ` (${trackPart.replace("_", " ")})`;

        return `${base}${extra}`;
    }

    function formatReasoningList(reasonCodes, selectedTrackId = null) {
        if (!reasonCodes || reasonCodes.length === 0) {
            return ["No active risk factors; clear corridor."];
        }

        let filtered = reasonCodes;
        if (selectedTrackId !== null && selectedTrackId !== undefined) {
            const matching = reasonCodes.filter(c => typeof c === "string" && c.includes(`track_${selectedTrackId}`));
            if (matching.length > 0) {
                filtered = matching;
            }
        }

        return filtered.map(c => formatCode(c));
    }

    return {
        format: formatCode,
        formatList: formatReasoningList,
    };
}));
