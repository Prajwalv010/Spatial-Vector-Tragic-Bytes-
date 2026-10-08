/**
 * SpatialVector-HMI — Subpage: Social Assist (Redesigned)
 * Improved UI: avatar initials, live recognition demo, animated scan, contact cards.
 * Also includes proximity buzzer controls (test + sensitivity).
 */

window.SocialAssistPage = (function () {
    "use strict";

    let containerEl = null;
    let socialEnabled = true;
    let activeTab = "contacts"; // "contacts" | "live"
    let recognitionState = "known"; // "known" | "unknown" | "possible"
    let scanAnimFrame = null;

    let contacts = [
        { id: "c1", name: "Rahul",  relationship: "Friend",    photoCount: 3, confidence: 91, status: "Ready",   initials: "RA", color: "#6366f1" },
        { id: "c2", name: "Ananya", relationship: "Family",    photoCount: 4, confidence: 87, status: "Ready",   initials: "AN", color: "#ec4899" },
        { id: "c3", name: "Arjun",  relationship: "Classmate", photoCount: 2, confidence: 94, status: "Ready",   initials: "AR", color: "#10b981" },
    ];

    let addFormState = { name: "", relationship: "Friend", photoDataUrls: [] };

    // ── colour helpers ────────────────────────────────────────────────────────
    const PALETTE = ["#6366f1","#ec4899","#10b981","#f59e0b","#3b82f6","#8b5cf6","#14b8a6","#f97316"];
    function colorFor(name) {
        let h = 0;
        for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffffffff;
        return PALETTE[Math.abs(h) % PALETTE.length];
    }
    function initials(name) {
        const parts = name.trim().split(/\s+/);
        return parts.length >= 2
            ? (parts[0][0] + parts[1][0]).toUpperCase()
            : name.substring(0, 2).toUpperCase();
    }

    function avatarCircle(name, color, size = 44, fontSize = 16) {
        return `<div style="width:${size}px;height:${size}px;border-radius:50%;background:${color};
                    display:flex;align-items:center;justify-content:center;
                    font-size:${fontSize}px;font-weight:800;color:#fff;flex-shrink:0;
                    box-shadow:0 2px 8px ${color}55;">
                    ${initials(name)}
                </div>`;
    }

    // ── relationship badge color ──────────────────────────────────────────────
    function relBadge(rel) {
        const map = {
            Friend:    { bg: "#eff6ff", col: "#1d4ed8" },
            Family:    { bg: "#fdf4ff", col: "#7e22ce" },
            Classmate: { bg: "#ecfdf5", col: "#065f46" },
            Colleague: { bg: "#fff7ed", col: "#c2410c" },
            Other:     { bg: "#f8fafc", col: "#475569" },
        };
        const t = map[rel] || map.Other;
        return `<span style="background:${t.bg};color:${t.col};font-size:10px;font-weight:700;
                    padding:2px 8px;border-radius:99px;">${rel}</span>`;
    }

    // ─────────────────────────────────────────────────────────────────────────
    function init(container) {
        containerEl = container;
        render();
    }

    /* ================================================================
       MAIN SOCIAL ASSIST SCREEN
       ================================================================ */
    function render() {
        // cancel any ongoing scan animation
        if (scanAnimFrame) { cancelAnimationFrame(scanAnimFrame); scanAnimFrame = null; }

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="App.closeSubpage()">‹ Back</button>
                <span class="subpage-title">Social Assist</span>
            </div>

            <!-- Privacy + Toggle row -->
            <div style="display:flex;align-items:center;justify-content:space-between;
                        background:var(--bg-card);border:1px solid var(--border-subtle);
                        border-radius:14px;padding:12px 14px;margin-bottom:14px;
                        box-shadow:var(--shadow-card);">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;
                                background:${socialEnabled ? '#eff6ff' : '#f1f5f9'};
                                display:flex;align-items:center;justify-content:center;font-size:18px;">
                        ${socialEnabled ? '👥' : '🔕'}
                    </div>
                    <div>
                        <div style="font-size:13px;font-weight:700;">Social Assist</div>
                        <div style="font-size:10px;color:var(--text-muted);">On-device · No data leaves your device</div>
                    </div>
                </div>
                <button id="social-toggle-btn"
                    class="toggle-btn ${socialEnabled ? 'on' : 'off'}"
                    onclick="SocialAssistPage.toggleEnabled()">
                    ${socialEnabled ? "ON" : "OFF"}
                </button>
            </div>

            <!-- Stat strip -->
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
                ${statChip('👤', contacts.length, 'Contacts')}
                ${statChip('✓', contacts.filter(c=>c.status==="Ready").length, 'Ready')}
                ${statChip('📷', socialEnabled ? 'Live' : 'Off', 'Recognition')}
            </div>

            <!-- Tab Switcher -->
            <div class="segmented-control" style="margin-bottom:14px;">
                <button id="tab-contacts-btn"
                    class="seg-btn ${activeTab === 'contacts' ? 'active' : ''}"
                    onclick="SocialAssistPage.switchTab('contacts')">
                    Known Contacts
                </button>
                <button id="tab-live-btn"
                    class="seg-btn ${activeTab === 'live' ? 'active' : ''}"
                    onclick="SocialAssistPage.switchTab('live')">
                    Live Recognition
                </button>
            </div>

            <!-- Proximity Buzzer Panel -->
            <div id="buzzer-panel"
                 style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                        border:1px solid #334155;border-radius:14px;
                        padding:14px;margin-bottom:14px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:34px;height:34px;border-radius:10px;
                                    background:rgba(239,68,68,.15);
                                    display:flex;align-items:center;justify-content:center;font-size:18px;">
                            🔔
                        </div>
                        <div>
                            <div style="font-size:13px;font-weight:700;color:#f1f5f9;">Proximity Alert</div>
                            <div style="font-size:10px;color:#64748b;">Audio buzzer for obstacles</div>
                        </div>
                    </div>
                    <button id="buzzer-toggle-btn"
                        class="toggle-btn ${window.ProximityBuzzer && window.ProximityBuzzer.isEnabled() ? 'on' : 'off'}"
                        onclick="SocialAssistPage.toggleBuzzer()"
                        style="flex-shrink:0;">
                        ${window.ProximityBuzzer && window.ProximityBuzzer.isEnabled() ? 'ON' : 'OFF'}
                    </button>
                </div>
                <!-- Threshold indicators -->
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;">
                    <div style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
                                border-radius:8px;padding:7px 10px;text-align:center;">
                        <div style="font-size:18px;font-weight:800;color:#f59e0b;">4s</div>
                        <div style="font-size:9px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Caution Beep</div>
                    </div>
                    <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
                                border-radius:8px;padding:7px 10px;text-align:center;">
                        <div style="font-size:18px;font-weight:800;color:#ef4444;">2s</div>
                        <div style="font-size:9px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Rapid Alert</div>
                    </div>
                </div>
                <button class="buzzer-btn" onclick="SocialAssistPage.testBuzzer()">
                    🔈 Test Alert Sound
                </button>
            </div>

            <!-- Tab Content -->
            <div id="social-tab-content"></div>
        `;

        renderTabContent();
    }


    function statChip(icon, value, label) {
        return `<div style="background:var(--bg-card);border:1px solid var(--border-subtle);
                    border-radius:12px;padding:10px 8px;text-align:center;box-shadow:var(--shadow-sm);">
                    <div style="font-size:16px;">${icon}</div>
                    <div style="font-size:15px;font-weight:800;color:var(--text-primary);margin-top:2px;">${value}</div>
                    <div style="font-size:9px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">${label}</div>
                </div>`;
    }

    function toggleEnabled() {
        socialEnabled = !socialEnabled;
        render();
    }

    function switchTab(tab) {
        activeTab = tab;
        if (scanAnimFrame) { cancelAnimationFrame(scanAnimFrame); scanAnimFrame = null; }
        const tabContacts = document.getElementById("tab-contacts-btn");
        const tabLive    = document.getElementById("tab-live-btn");
        if (tabContacts) tabContacts.className = `seg-btn ${tab === 'contacts' ? 'active' : ''}`;
        if (tabLive)     tabLive.className     = `seg-btn ${tab === 'live'     ? 'active' : ''}`;
        renderTabContent();
    }

    function renderTabContent() {
        const content = document.getElementById("social-tab-content");
        if (!content) return;
        if (activeTab === "contacts") renderContactsTab(content);
        else                          renderLiveTab(content);
    }

    /* ================================================================
       KNOWN CONTACTS TAB
       ================================================================ */
    function renderContactsTab(el) {
        if (contacts.length === 0) {
            el.innerHTML = `
                <div style="text-align:center;padding:40px 0;">
                    <div style="width:72px;height:72px;border-radius:50%;background:#f1f5f9;
                                margin:0 auto 14px;display:flex;align-items:center;
                                justify-content:center;font-size:32px;">👥</div>
                    <div style="font-size:15px;font-weight:700;margin-bottom:6px;">No contacts added yet</div>
                    <div style="font-size:12px;color:var(--text-muted);margin-bottom:20px;">
                        Add people you know for optional recognition.
                    </div>
                    <button class="action-btn" style="max-width:200px;margin:0 auto;"
                        onclick="SocialAssistPage.openAddFriend()">+ Add Contact</button>
                </div>`;
            return;
        }

        el.innerHTML = `
            <div style="font-size:11px;color:var(--text-muted);font-weight:600;
                        text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">
                ${contacts.length} saved contact${contacts.length !== 1 ? 's' : ''}
            </div>
            <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px;">
                ${contacts.map(c => `
                    <div onclick="SocialAssistPage.openContactProfile('${c.id}')"
                         style="background:var(--bg-card);border:1px solid var(--border-subtle);
                                border-radius:14px;padding:12px 14px;
                                display:flex;align-items:center;gap:12px;
                                cursor:pointer;transition:box-shadow .15s;box-shadow:var(--shadow-sm);"
                         onmouseover="this.style.boxShadow='0 4px 16px rgba(37,99,235,.12)'"
                         onmouseout="this.style.boxShadow='var(--shadow-sm)'">
                        ${avatarCircle(c.name, c.color || colorFor(c.name), 44, 16)}
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:14px;font-weight:700;margin-bottom:3px;">${c.name}</div>
                            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                                ${relBadge(c.relationship)}
                                <span style="font-size:10px;color:var(--text-muted);">
                                    ${c.photoCount} photo${c.photoCount !== 1 ? 's' : ''} · ${c.confidence}%
                                </span>
                            </div>
                        </div>
                        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
                            <span class="pill-badge green" style="font-size:9px;">${c.status}</span>
                            <span style="color:var(--text-muted);font-size:18px;line-height:1;">›</span>
                        </div>
                    </div>
                `).join('')}
            </div>
            <button class="action-btn" onclick="SocialAssistPage.openAddFriend()">+ Add Contact</button>
        `;
    }

    /* ================================================================
       CONTACT PROFILE VIEW
       ================================================================ */
    function openContactProfile(contactId) {
        const c = contacts.find(x => x.id === contactId);
        if (!c) return;
        const col = c.color || colorFor(c.name);

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="SocialAssistPage.render()">‹ Contacts</button>
                <span class="subpage-title">${c.name}</span>
            </div>

            <!-- Hero -->
            <div style="text-align:center;padding:20px 0 16px;">
                <div style="width:80px;height:80px;border-radius:50%;background:${col};
                            margin:0 auto 12px;display:flex;align-items:center;
                            justify-content:center;font-size:28px;font-weight:900;color:#fff;
                            box-shadow:0 4px 20px ${col}66;">
                    ${initials(c.name)}
                </div>
                <div style="font-size:20px;font-weight:800;">${c.name}</div>
                <div style="display:flex;justify-content:center;gap:8px;margin-top:6px;flex-wrap:wrap;">
                    ${relBadge(c.relationship)}
                    <span class="pill-badge green" style="font-size:10px;">✓ ${c.confidence}% confidence</span>
                </div>
            </div>

            <!-- Stats row -->
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
                ${statChip('📸', c.photoCount, 'Photos')}
                ${statChip('🎯', c.confidence+'%', 'Match')}
                ${statChip('✅', c.status, 'Status')}
            </div>

            <!-- Photo Grid -->
            <div class="expand-panel open" style="margin-bottom:12px;">
                <div class="expand-header" style="cursor:default;"><span>Photos</span></div>
                <div class="expand-body" style="display:block;">
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
                        ${Array.from({length: c.photoCount}).map((_, i) => `
                            <div style="background:linear-gradient(135deg,${col}22,${col}11);
                                        border-radius:10px;height:70px;
                                        display:flex;align-items:center;justify-content:center;
                                        font-size:22px;font-weight:700;color:${col};
                                        border:1px solid ${col}33;">
                                ${initials(c.name)}
                            </div>
                        `).join('')}
                        <div onclick="SocialAssistPage.openAddFriend('${c.id}')"
                             style="background:#f8fafc;border-radius:10px;height:70px;
                                    display:flex;flex-direction:column;align-items:center;
                                    justify-content:center;font-size:20px;
                                    border:2px dashed #cbd5e1;cursor:pointer;color:#94a3b8;gap:2px;">
                            <span>+</span>
                            <span style="font-size:9px;font-weight:600;">Photo</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Details -->
            <div class="expand-panel open" style="margin-bottom:14px;">
                <div class="expand-header" style="cursor:default;"><span>Contact Details</span></div>
                <div class="expand-body" style="display:block;">
                    <div class="kv-row"><span class="kv-key">Name</span><span class="kv-val">${c.name}</span></div>
                    <div class="kv-row"><span class="kv-key">Relationship</span><span class="kv-val">${relBadge(c.relationship)}</span></div>
                    <div class="kv-row"><span class="kv-key">Photos</span><span class="kv-val">${c.photoCount}</span></div>
                    <div class="kv-row"><span class="kv-key">Confidence</span><span class="kv-val">${c.confidence}%</span></div>
                    <div class="kv-row"><span class="kv-key">Status</span><span class="pill-badge green">${c.status}</span></div>
                </div>
            </div>

            <!-- Actions -->
            <div style="display:flex;flex-direction:column;gap:8px;">
                <button class="action-btn secondary"
                    onclick="SocialAssistPage.openEditContact('${c.id}')">✏️ Edit Contact</button>
                <button class="action-btn"
                    style="background:#fef2f2;color:#991b1b;border:1px solid #fecaca;"
                    onclick="SocialAssistPage.deleteContact('${c.id}')">🗑 Delete Contact</button>
            </div>
        `;
    }

    function deleteContact(contactId) {
        contacts = contacts.filter(c => c.id !== contactId);
        render();
    }

    function openEditContact(contactId) {
        const c = contacts.find(x => x.id === contactId);
        if (!c) return;

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="SocialAssistPage.openContactProfile('${c.id}')">‹ ${c.name}</button>
                <span class="subpage-title">Edit Contact</span>
            </div>

            <div class="expand-panel open" style="margin-bottom:14px;">
                <div class="expand-header" style="cursor:default;"><span>Edit Details</span></div>
                <div class="expand-body" style="display:block;">
                    <div style="margin-bottom:12px;">
                        <label style="font-size:12px;font-weight:700;display:block;margin-bottom:6px;">Name</label>
                        <input id="edit-name-input" type="text" value="${c.name}"
                            style="width:100%;padding:10px 12px;border-radius:10px;
                                   border:1px solid var(--border-subtle);font-size:14px;font-family:inherit;">
                    </div>
                    <div>
                        <label style="font-size:12px;font-weight:700;display:block;margin-bottom:6px;">Relationship</label>
                        <select id="edit-rel-input"
                            style="width:100%;padding:10px 12px;border-radius:10px;
                                   border:1px solid var(--border-subtle);font-size:14px;
                                   font-family:inherit;background:#fff;">
                            ${["Friend","Family","Classmate","Colleague","Other"].map(r =>
                                `<option ${r === c.relationship ? 'selected' : ''}>${r}</option>`).join('')}
                        </select>
                    </div>
                </div>
            </div>
            <button class="action-btn" onclick="SocialAssistPage.saveEditContact('${c.id}')">Save Changes</button>
        `;
    }

    function saveEditContact(contactId) {
        const nameInput = document.getElementById("edit-name-input");
        const relInput  = document.getElementById("edit-rel-input");
        const c = contacts.find(x => x.id === contactId);
        if (c && nameInput && relInput) {
            c.name         = nameInput.value.trim() || c.name;
            c.relationship = relInput.value;
            c.color        = colorFor(c.name);
        }
        openContactProfile(contactId);
    }

    /* ================================================================
       ADD FRIEND FLOW
       ================================================================ */
    function openAddFriend(existingId) {
        addFormState = { name: "", relationship: "Friend", photoDataUrls: [], existingId: existingId || null };
        renderAddFriendStep1();
    }

    function renderAddFriendStep1() {
        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="SocialAssistPage.render()">‹ Contacts</button>
                <span class="subpage-title">Add Contact</span>
            </div>

            <p style="font-size:12px;color:var(--text-muted);margin-bottom:14px;">
                Add a name and one or more photos of this person.
            </p>

            <!-- Photo Grid -->
            <div class="expand-panel open" style="margin-bottom:14px;">
                <div class="expand-header" style="cursor:default;"><span>Photos</span></div>
                <div class="expand-body" style="display:block;">
                    <div id="add-photo-grid"
                         style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;">
                        <div class="add-photo-slot" onclick="SocialAssistPage.triggerFilePicker()">
                            <span style="font-size:24px;">+</span>
                            <span style="font-size:10px;color:var(--text-muted);">Add Photo</span>
                        </div>
                    </div>
                    <input type="file" id="photo-file-input" accept="image/*" multiple
                           style="display:none;" onchange="SocialAssistPage.onPhotosSelected(event)">
                    <p style="font-size:10px;color:var(--text-muted);">
                        Tap + to select photos from your device. Photos are not uploaded anywhere.
                    </p>
                </div>
            </div>

            <!-- Name & Relationship -->
            <div class="expand-panel open" style="margin-bottom:14px;">
                <div class="expand-header" style="cursor:default;"><span>Contact Info</span></div>
                <div class="expand-body" style="display:block;">
                    <div style="margin-bottom:12px;">
                        <label style="font-size:12px;font-weight:700;display:block;margin-bottom:6px;">Name</label>
                        <input id="add-name-input" type="text" placeholder="Enter name..."
                               value="${addFormState.name}"
                               oninput="SocialAssistPage.onNameInput(this.value)"
                               style="width:100%;padding:10px 12px;border-radius:10px;
                                      border:1px solid var(--border-subtle);
                                      font-size:14px;font-family:inherit;">
                    </div>
                    <div>
                        <label style="font-size:12px;font-weight:700;display:block;margin-bottom:6px;">
                            Relationship <span style="font-weight:400;color:var(--text-muted);">(optional)</span>
                        </label>
                        <select id="add-rel-input"
                                onchange="SocialAssistPage.onRelInput(this.value)"
                                style="width:100%;padding:10px 12px;border-radius:10px;
                                       border:1px solid var(--border-subtle);font-size:14px;
                                       font-family:inherit;background:#fff;">
                            ${["Friend","Family","Classmate","Colleague","Other"].map(r =>
                                `<option ${r === addFormState.relationship ? 'selected' : ''}>${r}</option>`
                            ).join('')}
                        </select>
                    </div>
                </div>
            </div>

            <!-- Continue -->
            <button class="action-btn" onclick="SocialAssistPage.goToReview()">Continue to Review →</button>
        `;
        updateAddPhotoGrid();
    }

    function triggerFilePicker() {
        const input = document.getElementById("photo-file-input");
        if (input) input.click();
    }

    function onPhotosSelected(event) {
        const files = Array.from(event.target.files);
        let loaded = 0;
        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = (e) => {
                addFormState.photoDataUrls.push(e.target.result);
                loaded++;
                if (loaded === files.length) updateAddPhotoGrid();
            };
            reader.readAsDataURL(file);
        });
    }

    function updateAddPhotoGrid() {
        const grid = document.getElementById("add-photo-grid");
        if (!grid) return;
        const photos = addFormState.photoDataUrls;
        const photoSlots = photos.map((url, i) => `
            <div style="background:#f1f5f9;border-radius:10px;height:70px;overflow:hidden;
                        border:1px solid var(--border-subtle);position:relative;">
                <img src="${url}" style="width:100%;height:100%;object-fit:cover;">
                <button onclick="SocialAssistPage.removePhoto(${i})"
                        style="position:absolute;top:2px;right:2px;background:rgba(0,0,0,.55);
                               border:none;color:#fff;border-radius:50%;width:18px;height:18px;
                               font-size:10px;cursor:pointer;display:flex;align-items:center;
                               justify-content:center;">×</button>
            </div>
        `).join('');
        grid.innerHTML = photoSlots + `
            <div class="add-photo-slot" onclick="SocialAssistPage.triggerFilePicker()">
                <span style="font-size:24px;">+</span>
                <span style="font-size:10px;color:var(--text-muted);">Add</span>
            </div>
        `;
    }

    function removePhoto(idx) {
        addFormState.photoDataUrls.splice(idx, 1);
        updateAddPhotoGrid();
    }

    function onNameInput(val) { addFormState.name = val; }
    function onRelInput(val)  { addFormState.relationship = val; }

    function goToReview() {
        const nameEl = document.getElementById("add-name-input");
        const relEl  = document.getElementById("add-rel-input");
        if (nameEl) addFormState.name = nameEl.value.trim();
        if (relEl)  addFormState.relationship = relEl.value;
        renderReview();
    }

    function renderReview() {
        const { name, relationship, photoDataUrls } = addFormState;
        const photoCount = photoDataUrls.length || 0;
        const col = colorFor(name || "?");

        containerEl.innerHTML = `
            <div class="subpage-header">
                <button class="back-btn" onclick="SocialAssistPage.renderAddFriendStep1()">‹ Edit</button>
                <span class="subpage-title">Review Contact</span>
            </div>

            <div style="text-align:center;padding:16px 0 10px;">
                <div style="width:72px;height:72px;border-radius:50%;background:${col};
                            margin:0 auto 12px;display:flex;align-items:center;
                            justify-content:center;font-size:24px;font-weight:900;color:#fff;
                            box-shadow:0 4px 20px ${col}66;">
                    ${name ? initials(name) : '?'}
                </div>
                <div style="font-size:17px;font-weight:800;">${name || "(No name entered)"}</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
                    ${relationship} · ${photoCount} photo${photoCount !== 1 ? 's' : ''}
                </div>
            </div>

            ${photoCount > 0 ? `
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
                    ${photoDataUrls.slice(0,6).map(url => `
                        <div style="background:#f1f5f9;border-radius:10px;height:70px;
                                    overflow:hidden;border:1px solid var(--border-subtle);">
                            <img src="${url}" style="width:100%;height:100%;object-fit:cover;">
                        </div>
                    `).join('')}
                </div>
            ` : `
                <div style="text-align:center;padding:14px;background:#fffbeb;
                            border-radius:12px;border:1px solid #fde68a;margin-bottom:14px;">
                    <span style="font-size:12px;color:#92400e;">⚠ No photos added. Adding photos helps with recognition.</span>
                </div>
            `}

            <div class="expand-panel open" style="margin-bottom:14px;">
                <div class="expand-header" style="cursor:default;"><span>Summary</span></div>
                <div class="expand-body" style="display:block;">
                    <div class="kv-row"><span class="kv-key">Name</span><span class="kv-val">${name || "—"}</span></div>
                    <div class="kv-row"><span class="kv-key">Relationship</span><span class="kv-val">${relationship}</span></div>
                    <div class="kv-row"><span class="kv-key">Photos</span><span class="kv-val">${photoCount}</span></div>
                    <div class="kv-row"><span class="kv-key">Status</span><span class="pill-badge green">Ready</span></div>
                </div>
            </div>

            <button class="action-btn" style="margin-bottom:8px;" onclick="SocialAssistPage.saveFriend()">
                ✓ Save Contact
            </button>
            <button class="action-btn secondary" onclick="SocialAssistPage.renderAddFriendStep1()">← Go Back</button>
        `;
    }

    function saveFriend() {
        const { name, relationship, photoDataUrls } = addFormState;
        if (!name) { alert("Please enter a name for this contact."); return; }
        contacts.push({
            id:           `c${Date.now()}`,
            name,
            relationship,
            photos:       photoDataUrls.length > 0 ? photoDataUrls : ["👤"],
            photoCount:   photoDataUrls.length || 1,
            confidence:   Math.floor(85 + Math.random() * 12),
            status:       "Ready",
            color:        colorFor(name),
        });
        addFormState = { name: "", relationship: "Friend", photoDataUrls: [] };
        render();
    }

    /* ================================================================
       LIVE RECOGNITION TAB  (improved)
       ================================================================ */
    function renderLiveTab(el) {
        const states = {
            known: {
                label: "Known Contact", person: contacts[0] ? contacts[0].name : "Rahul",
                confidence: contacts[0] ? contacts[0].confidence : 91,
                color: "#10b981", badge: "green", icon: "✓",
                col: contacts[0] ? (contacts[0].color || colorFor(contacts[0].name || "Rahul")) : "#10b981",
            },
            unknown: {
                label: "Unknown Person", person: "Unknown", confidence: 0,
                color: "#ef4444", badge: "red", icon: "?", col: "#ef4444",
            },
            possible: {
                label: "Possible Match", person: contacts[0] ? contacts[0].name : "Rahul",
                confidence: 62, color: "#f59e0b", badge: "yellow", icon: "~",
                col: "#f59e0b",
            },
        };
        const s = states[recognitionState];

        el.innerHTML = `
            <!-- Demo banner -->
            <div class="guidance-prompt-card" style="margin-bottom:12px;">
                <span class="guidance-icon">🎭</span>
                <span class="guidance-text" style="font-size:11px;">
                    <strong>Demo Mode:</strong> Use the chips below to simulate recognition states.
                </span>
            </div>

            <!-- Camera viewport -->
            <div id="social-cam-box"
                 style="background:#0f172a;border-radius:16px;height:200px;margin-bottom:12px;
                        position:relative;overflow:hidden;border:1px solid #334155;">

                <!-- Subtle grid overlay -->
                <svg width="100%" height="100%" style="position:absolute;top:0;left:0;opacity:.06;">
                    <defs>
                        <pattern id="sg" width="24" height="24" patternUnits="userSpaceOnUse">
                            <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#94a3b8" stroke-width=".5"/>
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#sg)"/>
                </svg>

                <!-- Corner brackets -->
                ${['top:8px;left:8px;border-top:2px solid #475569;border-left:2px solid #475569',
                   'top:8px;right:8px;border-top:2px solid #475569;border-right:2px solid #475569',
                   'bottom:8px;left:8px;border-bottom:2px solid #475569;border-left:2px solid #475569',
                   'bottom:8px;right:8px;border-bottom:2px solid #475569;border-right:2px solid #475569'].map(s =>
                    `<div style="position:absolute;${s};width:12px;height:12px;border-radius:2px;"></div>`
                ).join('')}

                <!-- Scanning line -->
                <div id="social-scan-line"
                     style="position:absolute;left:0;right:0;height:2px;
                            background:linear-gradient(90deg,transparent,${s.color},transparent);
                            top:0;transition:top .05s;opacity:.7;pointer-events:none;"></div>

                <!-- Detection box -->
                <div id="social-det-box"
                     style="position:absolute;border:2px solid ${s.color};border-radius:8px;
                            width:80px;height:100px;top:40px;left:50%;transform:translateX(-50%);
                            box-shadow:0 0 12px ${s.color}66;transition:border-color .3s;">
                    <!-- Avatar initials inside box -->
                    <div style="position:absolute;top:0;left:0;right:0;bottom:0;
                                display:flex;align-items:center;justify-content:center;">
                        <div style="width:44px;height:44px;border-radius:50%;
                                    background:${recognitionState === 'unknown' ? '#ef444433' : s.col + '33'};
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:14px;font-weight:800;color:${s.color};">
                            ${recognitionState === 'unknown' ? '?' : initials(s.person)}
                        </div>
                    </div>
                    <!-- Name tag -->
                    <div style="position:absolute;bottom:-22px;left:50%;transform:translateX(-50%);
                                background:${s.color};color:#fff;font-size:9px;font-weight:700;
                                padding:2px 8px;border-radius:4px;white-space:nowrap;">
                        ${s.icon} ${s.person}
                    </div>
                </div>

                <!-- Track ID badge -->
                <div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,.6);
                            color:#94a3b8;font-size:9px;font-weight:700;padding:3px 7px;
                            border-radius:6px;">TRK #4</div>

                <!-- Live indicator -->
                <div style="position:absolute;top:10px;right:10px;display:flex;align-items:center;
                            gap:5px;background:rgba(0,0,0,.6);padding:3px 8px;border-radius:6px;">
                    <div style="width:6px;height:6px;border-radius:50%;background:#ef4444;
                                animation:pulse-dot 1s ease-in-out infinite;"></div>
                    <span style="font-size:9px;color:#e2e8f0;font-weight:700;">DEMO</span>
                </div>
            </div>

            <!-- Recognition Result Card -->
            <div class="expand-panel open" style="margin-bottom:12px;">
                <div class="expand-header" style="cursor:default;">
                    <span>Identified</span>
                    <span class="pill-badge ${s.badge}">${s.icon} ${s.label}</span>
                </div>
                <div class="expand-body" style="display:block;">
                    <div style="display:flex;align-items:center;gap:14px;padding:10px 0;">
                        ${avatarCircle(s.person, s.col, 52, 18)}
                        <div>
                            <div style="font-size:20px;font-weight:800;color:${s.color};">${s.person}</div>
                            <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">
                                ${recognitionState !== 'unknown'
                                    ? `${s.label} · ${s.confidence}% confidence`
                                    : 'No matching contact found'}
                            </div>
                            ${recognitionState === 'known' && contacts[0] ? `
                                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">
                                    ${contacts[0].relationship} · Track #4
                                </div>` : ''}
                        </div>
                    </div>
                    ${recognitionState !== 'unknown' ? `
                        <div style="background:#f1f5f9;border-radius:8px;overflow:hidden;height:6px;margin-top:4px;">
                            <div style="height:100%;width:${s.confidence}%;background:${s.color};
                                        border-radius:8px;transition:width .5s;"></div>
                        </div>
                        <div style="font-size:9px;color:var(--text-muted);margin-top:3px;">
                            Confidence: ${s.confidence}%
                        </div>
                    ` : ''}
                </div>
            </div>

            <!-- Demo State Selector -->
            <div class="expand-panel open" style="margin-bottom:12px;">
                <div class="expand-header" style="cursor:default;"><span>Simulate Recognition</span></div>
                <div class="expand-body" style="display:block;">
                    <p style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">
                        Tap a state to preview the recognition UI:
                    </p>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;">
                        <button class="quick-test-chip ${recognitionState === 'known'   ? 'active' : ''}"
                            onclick="SocialAssistPage.setRecognitionState('known')">✓ Known</button>
                        <button class="quick-test-chip ${recognitionState === 'unknown' ? 'active' : ''}"
                            onclick="SocialAssistPage.setRecognitionState('unknown')">? Unknown</button>
                        <button class="quick-test-chip ${recognitionState === 'possible'? 'active' : ''}"
                            onclick="SocialAssistPage.setRecognitionState('possible')">~ Possible</button>
                    </div>
                </div>
            </div>
        `;

        // start scan-line animation
        startScanLine(s.color);
    }

    function startScanLine(color) {
        const box = document.getElementById("social-cam-box");
        const line = document.getElementById("social-scan-line");
        if (!box || !line) return;
        let y = 0;
        const h = box.offsetHeight || 200;
        let dir = 1;
        function step() {
            y += dir * 1.2;
            if (y >= h) { y = h; dir = -1; }
            if (y <= 0) { y = 0;  dir =  1; }
            line.style.top = y + "px";
            line.style.background = `linear-gradient(90deg,transparent,${color},transparent)`;
            scanAnimFrame = requestAnimationFrame(step);
        }
        step();
    }

    function setRecognitionState(state) {
        recognitionState = state;
        if (scanAnimFrame) { cancelAnimationFrame(scanAnimFrame); scanAnimFrame = null; }
        renderTabContent();
    }

    // ── Buzzer helpers ──────────────────────────────────────────────────────
    function toggleBuzzer() {
        if (!window.ProximityBuzzer) return;
        const next = !ProximityBuzzer.isEnabled();
        ProximityBuzzer.setEnabled(next);
        const btn = document.getElementById("buzzer-toggle-btn");
        if (btn) {
            btn.textContent = next ? "ON" : "OFF";
            btn.className   = `toggle-btn ${next ? 'on' : 'off'}`;
        }
    }

    function testBuzzer() {
        if (window.ProximityBuzzer) ProximityBuzzer.testBeep();
    }

    // ── Public API ──────────────────────────────────────────────────────────
    return {
        init,
        render,
        toggle: () => {},
        toggleEnabled,
        toggleBuzzer,
        testBuzzer,
        switchTab,
        openContactProfile,
        deleteContact,
        openEditContact,
        saveEditContact,
        openAddFriend,
        renderAddFriendStep1,
        triggerFilePicker,
        onPhotosSelected,
        removePhoto,
        onNameInput,
        onRelInput,
        goToReview,
        renderReview,
        saveFriend,
        setRecognitionState,
    };
})();
