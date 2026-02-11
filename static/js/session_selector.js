/**
 * Session Selector Component
 *
 * Unified session selection modal for analysis pages.
 * Replaces individual file dropdowns with context-aware session management.
 */

class SessionSelector {
    constructor() {
        this.sessions = [];
        this.filteredSessions = [];
        this.currentTab = 'single';
        this.currentContext = null;
        this.onContextChange = null; // Callback when context changes
    }

    /**
     * Initialize the session selector
     */
    async init() {
        await this.loadSessions();
        await this.loadCurrentContext();
        this.renderScopeIndicator();
    }

    /**
     * Load all sessions from API
     */
    async loadSessions() {
        try {
            const response = await apiCall('/api/v2/sessions?limit=100');
            this.sessions = response.sessions || [];
            this.filteredSessions = [...this.sessions];
        } catch (error) {
            console.error('Failed to load sessions:', error);
            this.sessions = [];
            this.filteredSessions = [];
        }
    }

    /**
     * Load current analysis context
     */
    async loadCurrentContext() {
        try {
            const context = await apiCall('/api/context/current');
            this.currentContext = context;
        } catch (error) {
            this.currentContext = null;
        }
    }

    /**
     * Show the session selector modal
     */
    showModal() {
        const modal = document.getElementById('sessionSelectorModal');
        if (modal) {
            // Refresh data
            this.loadSessions().then(() => {
                this.renderSessionList();
            });

            // Show modal (Bootstrap 5)
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
        }
    }

    /**
     * Render the scope indicator (sticky header showing current selection)
     */
    renderScopeIndicator() {
        const container = document.getElementById('scopeIndicator');
        if (!container) return;

        if (!this.currentContext || !this.currentContext.scope) {
            container.innerHTML = `
                <div class="alert alert-warning mb-3">
                    <strong>No session selected</strong> -
                    <button class="btn btn-sm btn-primary ms-2" onclick="window._sessionSelector.showModal()">
                        Select Session
                    </button>
                </div>
            `;
            return;
        }

        const scope = this.currentContext.scope;
        const activeSessionId = this.currentContext.active_session_id;

        // Find the active session
        const activeSession = this.sessions.find(s => String(s.id) === String(activeSessionId));

        let html = '<div class="card mb-3 border-primary">';
        html += '<div class="card-body py-2">';
        html += '<div class="d-flex justify-content-between align-items-center">';
        html += '<div class="flex-grow-1">';
        html += '<strong class="text-primary">Analyzing:</strong> ';

        if (activeSession) {
            html += this.renderSessionCardMini(activeSession);
        } else {
            html += `<span class="text-muted">Session ${activeSessionId}</span>`;
        }

        // Show baseline if in comparison mode
        if (scope.mode === 'multi' && scope.baseline_session_id) {
            const baselineSession = this.sessions.find(s => String(s.id) === String(scope.baseline_session_id));
            html += ' <span class="text-muted">vs</span> ';
            if (baselineSession) {
                html += this.renderSessionCardMini(baselineSession, 'baseline');
            } else {
                html += `<span class="text-muted">Session ${scope.baseline_session_id}</span>`;
            }
        }

        html += '</div>';
        html += '<div>';
        html += '<button class="btn btn-sm btn-outline-primary" onclick="window._sessionSelector.showModal()">Change</button>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
        html += '</div>';

        container.innerHTML = html;
    }

    /**
     * Render a mini session card for the scope indicator
     */
    renderSessionCardMini(session, role = 'primary') {
        const trackBadge = session.track_name ? `<span class="badge bg-info me-1">${session.track_name}</span>` : '';
        const date = session.session_date ? new Date(session.session_date).toLocaleDateString() : 'Unknown date';
        const vehicle = session.vehicle_id ? session.vehicle_id.replace(/-/g, ' ') : '';
        const laps = session.total_laps || 0;

        return `
            <span class="session-mini ${role === 'baseline' ? 'text-muted' : ''}">
                ${trackBadge}
                <span>${date}</span>
                ${vehicle ? `<span class="text-muted ms-1">${vehicle}</span>` : ''}
                <span class="text-muted ms-1">(${laps} laps)</span>
            </span>
        `;
    }

    /**
     * Render the session list in the modal
     */
    renderSessionList() {
        const container = document.getElementById('sessionListContainer');
        if (!container) return;

        if (this.filteredSessions.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <p class="text-muted">No sessions found.</p>
                    <a href="/sessions/import" class="btn btn-primary">Import Session</a>
                </div>
            `;
            return;
        }

        let html = '';
        this.filteredSessions.forEach(session => {
            html += this.renderSessionCard(session);
        });

        container.innerHTML = html;
    }

    /**
     * Render a full session card
     */
    renderSessionCard(session) {
        const trackBadge = session.track_name ? `<span class="badge bg-info">${session.track_name}</span>` : '<span class="badge bg-secondary">Unknown Track</span>';
        const date = session.session_date ? new Date(session.session_date).toLocaleDateString() : 'Unknown date';
        const time = session.session_date ? new Date(session.session_date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '';
        const vehicle = session.vehicle_id || 'Unknown vehicle';
        const driver = session.driver_name ? `${session.driver_name}` : '';
        const runNumber = session.run_number ? ` - Run #${session.run_number}` : '';
        const laps = session.total_laps || 0;
        const bestLap = session.best_lap_time ? ` | Best: ${this.formatLapTime(session.best_lap_time)}` : '';
        const tags = session.tags || [];

        return `
            <div class="session-card card mb-2" data-session-id="${session.id}">
                <div class="card-body p-3">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="mb-1">
                                ${trackBadge}
                                <span class="ms-2 fw-bold">${date} ${time}</span>
                            </div>
                            <div class="text-muted small">
                                ${vehicle}${driver ? ` | ${driver}${runNumber}` : ''}
                            </div>
                            <div class="text-muted small">
                                ${laps} laps${bestLap}
                            </div>
                            ${tags.length > 0 ? `
                                <div class="mt-1">
                                    ${tags.map(tag => `<span class="badge bg-secondary me-1">${tag}</span>`).join('')}
                                </div>
                            ` : ''}
                        </div>
                        <div>
                            <button class="btn btn-sm btn-primary" onclick="window._sessionSelector.selectSession(${session.id})">
                                Select
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Select a session and set as analysis context
     */
    async selectSession(sessionId) {
        try {
            // Set as single session context
            await apiCall('/api/context/set', {
                method: 'POST',
                body: JSON.stringify({
                    mode: 'single',
                    session_ids: [String(sessionId)],
                })
            });

            // Reload context
            await this.loadCurrentContext();

            // Update UI
            this.renderScopeIndicator();

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('sessionSelectorModal'));
            if (modal) modal.hide();

            // Notify callback
            if (this.onContextChange) {
                this.onContextChange(this.currentContext);
            }

            showAlert('Session selected successfully!', 'success');
        } catch (error) {
            showAlert(`Failed to select session: ${error.message}`, 'danger');
        }
    }

    /**
     * Apply filters to session list
     */
    applyFilters() {
        const trackFilter = document.getElementById('filterTrack')?.value || '';
        const vehicleFilter = document.getElementById('filterVehicle')?.value || '';
        const driverFilter = document.getElementById('filterDriver')?.value || '';
        const searchQuery = document.getElementById('searchSessions')?.value.toLowerCase() || '';

        this.filteredSessions = this.sessions.filter(session => {
            if (trackFilter && session.track_id !== trackFilter) return false;
            if (vehicleFilter && session.vehicle_id !== vehicleFilter) return false;
            if (driverFilter && session.driver_name !== driverFilter) return false;

            if (searchQuery) {
                const searchText = `
                    ${session.track_name || ''}
                    ${session.driver_name || ''}
                    ${session.vehicle_id || ''}
                    ${session.tags ? session.tags.join(' ') : ''}
                `.toLowerCase();

                if (!searchText.includes(searchQuery)) return false;
            }

            return true;
        });

        this.renderSessionList();
    }

    /**
     * Format lap time (seconds to M:SS.mmm)
     */
    formatLapTime(seconds) {
        if (!seconds || seconds <= 0) return 'N/A';
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(3);
        return `${minutes}:${secs.padStart(6, '0')}`;
    }
}

// Global instance
window._sessionSelector = new SessionSelector();
