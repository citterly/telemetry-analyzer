/**
 * Audit Mode Manager for Telemetry Analyzer
 *
 * Provides UI controls and rendering for the safeguard system's
 * calculation trace and sanity check display.
 */

class AuditManager {
    constructor() {
        this.storageKey = 'telemetry_audit_enabled';
    }

    /**
     * Get current audit mode state from localStorage
     */
    get enabled() {
        return localStorage.getItem(this.storageKey) === 'true';
    }

    /**
     * Set audit mode state and persist to localStorage
     */
    set enabled(value) {
        localStorage.setItem(this.storageKey, value ? 'true' : 'false');
    }

    /**
     * Generate trace query parameter if audit is enabled
     * @param {string} prefix - URL prefix ('?' or '&')
     * @returns {string} - Empty string or '&trace=true'
     */
    traceParam(prefix = '&') {
        return this.enabled ? `${prefix}trace=true` : '';
    }

    /**
     * Render toggle button HTML
     * @returns {string} - Bootstrap switch HTML
     */
    renderToggleButton() {
        const checked = this.enabled ? 'checked' : '';
        return `
            <div class="form-check form-switch audit-toggle">
                <input class="form-check-input" type="checkbox" id="auditModeToggle" ${checked}
                       onchange="window._auditManager.onToggleChange(this.checked)">
                <label class="form-check-label" for="auditModeToggle">
                    Audit Mode
                </label>
            </div>
        `;
    }

    /**
     * Handle toggle change event
     */
    onToggleChange(checked) {
        this.enabled = checked;
        // Call page-specific callback if defined
        if (typeof window.onAuditToggle === 'function') {
            window.onAuditToggle(checked);
        }
    }

    /**
     * Determine status color from sanity checks
     * @param {Array} checks - Array of sanity check objects
     * @returns {string} - 'green', 'yellow', or 'red'
     */
    getStatus(checks) {
        if (!checks || checks.length === 0) return 'green';

        const hasFail = checks.some(c => c.status === 'fail');
        const hasWarn = checks.some(c => c.status === 'warn');

        if (hasFail) return 'red';
        if (hasWarn) return 'yellow';
        return 'green';
    }

    /**
     * Render status dot HTML
     * @param {string} status - 'green', 'yellow', or 'red'
     * @returns {string} - Colored dot span
     */
    renderDot(status) {
        return `<span class="audit-dot audit-dot-${status}"></span>`;
    }

    /**
     * Render section indicator (dot + summary count)
     * @param {object} trace - CalculationTrace object
     * @returns {string} - HTML with dot and summary
     */
    renderSectionIndicator(trace) {
        if (!trace || !trace.sanity_checks) return '';

        const status = this.getStatus(trace.sanity_checks);
        const passCount = trace.sanity_checks.filter(c => c.status === 'pass').length;
        const total = trace.sanity_checks.length;

        return `
            ${this.renderDot(status)}
            <span class="audit-summary">${passCount}/${total} checks passed</span>
        `;
    }

    /**
     * Render full audit panel HTML
     * @param {object} trace - CalculationTrace object
     * @param {string} title - Panel title
     * @returns {string} - Complete audit panel HTML
     */
    renderPanel(trace, title = "Calculation Audit") {
        if (!trace) return '';

        const status = this.getStatus(trace.sanity_checks);
        const statusClass = `audit-panel-${status}`;

        return `
            <div class="audit-panel ${statusClass}">
                <div class="audit-panel-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <span class="audit-panel-title">
                        ${this.renderDot(status)}
                        ${title}
                    </span>
                    <span class="audit-panel-arrow">▼</span>
                </div>
                <div class="audit-panel-body">
                    ${this.renderInputsSection(trace.inputs)}
                    ${this.renderConfigSection(trace.config)}
                    ${this.renderIntermediatesSection(trace.intermediates)}
                    ${this.renderChecksSection(trace.sanity_checks)}
                    ${this.renderWarningsSection(trace.warnings)}
                </div>
            </div>
        `;
    }

    /**
     * Render inputs section
     * @private
     */
    renderInputsSection(inputs) {
        if (!inputs || Object.keys(inputs).length === 0) return '';

        const rows = Object.entries(inputs).map(([key, value]) => `
            <tr>
                <td class="audit-key">${key}</td>
                <td class="audit-value">${this.formatValue(value)}</td>
            </tr>
        `).join('');

        return `
            <div class="audit-section">
                <div class="audit-section-title">INPUTS</div>
                <table class="audit-table">
                    ${rows}
                </table>
            </div>
        `;
    }

    /**
     * Render config section
     * @private
     */
    renderConfigSection(config) {
        if (!config || Object.keys(config).length === 0) return '';

        const rows = Object.entries(config).map(([key, value]) => `
            <tr>
                <td class="audit-key">${key}</td>
                <td class="audit-value">${this.formatValue(value)}</td>
            </tr>
        `).join('');

        return `
            <div class="audit-section">
                <div class="audit-section-title">CONFIG</div>
                <table class="audit-table">
                    ${rows}
                </table>
            </div>
        `;
    }

    /**
     * Render intermediates section
     * @private
     */
    renderIntermediatesSection(intermediates) {
        if (!intermediates || Object.keys(intermediates).length === 0) return '';

        const rows = Object.entries(intermediates).map(([key, value]) => `
            <tr>
                <td class="audit-key">${key}</td>
                <td class="audit-value">${this.formatValue(value)}</td>
            </tr>
        `).join('');

        return `
            <div class="audit-section">
                <div class="audit-section-title">INTERMEDIATES</div>
                <table class="audit-table">
                    ${rows}
                </table>
            </div>
        `;
    }

    /**
     * Render sanity checks section
     * @private
     */
    renderChecksSection(checks) {
        if (!checks || checks.length === 0) return '';

        const rows = checks.map(check => {
            const dotHtml = this.renderDot(check.status === 'pass' ? 'green' : check.status === 'warn' ? 'yellow' : 'red');
            return `
                <div class="audit-check-row audit-check-${check.status}">
                    <div class="audit-check-header">
                        ${dotHtml}
                        <span class="audit-check-name">${check.name}</span>
                        <span class="audit-check-status audit-check-status-${check.status}">${check.status.toUpperCase()}</span>
                    </div>
                    <div class="audit-check-message">${check.message}</div>
                    ${check.impact ? `<div class="audit-check-impact"><strong>Impact:</strong> ${check.impact}</div>` : ''}
                    ${check.expected || check.actual ? `
                        <div class="audit-check-values">
                            ${check.expected ? `<span>Expected: ${this.formatValue(check.expected)}</span>` : ''}
                            ${check.actual ? `<span>Actual: ${this.formatValue(check.actual)}</span>` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        return `
            <div class="audit-section">
                <div class="audit-section-title">SANITY CHECKS</div>
                ${rows}
            </div>
        `;
    }

    /**
     * Render warnings section
     * @private
     */
    renderWarningsSection(warnings) {
        if (!warnings || warnings.length === 0) return '';

        const items = warnings.map(w => `<li>${w}</li>`).join('');

        return `
            <div class="audit-section">
                <div class="audit-section-title">WARNINGS</div>
                <ul class="audit-warnings-list">
                    ${items}
                </ul>
            </div>
        `;
    }

    /**
     * Format value for display
     * @private
     */
    formatValue(value) {
        if (value === null || value === undefined) return 'null';
        if (typeof value === 'number') {
            return value.toFixed(value % 1 === 0 ? 0 : 2);
        }
        if (typeof value === 'object') {
            return JSON.stringify(value);
        }
        return String(value);
    }
}

// Create global instance
window._auditManager = new AuditManager();
