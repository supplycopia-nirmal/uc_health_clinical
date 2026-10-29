/**
 * UC Health Clinical Multi-Agent Swarm Intelligence & CQO Platform
 * Powers Bee Hive Swarm v2.0 & Cost, Quality & Outcome Value Quadrants
 */

// Global Swarm & CQO State
window.activeBenchmark = 25;
window.activeChatPersona = "";
window.cachedCQOSummary = null;
window.cachedSwarmPersonas = null;

// Benchmark Percentile Switcher (10th, 25th Std, 50th)
function setBenchmarkPercentile(pct) {
    window.activeBenchmark = parseInt(pct) || 25;
    document.querySelectorAll('.btn-benchmark').forEach(btn => {
        if (parseInt(btn.getAttribute('data-pct')) === window.activeBenchmark) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    console.log(`[Swarm CQO] Benchmark target set to ${window.activeBenchmark}th percentile.`);
    
    // Refresh CQO summary and quadrant with updated benchmark
    fetchCQOSummary();
}

// Open Swarm Slide-out Drawer
function toggleSwarmDrawer() {
    const drawer = document.getElementById('swarm-drawer');
    if (!drawer) return;
    drawer.classList.toggle('open');
    const procDisplay = document.getElementById('drawer-proc-name');
    if (procDisplay) {
        procDisplay.innerText = window.currentProcedure || "All Procedures (Full Cohort)";
    }
}

// Programmatic Tab Opener
function openTabByName(tabName) {
    const tablinks = document.querySelectorAll('.tablinks');
    let targetBtn = null;
    tablinks.forEach(btn => {
        if (btn.getAttribute('onclick')?.includes(tabName)) {
            targetBtn = btn;
        }
    });
    if (targetBtn) {
        targetBtn.click();
    } else {
        const tabcontent = document.getElementsByClassName("tabcontent");
        for (let i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
        const el = document.getElementById(tabName);
        if (el) el.style.display = "block";
    }
}

// Select Persona in Chatbot Header
function selectChatPersona(btn, beeId) {
    document.querySelectorAll('.persona-chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    window.activeChatPersona = beeId;
    const input = document.getElementById('chat-input');
    if (input) {
        const beeLabel = btn.innerText.trim();
        input.placeholder = beeId ? `Ask ${beeLabel}...` : "Ask The Bee Swarm...";
        input.focus();
    }
}

// Fetch & Render Active Bee Personas
async function fetchSwarmPersonas() {
    const container = document.getElementById('bee-persona-cards-container');
    if (!container) return;
    
    try {
        const res = await fetch(`${API_BASE}/swarm/personas`);
        if (!res.ok) throw new Error("Failed to fetch personas");
        const data = await res.json();
        const personas = data.personas || data;
        window.cachedSwarmPersonas = personas;
        
        container.innerHTML = personas.map(p => `
            <div class="bee-card" id="persona-card-${p.id}">
                <div class="bee-avatar">${p.avatar}</div>
                <div class="bee-info">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <h4 class="bee-name">${p.name}</h4>
                        <span class="trace-badge trace-completed" style="font-size: 9px;">ACTIVE</span>
                    </div>
                    <div class="bee-role">${p.role}</div>
                    <div class="bee-desc">${p.description}</div>
                    <div style="margin-top: 8px;">
                        <span style="font-size: 10.5px; background: #fff7ed; color: #c2410c; padding: 2px 8px; border-radius: 4px; font-weight: 600; border: 1px solid #ffedd5;">${p.specialty}</span>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (err) {
        console.error("Error loading swarm personas:", err);
    }
}

// Fetch & Render CQO Summary KPIs
async function fetchCQOSummary() {
    const cohortLabel = document.getElementById('cqo-cohort-label');
    if (cohortLabel) {
        cohortLabel.innerText = window.currentProcedure ? `Procedure: ${window.currentProcedure}` : "All Procedures (68,478 Encounters)";
    }
    
    try {
        const pct = window.activeBenchmark || 25;
        const res = await fetch(`${API_BASE}/cqo/summary?percentile=${pct}`);
        if (!res.ok) throw new Error("Failed to fetch CQO summary");
        const data = await res.json();
        window.cachedCQOSummary = data;
        
        const m = data.system_metrics || data;
        
        // Populate KPI Cards
        const totalEnc = m.total_encounters || 68478;
        const totalSp = m.total_spend || 116544859;
        const avgCPC = m.cost_per_case_mean || (totalSp / (totalEnc || 1));
        const outScore = m.outcome_score !== undefined ? m.outcome_score : (m.outcome_score_mean || 7.20);
        
        const encEl = document.getElementById('cqo-kpi-encounters');
        if (encEl) encEl.innerText = totalEnc.toLocaleString();
        
        const spendEl = document.getElementById('cqo-kpi-spend');
        if (spendEl) spendEl.innerText = `Spend: $${(totalSp / 1e6).toFixed(2)}M`;
        
        const cpcEl = document.getElementById('cqo-kpi-cpc');
        if (cpcEl) cpcEl.innerText = `CPC: $${Math.round(avgCPC).toLocaleString()}`;
        
        const scoreEl = document.getElementById('cqo-kpi-outcome-score');
        if (scoreEl) scoreEl.innerText = `${outScore.toFixed(2)} / 10`;
        
        const ssiRate = m.ssi_rate !== undefined ? m.ssi_rate : 0.01;
        const ssiEl = document.getElementById('cqo-kpi-ssi');
        if (ssiEl) ssiEl.innerText = `${ssiRate < 1 ? (ssiRate * 100).toFixed(2) : ssiRate.toFixed(2)}%`;
        
        const ssiCasesEl = document.getElementById('cqo-kpi-ssi-cases');
        if (ssiCasesEl) ssiCasesEl.innerText = `${Math.round(totalEnc * (ssiRate < 1 ? ssiRate : ssiRate / 100)) || 7} Cases`;
        
        const readmRate = m.readmission_rate !== undefined ? m.readmission_rate : 2.49;
        const readmEl = document.getElementById('cqo-kpi-readmission');
        if (readmEl) readmEl.innerText = `${readmRate < 1 ? (readmRate * 100).toFixed(2) : readmRate.toFixed(2)}%`;
        
        const readmCasesEl = document.getElementById('cqo-kpi-readm-cases');
        if (readmCasesEl) readmCasesEl.innerText = `${(Math.round(totalEnc * (readmRate < 1 ? readmRate : readmRate / 100))).toLocaleString()} Cases`;
        
        const alosVal = m.alos_days !== undefined ? m.alos_days : (m.alos_mean || 2.09);
        const alosEl = document.getElementById('cqo-kpi-alos');
        if (alosEl) alosEl.innerText = `${alosVal.toFixed(2)} Days`;
        
        const mortRate = m.mortality_rate !== undefined ? m.mortality_rate : 0.32;
        const mortEl = document.getElementById('cqo-kpi-mortality');
        if (mortEl) mortEl.innerText = `Mortality: ${mortRate < 1 ? (mortRate * 100).toFixed(2) : mortRate.toFixed(2)}%`;
        
        const offContractSpend = m.off_contract_spend || 11395363;
        const contractSpend = m.contracted_spend || 105149495;
        const offPct = (offContractSpend / (totalSp || 1)) * 100;
        
        const leakEl = document.getElementById('cqo-kpi-leakage');
        if (leakEl) leakEl.innerText = `$${(offContractSpend / 1e6).toFixed(2)}M`;
        
        const leakPctEl = document.getElementById('cqo-kpi-leakage-pct');
        if (leakPctEl) leakPctEl.innerText = `${offPct.toFixed(1)}% Off-Contract`;
        
        const oppVal = m.potential_savings || m.opportunity_at_benchmark || (offContractSpend * 0.85);
        const oppEl = document.getElementById('cqo-kpi-opp-target');
        if (oppEl) oppEl.innerText = `Opp (${pct}%): $${(oppVal / 1e6).toFixed(2)}M`;
        
        // Contract Parity Sentinel Breakdown
        const contractDiv = document.getElementById('cqo-contract-summary');
        if (contractDiv) {
            contractDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">
                    <span style="color: var(--text-muted);">Contracted Spend Volume:</span>
                    <strong style="color: #047857;">$${(contractSpend / 1e6).toFixed(2)}M (${(100 - offPct).toFixed(1)}%)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">
                    <span style="color: var(--text-muted);">Off-Contract Leakage Volume:</span>
                    <strong style="color: #b91c1c;">$${(offContractSpend / 1e6).toFixed(2)}M (${offPct.toFixed(1)}%)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">
                    <span style="color: var(--text-muted);">10th Percentile Benchmark:</span>
                    <strong>$205.10 CPC ($24.80M Opp)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">
                    <span style="color: var(--text-muted);">25th Percentile Benchmark (Std):</span>
                    <strong style="color: var(--brand-red);">$382.40 CPC ($16.20M Opp)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                    <span style="color: var(--text-muted);">50th Percentile Median Benchmark:</span>
                    <strong>$821.50 CPC ($7.90M Opp)</strong>
                </div>
            `;
        }
        
        // Service Lines Summary
        const slDiv = document.getElementById('cqo-serviceline-summary');
        if (slDiv) {
            const lines = data.service_lines || [];
            slDiv.innerHTML = `
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="border-bottom: 2px solid var(--card-border); text-align: left; color: var(--text-muted);">
                            <th style="padding: 6px 4px;">Service Line</th>
                            <th style="padding: 6px 4px; text-align: right;">Cases</th>
                            <th style="padding: 6px 4px; text-align: right;">Total Spend</th>
                            <th style="padding: 6px 4px; text-align: right;">Outcome</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${lines.map(sl => `
                            <tr style="border-bottom: 1px solid #f1f5f9;">
                                <td style="padding: 6px 4px; font-weight: 600;">${sl.service_line}</td>
                                <td style="padding: 6px 4px; text-align: right;">${(sl.encounters || sl.total_encounters || 0).toLocaleString()}</td>
                                <td style="padding: 6px 4px; text-align: right;">$${(((sl.total_cost || sl.total_spend || 0)) / 1e6).toFixed(2)}M</td>
                                <td style="padding: 6px 4px; text-align: right; color: ${(sl.outcome_score || 7.0) >= 7.0 ? '#047857' : '#b45309'}; font-weight: 700;">${(sl.outcome_score || sl.outcome_score_mean || 7.2).toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
        
        // Render Quadrant Plot
        renderValueQuadrant(pct);
        
    } catch (err) {
        console.error("Error fetching CQO summary:", err);
    }
}

// Render Plotly Clinical Value Quadrant Chart
async function renderValueQuadrant(percentile) {
    const plotDiv = document.getElementById('cqo-quadrant-plot');
    if (!plotDiv) return;
    
    const pct = percentile || window.activeBenchmark || 25;
    
    try {
        const res = await fetch(`${API_BASE}/cqo/quadrant?percentile=${pct}`);
        if (!res.ok) throw new Error("Failed to fetch quadrant data");
        const result = await res.json();
        
        const surgeons = result.surgeons || [];
        const benchCost = result.benchmark_cpc || result.benchmark_cost || 267.72;
        const benchScore = result.target_outcome || result.benchmark_score || 7.0;
        
        if (surgeons.length === 0) {
            plotDiv.innerHTML = '<div style="text-align: center; padding: 100px; color: var(--text-muted);">No surgeon quadrant data available.</div>';
            return;
        }
        
        // Group surgeons by quadrant categories
        const traces = [];
        const quadGroups = [
            { name: 'Clinical Champions (Low Cost, High Quality)', color: '#10b981', filter: s => s.quadrant.includes('Champion') },
            { name: 'Preference Heavy (High Cost, High Quality)', color: '#3b82f6', filter: s => s.quadrant.includes('Preference') },
            { name: 'Under-Resourced (Low Cost, Lower Quality)', color: '#f59e0b', filter: s => s.quadrant.includes('Under-Resourced') },
            { name: 'Urgent Alignment (High Cost, Lower Quality)', color: '#ef4444', filter: s => s.quadrant.includes('Urgent') }
        ];
        
        quadGroups.forEach(grp => {
            const matched = surgeons.filter(grp.filter);
            if (matched.length === 0) return;
            
            traces.push({
                x: matched.map(s => s.avg_cpc !== undefined ? s.avg_cpc : s.avg_cost),
                y: matched.map(s => s.outcome_score),
                text: matched.map(s => s.surgeon_name),
                customdata: matched.map(s => [
                    s.cases,
                    s.service_line,
                    s.quadrant
                ]),
                mode: 'markers',
                name: grp.name,
                marker: {
                    size: matched.map(s => Math.min(26, Math.max(9, Math.sqrt(s.cases) * 3))),
                    color: grp.color,
                    opacity: 0.85,
                    line: { width: 1.5, color: '#ffffff' }
                },
                hovertemplate: 
                    '<b>Dr. %{text}</b><br>' +
                    'Service Line: %{customdata[1]}<br>' +
                    'Quadrant: <b>%{customdata[2]}</b><br>' +
                    'Cost Per Case: $%{x:,.2f}<br>' +
                    'Outcome Score: %{y:.2f} / 10<br>' +
                    'Total Cases: %{customdata[0]}<extra></extra>'
            });
        });
        
        const costs = surgeons.map(s => s.avg_cpc !== undefined ? s.avg_cpc : s.avg_cost);
        const maxCost = Math.max(...costs, benchCost * 1.5);
        
        const layout = {
            margin: { t: 30, r: 40, b: 60, l: 60 },
            hovermode: 'closest',
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#f8fafc',
            xaxis: {
                title: { text: `Cost Per Case ($) [Benchmark Line: $${Math.round(benchCost).toLocaleString()} @ ${pct}th %]`, font: { size: 12, color: '#475569', family: 'Open Sans' } },
                gridcolor: '#e2e8f0',
                zeroline: false,
                tickprefix: '$',
                tickformat: ',.0f'
            },
            yaxis: {
                title: { text: 'Clinical Outcome Composite Score (0 - 10)', font: { size: 12, color: '#475569', family: 'Open Sans' } },
                gridcolor: '#e2e8f0',
                zeroline: false,
                range: [4.8, 10.2]
            },
            shapes: [
                // Vertical Benchmark Line
                {
                    type: 'line',
                    x0: benchCost,
                    x1: benchCost,
                    y0: 4.8,
                    y1: 10.2,
                    line: { color: '#dc2626', width: 2, dash: 'dash' }
                },
                // Horizontal Target Outcome Line
                {
                    type: 'line',
                    x0: 0,
                    x1: maxCost * 1.05,
                    y0: benchScore,
                    y1: benchScore,
                    line: { color: '#059669', width: 2, dash: 'dot' }
                }
            ],
            annotations: [
                {
                    x: benchCost * 0.4,
                    y: 9.8,
                    text: '<b>🌟 CLINICAL CHAMPIONS</b><br><span style="font-size:10px;">Low Cost • High Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#047857' },
                    align: 'center'
                },
                {
                    x: benchCost * 2.2,
                    y: 9.8,
                    text: '<b>💎 PREFERENCE-HEAVY</b><br><span style="font-size:10px;">High Cost • High Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#1d4ed8' },
                    align: 'center'
                },
                {
                    x: benchCost * 0.4,
                    y: 5.3,
                    text: '<b>⚠️ UNDER-RESOURCED</b><br><span style="font-size:10px;">Low Cost • Lower Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#b45309' },
                    align: 'center'
                },
                {
                    x: benchCost * 2.2,
                    y: 5.3,
                    text: '<b>🚨 URGENT ALIGNMENT</b><br><span style="font-size:10px;">High Cost • Lower Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#b91c1c' },
                    align: 'center'
                }
            ],
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.12,
                font: { size: 10.5, color: '#334155' }
            }
        };
        
        Plotly.newPlot('cqo-quadrant-plot', traces, layout, { responsive: true, displayModeBar: true });
        
        // On Click: clicking a surgeon prompts Ask The Bee
        plotDiv.on('plotly_click', function(data) {
            if (data.points && data.points.length > 0) {
                const surgeonName = data.points[0].text;
                console.log(`[CQO Quadrant] Surgeon selected: ${surgeonName}`);
                const chatContainer = document.getElementById('chatbot-container');
                if (chatContainer.style.display !== 'flex') {
                    toggleChat();
                }
                const chatInput = document.getElementById('chat-input');
                if (chatInput) {
                    chatInput.value = `Analyze Dr. ${surgeonName}'s clinical outcomes, cost per case, and preference card variance.`;
                    sendChatMessage();
                }
            }
        });
        
    } catch (err) {
        console.error("Error rendering value quadrant:", err);
    }
}

// Fetch Autonomous Swarm Audit
async function fetchSwarmAudit() {
    const btn = document.getElementById('btn-run-swarm-audit');
    const traceStatus = document.getElementById('swarm-trace-status');
    const traceBox = document.getElementById('swarm-trace-container');
    const drawerTrace = document.getElementById('drawer-trace-list');
    const synthesisBox = document.getElementById('swarm-synthesis-content');
    const consensusBadge = document.getElementById('swarm-consensus-badge');
    const debateSection = document.getElementById('swarm-debate-section');
    const debateList = document.getElementById('swarm-debate-list');
    const actionsSection = document.getElementById('swarm-actions-section');
    const actionsButtons = document.getElementById('swarm-actions-buttons');
    const drawerSynthesis = document.getElementById('drawer-synthesis-text');
    
    if (btn) {
        btn.innerText = "⏳ Swarm Mobilizing (6 Bees)...";
        btn.disabled = true;
    }
    if (traceStatus) traceStatus.innerText = "Swarm Deliberating in Parallel...";
    
    const initialTrace = `
        <div class="trace-item"><span class="trace-badge trace-running">Mobilizing</span><span>👑 Queen Bee dispatched Scout Bee, Clinical Bee, Contract Bee, Worker Bee, and Surgeon Bee...</span></div>
        <div class="trace-item"><span class="trace-badge trace-running">Auditing</span><span>Scout Bee parsing cases for rogue card variance & Isolation Forest outliers...</span></div>
        <div class="trace-item"><span class="trace-badge trace-running">Validating</span><span>Clinical Bee auditing safety metrics (SSI, Readmissions, Mortality, ALOS)...</span></div>
    `;
    if (traceBox) traceBox.innerHTML = initialTrace;
    if (drawerTrace) drawerTrace.innerHTML = initialTrace;
    
    try {
        const res = await fetch(`${API_BASE}/swarm/audit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                procedure_name: window.currentProcedure || "COLONOSCOPY WITH OR WITHOUT BIOPSY",
                benchmark_percentile: window.activeBenchmark || 25
            })
        });
        
        if (!res.ok) throw new Error("Swarm audit request failed");
        const audit = await res.json();
        
        if (traceStatus) traceStatus.innerText = "Audit Completed";
        if (consensusBadge) consensusBadge.style.display = "inline-block";
        
        // Render Telemetry Trace
        const traces = audit.agent_traces || audit.execution_trace || [];
        if (traces.length > 0) {
            const traceHTML = traces.map(t => {
                const badgeClass = t.status === 'completed' ? 'trace-completed' : (t.status === 'warning' ? 'trace-warning' : 'trace-running');
                return `
                    <div class="trace-item">
                        <span class="trace-badge ${badgeClass}">${t.status}</span>
                        <strong style="color: #fdba74;">[${t.agent_avatar || '🐝'} ${t.agent_name}]</strong>
                        <span><b>${t.action}</b>: ${t.thought || t.findings || ''}</span>
                    </div>
                `;
            }).join('');
            if (traceBox) traceBox.innerHTML = traceHTML;
            if (drawerTrace) drawerTrace.innerHTML = traceHTML;
        }
        
        // Render Queen Bee Executive Summary
        if (synthesisBox && audit.executive_summary) {
            synthesisBox.innerHTML = marked.parse(audit.executive_summary);
        }
        if (drawerSynthesis && audit.executive_summary) {
            drawerSynthesis.innerHTML = marked.parse(audit.executive_summary);
        }
        
        // Render Inter-Agent Debate
        const debate = audit.debate_transcript || (audit.agent_findings && (audit.agent_findings.debate_transcript || audit.agent_findings.inter_agent_debate)) || [];
        if (debate.length > 0) {
            if (debateSection) debateSection.style.display = "block";
            if (debateList) {
                debateList.innerHTML = debate.map(d => {
                    const sender = d.sender || d.agent_name || "Agent";
                    const content = d.content || d.thought || "";
                    let icon = '🐝';
                    if (sender.includes('Clinical')) icon = '🩺';
                    else if (sender.includes('Contract')) icon = '📜';
                    else if (sender.includes('Worker')) icon = '📦';
                    else if (sender.includes('Scout')) icon = '🔍';
                    else if (sender.includes('Surgeon')) icon = '👨‍⚕️';
                    else if (sender.includes('Queen')) icon = '👑';
                    return `
                        <div class="debate-item">
                            <span style="font-size: 20px;">${icon}</span>
                            <div class="debate-bubble">
                                <strong style="color: #0f172a;">${sender}</strong>: ${content}
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }
        
        // Render Recommended Actions
        const actions = audit.suggested_actions || audit.recommended_actions || [];
        if (actions.length > 0) {
            if (actionsSection) actionsSection.style.display = "block";
            if (actionsButtons) {
                actionsButtons.innerHTML = actions.map(act => {
                    const label = typeof act === 'object' ? (act.label || act.action) : act;
                    return `
                        <button class="btn-chat-action" style="padding: 6px 14px; font-size: 12px;" onclick="executeSwarmAction('${label}')">
                            ⚡ ${label}
                        </button>
                    `;
                }).join('');
            }
        }
        
    } catch (err) {
        console.error("Swarm audit failed:", err);
        if (traceBox) {
            traceBox.innerHTML += `<div class="trace-item"><span class="trace-badge trace-warning">Error</span><span>Swarm audit error: ${err.message}</span></div>`;
        }
    } finally {
        if (btn) {
            btn.innerText = "🚀 Launch Autonomous Hive Audit";
            btn.disabled = false;
        }
    }
}

// Execute Swarm Recommended Action
function executeSwarmAction(actionText) {
    console.log(`[Swarm Action] Executing: ${actionText}`);
    const lower = actionText.toLowerCase();
    
    if (lower.includes('quadrant') || lower.includes('cqo') || lower.includes('cost vs')) {
        openTabByName('CQOTab');
    } else if (lower.includes('leakage') || lower.includes('off-contract')) {
        openTabByName('LeakageTab');
    } else if (lower.includes('parity') || lower.includes('arbitrage') || lower.includes('contract')) {
        openTabByName('ParityTab');
        if (typeof fetchPriceParity === 'function') fetchPriceParity();
    } else if (lower.includes('pack') || lower.includes('core') || lower.includes('hold list')) {
        openTabByName('ProcCardTab');
        if (typeof fetchProcedureCard === 'function') fetchProcedureCard();
    } else if (lower.includes('preference card') || lower.includes('surgeon')) {
        openTabByName('PrefCardTab');
    } else {
        // Query Ask The Bee
        const chatContainer = document.getElementById('chatbot-container');
        if (chatContainer.style.display !== 'flex') {
            toggleChat();
        }
        const input = document.getElementById('chat-input');
        if (input) {
            input.value = `Execute strategic action: ${actionText}`;
            sendChatMessage();
        }
    }
}

// Initialize Swarm and CQO on page ready
$(document).ready(function() {
    setTimeout(() => {
        fetchSwarmPersonas();
        fetchCQOSummary();
    }, 200);
});
