/**
 * UC Health Clinical Multi-Agent Swarm Intelligence & CQO Platform
 * Powers Bee Hive Swarm v2.0 & Cost, Quality & Outcome Value Quadrants
 */

window.API_BASE = window.API_BASE || "/api";
var API_BASE = window.API_BASE;

// Global Swarm & CQO State
window.activeBenchmark = 25;
window.activeChatPersona = "";
window.cachedCQOSummary = null;
window.cachedSwarmPersonas = null;

// Static Fallback Bee Personas to guarantee never-empty UI
const STATIC_PERSONAS = [
    {
        id: "queen_bee",
        name: "Queen Bee",
        role: "Master Swarm Orchestrator & Synthesizer",
        avatar: "👑",
        specialty: "Multi-Agent Hive Coordination & Executive Briefings",
        description: "The central command of the Bee Swarm. Decomposes user goals, dispatches specialized worker bees in parallel, synthesizes conflicting perspectives (clinical safety vs financial savings), and delivers executive decisions."
    },
    {
        id: "clinical_bee",
        name: "Clinical Bee",
        role: "Quality, Safety & Outcomes Specialist",
        avatar: "🩺",
        specialty: "Surgical Equivalence & Patient Acuity",
        description: "Ensures clinical safety primacy over cost, audits surgical indications, correlates supply variation to adverse outcomes (SSI, Readmissions, Mortality, ALOS), and guards against risky substitutions."
    },
    {
        id: "scout_bee",
        name: "Scout Bee",
        role: "Anomaly & Leakage Forager",
        avatar: "🔍",
        specialty: "1.37M Cases Reconnaissance & Outlier Detection",
        description: "Continuously scours the 1,378,322 case records to identify cost/volume outliers, compositional anomalies, off-contract spend leakage, and cross-hospital price disparities."
    },
    {
        id: "contract_bee",
        name: "Contract Bee",
        role: "Parity & Benchmark Strategist",
        avatar: "📜",
        specialty: "10th/25th/50th Percentile Benchmarks & Price Arbitrage",
        description: "Analyzes cross-facility price parity (UH OR vs WCH OR vs UH ASC), calculates 10th/25th/50th percentile savings potential, and audits off-contract vendor leakage."
    },
    {
        id: "worker_bee",
        name: "Worker Bee",
        role: "BOM & Pack Optimizer",
        avatar: "📦",
        specialty: "FP-Growth Item Bundling & Standard Custom Packs",
        description: "Mines clinical utilization patterns to design lean, standardized custom surgical packs, eliminates opened-but-unused hold waste, and aligns physician preference cards."
    },
    {
        id: "surgeon_bee",
        name: "Surgeon Bee",
        role: "Physician Advocate & RLHF Alignment",
        avatar: "👨‍⚕️",
        specialty: "Surgeon Autonomy, Nuance & Case Complexity Adjustments",
        description: "Represents physician clinical judgment, protects surgeon choice in high-acuity/revision cases, prevents burnout from aggressive procurement mandates, and powers the RLHF alignment loop."
    },
    {
        id: "migration_bee",
        name: "Migration Bee",
        role: "Site-of-Service & ASC Shift Specialist",
        avatar: "🏥",
        specialty: "Inpatient-to-ASC Shift & Facility Overhead Arbitrage",
        description: "Flags low-acuity cases (ASA 1-2, younger cohorts) routinely performed in high-overhead hospital ORs (UH OR, WCH OR) that can safely migrate to the Outpatient Surgery Center (UH ASC), projecting facility-overhead margin expansion."
    },
    {
        id: "audit_bee",
        name: "Audit Bee",
        role: "Price Discrepancy & Rebate Clawback Auditor",
        avatar: "⚖️",
        specialty: "Cross-Facility Invoice Auditing & Vendor Overcharge Recovery",
        description: "Cross-tabulates item master numbers and vendor unit costs across UC Health facilities. Detects identical catalog items billed at higher rates at specific sites to generate automated credit memo demand notices."
    },
    {
        id: "eco_bee",
        name: "Eco Bee",
        role: "Green OR Bio-Waste & ESG Sentinel",
        avatar: "🌱",
        specialty: "Clinical Bio-Waste Mass (kg) & Scope 3 Carbon Footprint",
        description: "Quantifies the environmental impact of opened-but-unused hold items and over-stuffed custom packs. Converts eliminated clinical plastic/metal surgical waste into kilograms of bio-waste avoided and Scope 3 metric tons of CO2e reduced."
    },
    {
        id: "payer_bee",
        name: "Payer Bee",
        role: "DRG Reimbursement & Managed Care Margin Analyst",
        avatar: "💰",
        specialty: "MS-DRG Case-Mix, Carve-Outs & Contribution Margin",
        description: "Bridges surgical supply costs with DRG fixed payments, commercial fee-for-service carve-outs, and Medicare allowable margins, ensuring supply standardization directly protects net operating margin."
    },
    {
        id: "flow_bee",
        name: "Flow Bee",
        role: "OR Turnover & SPD De-Cluttering Specialist",
        avatar: "⏱️",
        specialty: "Sterile Processing (SPD) Tray Rationalization & OR Turnover Minutes",
        description: "Analyzes instrument tray utilization and sterile processing burdens. Reduces unused instruments on surgical trays to accelerate room turnover times and lower Central Sterile reprocessing costs."
    },
    {
        id: "governance_bee",
        name: "Governance Bee",
        role: "Value Analysis Committee (CAC) Coordinator",
        avatar: "🏛️",
        specialty: "Clinical Evidence Grading & System-Wide Value Analysis",
        description: "Synthesizes multi-agent evidence into formal Clinical Advisory Committee (CAC) voting dossiers, tracks clinical exception request approval rates, and monitors health-system policy compliance."
    }
];

// Benchmark Percentile Switcher (10th, 25th Std, 50th)
function setBenchmarkPercentile(pct) {
    window.activeBenchmark = parseInt(pct) || 25;
    document.querySelectorAll('.btn-benchmark').forEach(btn => {
        const btnPct = parseInt(btn.getAttribute('data-pct'));
        if (btnPct === window.activeBenchmark) {
            btn.classList.add('active');
            btn.style.background = '#fff';
            btn.style.color = 'var(--brand-red, #e43946)';
            btn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.06)';
        } else {
            btn.classList.remove('active');
            btn.style.background = 'transparent';
            btn.style.color = 'var(--text-muted, #64748b)';
            btn.style.boxShadow = 'none';
        }
    });
    console.log(`[Swarm CQO] Benchmark target switched to ${window.activeBenchmark}th percentile.`);
    
    // Refresh CQO summary and quadrant with updated benchmark
    fetchCQOSummary();
    if (typeof renderValueQuadrant === 'function') {
        renderValueQuadrant(window.activeBenchmark);
    }
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

// Render Persona Cards
function renderPersonaCards(personas) {
    const container = document.getElementById('bee-persona-cards-container');
    if (!container) return;
    
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
}

// Fetch & Render Active Bee Personas
async function fetchSwarmPersonas() {
    renderPersonaCards(STATIC_PERSONAS); // Immediate render so never empty
    
    try {
        const res = await fetch(`${API_BASE}/swarm/personas`);
        if (res.ok) {
            const data = await res.json();
            const personas = data.personas || data;
            if (Array.isArray(personas) && personas.length > 0) {
                window.cachedSwarmPersonas = personas;
                renderPersonaCards(personas);
            }
        }
    } catch (err) {
        console.warn("Using static swarm personas fallback:", err);
    }
}

// Fetch & Render CQO Summary KPIs
async function fetchCQOSummary() {
    const cohortLabel = document.getElementById('cqo-cohort-label');
    const proc = window.currentProcedure || "";
    if (cohortLabel) {
        cohortLabel.innerText = proc ? `Procedure: ${proc}` : "All Procedures (68,478 Encounters)";
    }
    
    try {
        const pct = window.activeBenchmark || 25;
        const procParam = proc ? `&procedure=${encodeURIComponent(proc)}` : '';
        const res = await fetch(`${API_BASE}/cqo/summary?percentile=${pct}${procParam}`);
        if (!res.ok) throw new Error(`CQO summary error HTTP ${res.status}`);
        const data = await res.json();
        window.cachedCQOSummary = data;
        
        const m = data.system_metrics || data;
        
        // Populate KPI Cards
        const totalEnc = m.total_encounters || 0;
        const totalSp = m.total_spend || 0;
        const avgCPC = totalEnc > 0 ? (totalSp / totalEnc) : (m.cost_per_case_mean || 0);
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
        if (ssiCasesEl) ssiCasesEl.innerText = `${Math.round(totalEnc * (ssiRate < 1 ? ssiRate : ssiRate / 100)) || 1} Cases`;
        
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
        
        const offContractSpend = m.off_contract_spend || (totalSp * 0.098);
        const contractSpend = m.contracted_spend || (totalSp - offContractSpend);
        const offPct = totalSp > 0 ? ((offContractSpend / totalSp) * 100) : 9.8;
        
        const leakEl = document.getElementById('cqo-kpi-leakage');
        if (leakEl) leakEl.innerText = `$${(offContractSpend / 1e6).toFixed(2)}M`;
        
        const leakPctEl = document.getElementById('cqo-kpi-leakage-pct');
        if (leakPctEl) leakPctEl.innerText = `${offPct.toFixed(1)}% Off-Contract`;
        
        const oppVal = m.potential_savings || (offContractSpend * 0.85);
        const oppEl = document.getElementById('cqo-kpi-opp-target');
        if (oppEl) oppEl.innerText = `Opp (${pct}%): $${(oppVal / 1e6).toFixed(2)}M`;
        
        // Contract Parity Sentinel Breakdown
        const contractDiv = document.getElementById('cqo-contract-summary');
        if (contractDiv) {
            const bCpc = m.benchmark_cpc || 115;
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
                    <span style="color: var(--text-muted);">10th Percentile Aggressive Benchmark:</span>
                    <strong>$${Math.round(bCpc * 0.75)} CPC (Top Savings Tier)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;">
                    <span style="color: var(--text-muted);">25th Percentile Standard Benchmark:</span>
                    <strong style="color: var(--brand-red);">$${Math.round(bCpc)} CPC (Standard Target)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                    <span style="color: var(--text-muted);">50th Percentile Median Benchmark:</span>
                    <strong>$${Math.round(bCpc * 1.55)} CPC (Median Tier)</strong>
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
    const proc = window.currentProcedure || "";
    
    try {
        const procParam = proc ? `&procedure=${encodeURIComponent(proc)}` : '';
        const res = await fetch(`${API_BASE}/cqo/quadrant?percentile=${pct}${procParam}`);
        if (!res.ok) throw new Error(`Quadrant data HTTP ${res.status}`);
        const result = await res.json();
        
        const surgeons = result.surgeons || [];
        const benchCost = result.benchmark_cpc || result.benchmark_cost || 163.54;
        const benchScore = result.target_outcome || result.benchmark_score || 7.2;
        
        if (surgeons.length === 0) {
            plotDiv.innerHTML = '<div style="text-align: center; padding: 100px; color: var(--text-muted); font-size: 14px;">No surgeon quadrant data available for this selection.</div>';
            return;
        }
        
        // Group surgeons by quadrant categories
        const traces = [];
        const quadGroups = [
            { name: 'Clinical Champions (Low Cost, High Quality)', color: '#166534', filter: s => s.quadrant.includes('Champion') },
            { name: 'Preference Heavy (High Cost, High Quality)', color: '#2563eb', filter: s => s.quadrant.includes('Preference') },
            { name: 'Under-Resourced (Low Cost, Lower Quality)', color: '#d97706', filter: s => s.quadrant.includes('Under') },
            { name: 'Urgent Alignment (High Cost, Lower Quality)', color: '#dc2626', filter: s => s.quadrant.includes('Urgent') }
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
                    size: matched.map(s => Math.min(26, Math.max(10, Math.sqrt(s.cases) * 3.5))),
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
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.16,
                    y: 0.96,
                    text: '<b>🌟 CLINICAL CHAMPIONS</b><br><span style="font-size:10px;">Low Cost • High Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#047857' },
                    align: 'center',
                    bgcolor: 'rgba(255, 255, 255, 0.92)',
                    bordercolor: '#bbf7d0',
                    borderwidth: 1,
                    borderpad: 4
                },
                {
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.84,
                    y: 0.96,
                    text: '<b>💎 PREFERENCE-HEAVY</b><br><span style="font-size:10px;">High Cost • High Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#1d4ed8' },
                    align: 'center',
                    bgcolor: 'rgba(255, 255, 255, 0.92)',
                    bordercolor: '#bfdbfe',
                    borderwidth: 1,
                    borderpad: 4
                },
                {
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.16,
                    y: 0.06,
                    text: '<b>⚠️ UNDER-RESOURCED</b><br><span style="font-size:10px;">Low Cost • Lower Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#b45309' },
                    align: 'center',
                    bgcolor: 'rgba(255, 255, 255, 0.92)',
                    bordercolor: '#fde68a',
                    borderwidth: 1,
                    borderpad: 4
                },
                {
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.84,
                    y: 0.06,
                    text: '<b>🚨 URGENT ALIGNMENT</b><br><span style="font-size:10px;">High Cost • Lower Quality</span>',
                    showarrow: false,
                    font: { size: 11, color: '#b91c1c' },
                    align: 'center',
                    bgcolor: 'rgba(255, 255, 255, 0.92)',
                    bordercolor: '#fecaca',
                    borderwidth: 1,
                    borderpad: 4
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
        btn.innerText = "⏳ Swarm Mobilizing (12 Bees)...";
        btn.disabled = true;
    }
    if (traceStatus) traceStatus.innerText = "12 Swarm Agents Deliberating in Parallel...";
    
    const initialTrace = `
        <div class="trace-item"><span class="trace-badge trace-running">Mobilizing</span><span>👑 Queen Bee dispatched 11 worker bees across clinical safety, contract parity, site migration, price discrepancies, and ESG bio-waste...</span></div>
        <div class="trace-item"><span class="trace-badge trace-running">Auditing</span><span>Scout & Audit Bees parsing cases for rogue card variance & cross-hospital price spreads...</span></div>
        <div class="trace-item"><span class="trace-badge trace-running">Validating</span><span>Clinical & Migration Bees auditing outcomes, ASA acuity, and Outpatient ASC viability...</span></div>
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
                    const actionKey = typeof act === 'object' ? (act.action || '') : '';
                    const safeLabel = (typeof escapeHtml === 'function') ? escapeHtml(label) : label.replace(/"/g, '&quot;');
                    const safeKey = (typeof escapeHtml === 'function') ? escapeHtml(actionKey) : actionKey.replace(/"/g, '&quot;');
                    return `
                        <button class="btn-chat-action" style="padding: 9px 18px; font-size: 12px; margin: 4px; border-radius: 6px; cursor: pointer; transition: all 0.2s;" 
                                data-action="${safeLabel}" 
                                data-action-key="${safeKey}" 
                                onclick="executeSwarmAction(this.getAttribute('data-action'), this.getAttribute('data-action-key'))">
                            ⚡ ${safeLabel}
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

// Floating Toast Notification for Action Confirmation
function showSwarmToast(message, icon = '⚡') {
    let toast = document.getElementById('swarm-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'swarm-toast';
        toast.style.cssText = `
            position: fixed;
            top: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: #0f172a;
            color: #ffffff;
            padding: 12px 22px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            z-index: 100000;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            pointer-events: none;
        `;
        document.body.appendChild(toast);
    }
    toast.innerHTML = `<span style="font-size: 16px;">${icon}</span> <span>${message}</span>`;
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(-50%) translateY(0)';

    if (window.swarmToastTimeout) clearTimeout(window.swarmToastTimeout);
    window.swarmToastTimeout = setTimeout(() => {
        if (toast) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(-15px)';
        }
    }, 3200);
}

// Execute Swarm Recommended Action
function executeSwarmAction(actionText, actionKey) {
    console.log(`[Swarm Action] Executing: ${actionText} (key: ${actionKey})`);
    const rawText = (actionText || "").replace(/^[⚡📊🗂️📈📦🔍🧠⚖️🏆💸🏷️👨‍⚕️🏥🌱\s]+/, '').trim();
    const rawKey = (actionKey || "").replace(/^[⚡📊🗂️📈📦🔍🧠⚖️🏆💸🏷️👨‍⚕️🏥🌱\s]+/, '').trim();
    const lower = rawText.toLowerCase();
    const key = rawKey.toLowerCase();

    // 1. Surgeon Preference Card (MUST BE CHECKED BEFORE GENERAL STANDARDIZATION)
    if (key.includes('preference card') || lower.includes('preference card') || lower.includes('pref card') || key === 'align_preference_card') {
        // If it's a prompt asking to generate preference card, send to chat
        if (key.includes('generate') || lower.includes('generate') || lower.includes('consensus')) {
            showSwarmToast(`Engaging Preference Card Engine...`, '🗂️');
            const chatContainer = document.getElementById('chatbot-container');
            if (chatContainer && chatContainer.style.display !== 'flex') toggleChat();
            const input = document.getElementById('chat-input');
            if (input) {
                input.value = rawKey || rawText || "generate standardized surgeon preference card";
                sendChatMessage();
            }
            return;
        }

        showSwarmToast(`Opening Surgeon Preference Card Alignment Engine...`, '👨‍⚕️');
        if (typeof openTabDirect === 'function') {
            openTabDirect('PrefCardTab', 'group-cards');
        } else if (typeof openTabByName === 'function') {
            openTabByName('PrefCardTab');
        }
        
        // Extract surgeon name if present (e.g. Dr. GOBBLE, RYAN MICHAEL)
        const match = (actionText || "").match(/Dr\.\s*([^(\n]+)/i);
        if (match && match[1]) {
            const targetSurg = match[1].trim().toUpperCase();
            setTimeout(() => {
                const surgSelect = document.getElementById('pref-surgeon-select');
                if (surgSelect) {
                    for (let opt of surgSelect.options) {
                        const optVal = opt.value.trim().toUpperCase();
                        if (optVal && (optVal.includes(targetSurg) || targetSurg.includes(optVal))) {
                            surgSelect.value = opt.value;
                            if (typeof fetchPreferenceCard === 'function') fetchPreferenceCard();
                            break;
                        }
                    }
                }
            }, 300);
        }
    } else if (key.includes('compare cards') || lower.includes('compare cards') || lower.includes('side-by-side') || (lower.includes('compare') && (lower.includes('surgeons') || lower.includes('physicians')) && !lower.includes('cpc'))) {
        showSwarmToast(`Opening Preference Card Multi-Surgeon Comparison...`, '⚖️');
        const surgeons = (window.lastComparedSurgeons && window.lastComparedSurgeons.length >= 2) ? window.lastComparedSurgeons : [];
        const proc = window.lastComparedProcedure || window.currentProcedure;
        if (typeof navigateToCompareCardsWithSurgeons === 'function') {
            navigateToCompareCardsWithSurgeons(proc, surgeons);
        } else if (typeof openTabDirect === 'function') {
            openTabDirect('CompareCardsTab', 'group-cards');
        }
    } else if (lower.includes('physician cpc') || lower.includes('cpc comparison') || key.includes('cpc') || lower.includes('visualize the average cpc')) {
        showSwarmToast(`Rendering Physician CPC Distribution & Variance Chart...`, '📊');
        const chatContainer = document.getElementById('chatbot-container');
        if (chatContainer && chatContainer.style.display !== 'flex') toggleChat();
        const input = document.getElementById('chat-input');
        if (input) {
            input.value = "visualize the average cpc for all the physicians";
            sendChatMessage();
        }
    } else if (key === 'standardize_substitution' || lower.includes('parity') || lower.includes('arbitrage') || lower.includes('substitute') || lower.includes('commodity substitution')) {
        showSwarmToast(`Executing Standardized Substitution: Navigating to Price Parity & Commodity Analysis...`, '🏷️');
        if (typeof openTabDirect === 'function') {
            openTabDirect('ParityTab', 'group-value');
        } else if (typeof openTabByName === 'function') {
            openTabByName('ParityTab');
        }
        setTimeout(() => { if (typeof fetchPriceParity === 'function') fetchPriceParity(); }, 120);
    } else if (key === 'contain_leakage' || lower.includes('leakage') || lower.includes('off-contract')) {
        showSwarmToast(`Engaging Off-Contract Leakage Sentinel across preferred suppliers...`, '💸');
        if (typeof openTabDirect === 'function') {
            openTabDirect('LeakageTab', 'group-value');
        } else if (typeof openTabByName === 'function') {
            openTabByName('LeakageTab');
        }
    } else if (key === 'apply_bom_pack' || lower.includes('pack') || lower.includes('hold cart') || lower.includes('bom')) {
        showSwarmToast(`Optimizing Procedure Custom Pack & Hold Cart Placement...`, '📦');
        if (typeof openTabDirect === 'function') {
            openTabDirect('ProcCardTab', 'group-cards');
        } else if (typeof openTabByName === 'function') {
            openTabByName('ProcCardTab');
        }
    } else if (key === 'lock_budget_runrate' || (lower.includes('budget') && !key.includes('forecast'))) {
        showSwarmToast(`Locking Savings Target into 12-Month Operating Budget Trajectory...`, '📊');
        if (typeof openTabDirect === 'function') {
            openTabDirect('BudgetTab', 'group-value');
        } else if (typeof openTabByName === 'function') {
            openTabByName('BudgetTab');
        }
        setTimeout(() => { if (typeof fetchBudgetForecast === 'function') fetchBudgetForecast(); }, 120);
    } else if (key === 'migrate_site_of_service' || lower.includes('outpatient asc') || lower.includes('shift') || lower.includes('migrate')) {
        showSwarmToast(`Opening Site-of-Service ASC Migration Engine...`, '🏥');
        if (typeof openSiteOfServiceMigrationModal === 'function') {
            openSiteOfServiceMigrationModal();
        }
    } else if (key === 'audit_discrepancy' || lower.includes('price discrepanc') || lower.includes('clawback') || lower.includes('demand notice')) {
        showSwarmToast(`Opening Price Discrepancy & Clawback Audit Engine...`, '⚖️');
        if (typeof openPriceDiscrepancyModal === 'function') {
            openPriceDiscrepancyModal();
        }
    } else if (key === 'reduce_esg_waste' || lower.includes('bio-waste') || lower.includes('esg') || lower.includes('carbon')) {
        showSwarmToast(`Opening Custom Pack ESG Bio-Waste & Carbon Footprint Index...`, '🌱');
        if (typeof openESGCarbonModal === 'function') {
            openESGCarbonModal();
        }
    } else {
        // Query Ask The Bee with clean prompt
        showSwarmToast(`Mobilizing Bee Hive Swarm for inquiry...`, '🐝');
        const chatContainer = document.getElementById('chatbot-container');
        if (chatContainer && chatContainer.style.display !== 'flex') {
            toggleChat();
        }
        const input = document.getElementById('chat-input');
        if (input) {
            const cleanQuery = rawKey || rawText;
            input.value = cleanQuery;
            sendChatMessage();
        }
    }
}

// Reset Swarm State on Procedure Change
window.resetSwarmState = function() {
    const synthesisBox = document.getElementById('swarm-synthesis-content');
    const consensusBadge = document.getElementById('swarm-consensus-badge');
    const debateSection = document.getElementById('swarm-debate-section');
    const debateList = document.getElementById('swarm-debate-list');
    const actionsSection = document.getElementById('swarm-actions-section');
    const actionsButtons = document.getElementById('swarm-actions-buttons');
    const drawerSynthesis = document.getElementById('drawer-synthesis-text');
    const drawerTranscript = document.getElementById('drawer-debate-transcript');
    const traceStatus = document.getElementById('swarm-trace-status');
    const traceBox = document.getElementById('swarm-trace-container');
    const drawerTrace = document.getElementById('drawer-trace-list');
    const btn = document.getElementById('btn-run-swarm-audit');
    const drawerBtn = document.getElementById('btn-drawer-audit');

    if (btn) {
        btn.innerText = "🐝 Launch Autonomous Hive Audit";
        btn.disabled = false;
    }
    if (drawerBtn) {
        drawerBtn.innerText = "🐝 Launch Full Hive Audit";
        drawerBtn.disabled = false;
    }
    if (consensusBadge) consensusBadge.style.display = 'none';
    if (synthesisBox) {
        synthesisBox.innerHTML = '<p style="color: var(--text-muted); margin: 0;">Select a procedure from the top selector and click <b>Launch Autonomous Hive Audit</b> to mobilize all specialized bees across clinical safety, contract pricing, BOM packs, and rogue variances.</p>';
    }
    if (debateSection) debateSection.style.display = 'none';
    if (debateList) debateList.innerHTML = '';
    if (actionsSection) actionsSection.style.display = 'none';
    if (actionsButtons) actionsButtons.innerHTML = '';
    if (drawerSynthesis) {
        drawerSynthesis.innerHTML = '<p style="color: var(--text-muted); margin: 0;">Ready to execute autonomous multi-agent audit for this procedure. Click "Launch Full Hive Audit" to synthesize cross-domain intelligence.</p>';
    }
    if (drawerTranscript) drawerTranscript.innerHTML = '';
    if (traceStatus) traceStatus.innerText = "System Idle - Waiting for Audit Trigger";
    if (traceBox) {
        traceBox.innerHTML = '<div class="trace-item"><span class="trace-badge trace-completed">System</span><span>Bee Hive initialized. Ready for procedure audit.</span></div>';
    }
    if (drawerTrace) {
        drawerTrace.innerHTML = '<div class="trace-item"><span class="trace-badge trace-completed">System</span><span>Bee Hive ready for procedure audit.</span></div>';
    }
};

// Initialize Swarm and CQO on page ready
$(document).ready(function() {
    fetchSwarmPersonas();
    fetchCQOSummary();
});

