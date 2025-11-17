// demo/static/js/app.js
/*
   Hybrid VDB - Interactive Frontend
   
   Key Features (2025-11-17 Update):
   - TIER 2 hit rate tracking (prefetch effectiveness)
   - Retrieval latency display (excludes LLM)
   - Query count and hit count display
   - Learning curve with TIER 2 prefetch rate
   - Anchor details and velocity equation display
   
   Author: Saberzerker
   Date: 2025-11-17
*/

// ═══════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════

let learningCurveChart = null;
let anchorNetwork = null;
let queryHistory = [];

// ═══════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    console.log('[APP] Initializing frontend...');
    
    // Initialize chart
    initLearningCurveChart();
    
    // Initialize anchor graph
    initAnchorGraph();
    
    // Event listeners
    document.getElementById('query-btn').addEventListener('click', handleQuery);
    document.getElementById('query-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleQuery();
    });
    
    // Suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('query-input').value = btn.dataset.query;
            handleQuery();
        });
    });
    
    console.log('[APP] Frontend initialized ✓');
});

// ═══════════════════════════════════════════════════════════
// QUERY HANDLING
// ═══════════════════════════════════════════════════════════

async function handleQuery() {
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    console.log(`[QUERY] "${question}"`);
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log('[RESPONSE]', data);
        
        // Update UI
        displayAnswer(data.answer, data.retrieved_docs);
        displayFlow(data.flow);
        updateMetrics(data.metrics);
        updateLearningCurve(data.metrics);
        displayAnchorDetails(data.metrics);
        displayVelocityEquation(data.metrics.total_queries);
        
        // Store in history
        queryHistory.push({
            question,
            timestamp: Date.now(),
            metrics: data.metrics
        });
        
        // Fetch anchor graph
        await updateAnchorGraph();
        
    } catch (error) {
        console.error('[ERROR]', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// ═══════════════════════════════════════════════════════════
// UI UPDATES
// ═══════════════════════════════════════════════════════════

function displayAnswer(answer, docs) {
    const answerContent = document.getElementById('answer-content');
    answerContent.innerHTML = `<p>${answer}</p>`;
    
    // Display retrieved docs
    const docsSection = document.getElementById('retrieved-docs');
    const docsList = document.getElementById('docs-list');
    
    if (docs && docs.length > 0) {
        docsSection.style.display = 'block';
        docsList.innerHTML = docs.map(doc => 
            `<div class="doc-item">${doc}</div>`
        ).join('');
    } else {
        docsSection.style.display = 'none';
    }
}

function displayFlow(steps) {
    const flowSteps = document.getElementById('flow-steps');
    
    if (!steps || steps.length === 0) {
        flowSteps.innerHTML = '<p class="placeholder">No flow data</p>';
        return;
    }
    
    flowSteps.innerHTML = steps.map(step => `
        <div class="flow-step ${step.status}" style="animation-delay: ${step.step * 0.1}s">
            <div class="flow-step-content">
                <div class="flow-step-name">${step.step}. ${step.name}</div>
                <div class="flow-step-detail">${step.detail}</div>
            </div>
            <div class="flow-step-time">${step.time_ms}ms</div>
        </div>
    `).join('');
}

function updateMetrics(metrics) {
    // TIER 2 Hit Rate (prefetch performance) - THIS IS THE LEARNING METRIC!
    const tier2Rate = metrics.tier2_prefetch_rate || 0;
    document.getElementById('local-hit-rate').textContent = 
        `${tier2Rate.toFixed(1)}%`;
    
    // Show hit count detail (X/Y format)
    const tier2Hits = metrics.tier2_hits || 0;
    const tier2Attempts = metrics.tier2_attempts || 0;
    document.getElementById('hit-detail').textContent = 
        `${tier2Hits}/${tier2Attempts} from prefetch`;
    
    // Avg retrieval latency (excluding LLM)
    document.getElementById('avg-latency').textContent = 
        `${metrics.avg_retrieval_latency_ms.toFixed(1)}ms`;
    
    // Query count
    document.getElementById('query-count').textContent = 
        `${metrics.total_queries} queries processed`;
    
    // Dynamic space
    document.getElementById('dynamic-space').textContent = 
        `${metrics.dynamic_vectors}/${metrics.dynamic_capacity}`;
    
    const fillRate = (metrics.dynamic_vectors / metrics.dynamic_capacity) * 100;
    document.getElementById('dynamic-progress').style.width = `${fillRate}%`;
    
    // Anchors
    document.getElementById('anchors-count').textContent = 
        `${metrics.total_anchors} / ${metrics.active_predictions}`;
    
    // Prefetch efficiency
    document.getElementById('prefetch-efficiency').textContent = 
        `${metrics.prefetch_cache_hit_rate.toFixed(1)}%`;
}

function displayAnchorDetails(metrics) {
    const container = document.getElementById('anchor-details');
    
    if (!metrics.total_anchors || metrics.total_anchors === 0) {
        container.innerHTML = '<p class="placeholder">No anchors created yet...</p>';
        return;
    }
    
    // Show anchor summary
    container.innerHTML = `
        <div class="anchor-card">
            <div class="anchor-header">
                <div class="anchor-id">📍 Total Anchors: ${metrics.total_anchors}</div>
                <div class="anchor-strength">${metrics.active_predictions} Active Predictions</div>
            </div>
            <div class="anchor-body">
                <p><strong>Prediction Accuracy:</strong> ${metrics.prediction_accuracy}%</p>
                <p><strong>Prefetch Cache Hit Rate:</strong> ${metrics.prefetch_cache_hit_rate}%</p>
                <div class="anchor-predictions">
                    <strong>Status:</strong> 
                    ${metrics.active_predictions > 0 
                        ? `✅ System is predicting future queries (${metrics.active_predictions} trajectories active)` 
                        : '⏳ Generating first predictions...'}
                </div>
            </div>
        </div>
    `;
}

function displayVelocityEquation(queryNumber) {
    const container = document.getElementById('velocity-display');
    
    if (queryNumber === 0) {
        container.innerHTML = '<p class="placeholder">Submit a query to see prediction math...</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="equation">
            <div style="margin-bottom: 1rem;">
                <span class="equation-var">v⃗</span><sub>prediction</sub> = 
                <span class="equation-var">v⃗</span><sub>current</sub> + 
                <span class="equation-var">α</span> × <span class="equation-var">Δ⃗</span><sub>momentum</sub> + 
                <span class="equation-var">β</span> × <span class="equation-var">C⃗</span><sub>centroid</sub>
            </div>
        </div>
        <div class="equation-description">
            <p><strong>Where:</strong></p>
            <ul style="list-style: none; padding-left: 0;">
                <li>• <span class="equation-var">v⃗</span><sub>current</sub> = Current query vector (384-dim)</li>
                <li>• <span class="equation-var">Δ⃗</span><sub>momentum</sub> = Velocity from previous query</li>
                <li>• <span class="equation-var">C⃗</span><sub>centroid</sub> = Semantic cluster center (gravity)</li>
                <li>• <span class="equation-var">α</span> = Momentum weight (0.7 in cold start, 0.9 in steady state)</li>
                <li>• <span class="equation-var">β</span> = Centroid bias (0.3 → pulls toward cluster)</li>
            </ul>
            <p style="margin-top: 1rem; font-style: italic; color: var(--info);">
                🎯 This equation predicts where you'll query next based on your trajectory!
            </p>
            <p style="margin-top: 0.5rem; font-style: italic; color: var(--success);">
                ✅ We prefetch NEIGHBORHOODS (k=3) around each prediction, creating coverage zones!
            </p>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════
// LEARNING CURVE CHART
// ═══════════════════════════════════════════════════════════

function initLearningCurveChart() {
    const ctx = document.getElementById('learning-curve-chart').getContext('2d');
    
    learningCurveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'TIER 2 Hit Rate (Prefetch Effectiveness) %',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#cbd5e1'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Hit Rate: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: '#334155'
                    },
                    title: {
                        display: true,
                        text: 'Hit Rate (%)',
                        color: '#cbd5e1'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#334155'
                    },
                    title: {
                        display: true,
                        text: 'Query Number',
                        color: '#cbd5e1'
                    }
                }
            }
        }
    });
}

function updateLearningCurve(metrics) {
    const queryNum = `Q${metrics.total_queries}`;
    
    // Use TIER 2 prefetch rate (not local_hit_rate)
    const tier2Rate = metrics.tier2_prefetch_rate || 0;
    
    learningCurveChart.data.labels.push(queryNum);
    learningCurveChart.data.datasets[0].data.push(tier2Rate);
    
    // Keep last 20 queries
    if (learningCurveChart.data.labels.length > 20) {
        learningCurveChart.data.labels.shift();
        learningCurveChart.data.datasets[0].data.shift();
    }
    
    learningCurveChart.update();
}

// ═══════════════════════════════════════════════════════════
// ANCHOR GRAPH
// ═══════════════════════════════════════════════════════════

function initAnchorGraph() {
    const container = document.getElementById('anchor-graph');
    
    const data = {
        nodes: [],
        edges: []
    };
    
    const options = {
        nodes: {
            shape: 'dot',
            size: 20,
            font: {
                size: 14,
                color: '#cbd5e1'
            },
            borderWidth: 2
        },
        edges: {
            width: 2,
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.5
                }
            },
            color: {
                color: '#475569',
                highlight: '#6366f1'
            },
            smooth: {
                type: 'cubicBezier'
            }
        },
        physics: {
            stabilization: {
                iterations: 200
            },
            barnesHut: {
                gravitationalConstant: -2000,
                springConstant: 0.001,
                springLength: 200
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200
        }
    };
    
    anchorNetwork = new vis.Network(container, data, options);
}

async function updateAnchorGraph() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.anchor_graph) {
            const nodes = data.anchor_graph.nodes.map(node => ({
                id: node.id,
                label: `#${node.id}\n${node.hits}H/${node.misses}M`,
                title: `${node.label}<br>Hits: ${node.hits}, Misses: ${node.misses}<br>Type: ${node.type}`,
                color: getAnchorColor(node.type),
                size: 15 + (node.strength * 2)
            }));
            
            const edges = data.anchor_graph.edges.map(edge => ({
                from: edge.from,
                to: edge.to,
                width: 1 + (edge.weight / 20)
            }));
            
            anchorNetwork.setData({ nodes, edges });
        }
    } catch (error) {
        console.error('[GRAPH] Update failed:', error);
    }
}

function getAnchorColor(type) {
    const colors = {
        'weak': '#94a3b8',
        'medium': '#f59e0b',
        'strong': '#8b5cf6',
        'permanent': '#10b981'
    };
    return colors[type] || '#94a3b8';
}

// ═══════════════════════════════════════════════════════════
// LOADING OVERLAY
// ═══════════════════════════════════════════════════════════

function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}