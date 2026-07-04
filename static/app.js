document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const tickerInput = document.getElementById('ticker-input');
    const loadingDiv = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const predictionsGrid = document.getElementById('predictions-grid');
    const summaryTableBody = document.querySelector('#summary-table tbody');
    const assetNameEl = document.getElementById('asset-name');
    const marketPriceEl = document.getElementById('market-price');
    const downloadBtn = document.getElementById('download-btn');
    
    // Tab switching logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            
            // Trigger Plotly resize when tab becomes visible
            if(btn.dataset.tab === 'charts') {
                window.dispatchEvent(new Event('resize'));
            }
        });
    });

    predictBtn.addEventListener('click', handlePredict);
    tickerInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') handlePredict();
    });

    async function handlePredict() {
        const ticker = tickerInput.value.trim().toUpperCase();
        if(!ticker) return;

        // UI Reset
        predictBtn.disabled = true;
        loadingDiv.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        predictionsGrid.innerHTML = '';
        summaryTableBody.innerHTML = '';
        
        try {
            // 1. Fetch Predictions
            const predRes = await fetch(`/api/predict/${ticker}`);
            if(!predRes.ok) {
                const err = await predRes.json();
                throw new Error(err.detail || 'Failed to fetch predictions');
            }
            const predData = await predRes.json();
            
            // Update Headers
            const currency = (ticker.includes('.NS') || ticker.includes('.BO')) ? '₹' : '$';
            assetNameEl.textContent = `Asset: ${predData.ticker}`;
            marketPriceEl.textContent = `Latest Market Price: ${currency}${predData.currentPrice.toFixed(2)}`;
            
            // Build Cards and Table
            predData.predictions.forEach(p => {
                const isPositive = p.returnPct > 0;
                const deltaClass = isPositive ? 'positive' : 'negative';
                const sign = isPositive ? '+' : '';
                
                // Card
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <h4>${p.horizon}</h4>
                    <div class="metric-value">${currency}${p.expectedPrice.toFixed(2)}</div>
                    <div class="metric-delta ${deltaClass}">${sign}${p.priceDelta.toFixed(2)} (${sign}${p.returnPct.toFixed(2)}%)</div>
                    <div class="metric-detail">Signal: <span>${p.signal}</span></div>
                    <div class="metric-detail">Confidence: <span>${p.confidence.toFixed(1)}%</span></div>
                `;
                predictionsGrid.appendChild(card);
                
                // Table Row
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${p.horizon}</td>
                    <td>${currency}${p.expectedPrice.toFixed(2)}</td>
                    <td class="${deltaClass}">${sign}${p.returnPct.toFixed(2)}%</td>
                    <td>${p.signal}</td>
                    <td>${p.confidence.toFixed(1)}%</td>
                `;
                summaryTableBody.appendChild(tr);
            });

            // 2. Fetch Chart Data
            const chartRes = await fetch(`/api/chart/${ticker}`);
            if(!chartRes.ok) {
                console.warn('Could not load chart data');
            } else {
                const chartData = await chartRes.json();
                renderCharts(chartData);
            }

            // Setup download button
            downloadBtn.onclick = () => {
                window.location.href = `/api/download/${ticker}`;
            };

            // Show results
            resultsContainer.classList.remove('hidden');
            window.dispatchEvent(new Event('resize')); // Fix plotly render sizing

        } catch(error) {
            alert(`Error: ${error.message}`);
        } finally {
            predictBtn.disabled = false;
            loadingDiv.classList.add('hidden');
        }
    }

    function renderCharts(data) {
        const layoutBase = {
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#cbd5e1' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
            margin: { t: 20, r: 20, l: 40, b: 40 }
        };

        // 1. Candlestick + BB
        const traceCandle = {
            x: data.dates,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            type: 'candlestick',
            name: 'Price'
        };
        const traceBBUpper = {
            x: data.dates, y: data.bb_upper,
            type: 'scatter', mode: 'lines',
            line: { color: 'rgba(255, 255, 255, 0.3)' },
            name: 'BB Upper'
        };
        const traceBBLower = {
            x: data.dates, y: data.bb_lower,
            type: 'scatter', mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(255, 255, 255, 0.1)',
            line: { color: 'rgba(255, 255, 255, 0.3)' },
            name: 'BB Lower'
        };
        
        Plotly.newPlot('chart-main', [traceCandle, traceBBUpper, traceBBLower], {
            ...layoutBase,
            xaxis: { ...layoutBase.xaxis, rangeslider: { visible: false } }
        }, {responsive: true});

        // 2. RSI
        const traceRSI = {
            x: data.dates, y: data.rsi,
            type: 'scatter', mode: 'lines',
            line: { color: '#a855f7' },
            name: 'RSI'
        };
        Plotly.newPlot('chart-rsi', [traceRSI], {
            ...layoutBase,
            shapes: [
                { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 70, y1: 70, line: { color: '#ef4444', dash: 'dash' } },
                { type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], y0: 30, y1: 30, line: { color: '#22c55e', dash: 'dash' } }
            ]
        }, {responsive: true});

        // 3. MACD
        const traceMACD = {
            x: data.dates, y: data.macd,
            type: 'scatter', mode: 'lines', line: { color: '#3b82f6' }, name: 'MACD'
        };
        const traceSignal = {
            x: data.dates, y: data.macd_signal,
            type: 'scatter', mode: 'lines', line: { color: '#f97316' }, name: 'Signal'
        };
        const traceHist = {
            x: data.dates, y: data.macd_hist,
            type: 'bar', marker: { color: '#64748b' }, name: 'Histogram'
        };
        Plotly.newPlot('chart-macd', [traceMACD, traceSignal, traceHist], layoutBase, {responsive: true});
    }
});
