import React, { useState, useEffect, useMemo } from 'react';
import { useData, API_BASE } from '../hooks/useData';
import { Line } from 'react-chartjs-2';

const TimeSeries = () => {
    const { backendData, loading: globalLoading } = useData();
    const [smoothingType, setSmoothingType] = useState('none');
    const [metric, setMetric] = useState('revenue');
    const [windowSize, setWindowSize] = useState(3);
    const [dateRange, setDateRange] = useState({ start: '', end: '' });

    const [tsData, setTsData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!backendData) return;

        const fetchTS = async () => {
            setLoading(true);
            try {
                const query = new URLSearchParams({
                    method: smoothingType,
                    metric: metric,
                    window: windowSize,
                    start: dateRange.start,
                    end: dateRange.end
                });
                const response = await fetch(`${API_BASE}/api/time-series?${query}`);
                const data = await response.json();
                setTsData(data);
            } catch (err) {
                console.error("Time series error:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchTS();
    }, [backendData, smoothingType, metric, windowSize, dateRange]);

    const chartData = useMemo(() => {
        if (!tsData) return { labels: [], datasets: [] };

        const label = metric === 'revenue' ? 'Revenue (Rs.)' : 'Transaction Count';

        return {
            labels: tsData.labels,
            datasets: [
                {
                    label: `Original ${label}`,
                    data: tsData.original,
                    borderColor: 'rgba(79, 70, 229, 0.3)',
                    borderDash: [5, 5],
                    tension: 0.4
                },
                {
                    label: smoothingType === 'none' ? `Current ${label}` : `Smoothed (${smoothingType.toUpperCase()})`,
                    data: tsData.smoothed,
                    borderColor: '#22D3EE',
                    backgroundColor: 'rgba(34, 211, 238, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        };
    }, [tsData, smoothingType, metric]);

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { labels: { color: '#94A3B8' } }
        },
        scales: {
            x: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94A3B8' } },
            y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94A3B8' } }
        }
    };

    if (globalLoading && !backendData) return <div className="card">Loading Dataset...</div>;
    if (!backendData) return <div className="card">Connecting to Python Backend...</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="card">
                <h2>Time Series Analysis (Enhanced Controls)</h2>
                <p style={{ color: 'var(--text-muted)' }}>Analyze monthly revenue and volume trends with customizable smoothing techniques.</p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginTop: '1.5rem' }}>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>📅 Date Range</label>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <input type="date" value={dateRange.start} onChange={e => setDateRange(prev => ({ ...prev, start: e.target.value }))} style={{ width: '100%', padding: '0.4rem', borderRadius: '4px', background: 'var(--bg-app)', color: 'white', border: '1px solid var(--border)' }} />
                            <input type="date" value={dateRange.end} onChange={e => setDateRange(prev => ({ ...prev, end: e.target.value }))} style={{ width: '100%', padding: '0.4rem', borderRadius: '4px', background: 'var(--bg-app)', color: 'white', border: '1px solid var(--border)' }} />
                        </div>
                    </div>

                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>📊 Select Metric</label>
                        <select value={metric} onChange={e => setMetric(e.target.value)} style={{ width: '100%' }}>
                            <option value="revenue">Final Price (Revenue)</option>
                            <option value="transactions">Transaction Count</option>
                        </select>
                    </div>

                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>⚙️ Smoothing Method</label>
                        <select value={smoothingType} onChange={e => setSmoothingType(e.target.value)} style={{ width: '100%' }}>
                            <option value="none">None</option>
                            <option value="mean">Mean Smoothing</option>
                            <option value="median">Median Smoothing</option>
                            <option value="boundary">Boundary Smoothing</option>
                        </select>
                    </div>

                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>🔢 Window Size: {windowSize}</label>
                        <input type="range" min="3" max="7" step="2" value={windowSize} onChange={e => setWindowSize(parseInt(e.target.value))} style={{ width: '100%' }} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', marginTop: '0.2rem' }}>
                            <span>3</span>
                            <span>5</span>
                            <span>7</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="card">
                {loading ? <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Processing...</div> : (
                    <Line data={chartData} options={chartOptions} />
                )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="card">
                    <h3>Processing Insights</h3>
                    <ul style={{ color: 'var(--text-muted)', marginTop: '1rem', lineHeight: '1.8' }}>
                        <li><strong>Environment:</strong> Python 3 / Pandas / NumPy</li>
                        <li><strong>Engine:</strong> Rolling Window Calculation (Centered)</li>
                        <li><strong>Data Continuity:</strong> Handled by min_periods=1</li>
                    </ul>
                </div>
                <div className="card">
                    <h3>Analytical Overview</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Total Data Points</span>
                            <strong>{tsData?.labels?.length || 0} Months</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Calculation Time</span>
                            <strong>~2ms (Python)</strong>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TimeSeries;
