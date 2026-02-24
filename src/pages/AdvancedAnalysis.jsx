import React, { useState, useEffect, useRef } from 'react';
import {
    Chart as ChartJS,
    ScatterController,
    PointElement,
    LineElement,
    LinearScale,
    Tooltip,
    Legend,
    Title
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { API_BASE } from '../hooks/useData';

ChartJS.register(ScatterController, PointElement, LineElement, LinearScale, Tooltip, Legend, Title);

const CLUSTER_COLORS = [
    '#6366F1', '#22D3EE', '#F472B6', '#FBBF24', '#10B981', '#8B5CF6', '#EF4444', '#F97316'
];

const cardStyle = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '16px',
    padding: '1.5rem'
};

const labelStyle = { display: 'block', fontSize: '0.85rem', marginBottom: '0.4rem', color: 'var(--text-muted)' };

const inputStyle = {
    background: 'var(--bg-main)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    borderRadius: '8px',
    padding: '0.5rem 0.75rem',
    width: '100%'
};

const btnStyle = {
    background: 'var(--primary)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    padding: '0.6rem 1.5rem',
    cursor: 'pointer',
    fontWeight: 600
};

const statBox = (label, value, color = 'var(--accent)') => (
    <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(99,102,241,0.08)', borderRadius: '10px' }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color }}>{value}</div>
    </div>
);

// ----- Linear Regression Section -----
const RegressionSection = () => {
    const [feature, setFeature] = useState('Discount (%)');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetchRegression = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/regression?feature=${encodeURIComponent(feature)}`);
            const json = await res.json();
            setData(json);
        } catch (e) {
            console.error('Regression fetch error:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchRegression(); }, []);

    const chartData = data ? {
        datasets: [
            {
                label: 'Data Points',
                data: data.scatter,
                backgroundColor: 'rgba(99,102,241,0.5)',
                pointRadius: 4
            },
            {
                label: 'Regression Line',
                data: data.line,
                type: 'line',
                borderColor: '#22D3EE',
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.1
            }
        ]
    } : { datasets: [] };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94A3B8' } } },
        scales: {
            x: { title: { display: true, text: feature, color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
            y: { title: { display: true, text: 'Final Price (Rs.)', color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } }
        }
    };

    return (
        <div style={cardStyle}>
            <h2 style={{ marginBottom: '0.5rem' }}>Linear Regression</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                Predict <strong>Final Price</strong> from a selected feature using Ordinary Least Squares.
            </p>

            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: '180px' }}>
                    <label style={labelStyle}>Input Feature (X-axis)</label>
                    <select value={feature} onChange={e => setFeature(e.target.value)} style={inputStyle}>
                        <option value="Discount (%)">Discount (%)</option>
                        <option value="Price (Rs.)">Price (Rs.)</option>
                    </select>
                </div>
                <button style={btnStyle} onClick={fetchRegression} disabled={loading}>
                    {loading ? 'Running...' : 'Run Regression'}
                </button>
            </div>

            <div style={{ height: '380px', marginBottom: '1.5rem' }}>
                {data ? <Scatter data={chartData} options={chartOptions} /> : <div style={{ color: 'var(--text-muted)' }}>Loading...</div>}
            </div>

            {data && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                    {statBox('Coefficient (m)', data.coef.toFixed(4))}
                    {statBox('Intercept (b)', data.intercept.toFixed(2), '#F472B6')}
                    {statBox('R² Score', data.r2.toFixed(4), '#10B981')}
                </div>
            )}
        </div>
    );
};

// ----- K-Means Clustering Section -----
const ClusteringSection = () => {
    const [k, setK] = useState(3);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetchClusters = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/cluster?n_clusters=${k}`);
            const json = await res.json();
            setData(json);
        } catch (e) {
            console.error('Cluster fetch error:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchClusters(); }, [k]);

    const buildChartData = () => {
        if (!data) return { datasets: [] };

        const datasets = [];
        for (let i = 0; i < data.n_clusters; i++) {
            datasets.push({
                label: `Cluster ${i + 1}`,
                data: data.clusters.filter(p => p.cluster === i),
                backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] + 'BB',
                pointRadius: 5
            });
        }
        // Centroids
        datasets.push({
            label: 'Centroids',
            data: data.centers,
            backgroundColor: '#FFFFFF',
            pointRadius: 12,
            pointStyle: 'star',
            borderColor: '#FFFFFF',
            borderWidth: 2
        });
        return { datasets };
    };

    const clusterOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94A3B8', boxWidth: 12 } } },
        scales: {
            x: { title: { display: true, text: 'PCA Component 1', color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
            y: { title: { display: true, text: 'PCA Component 2', color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } }
        }
    };

    return (
        <div style={cardStyle}>
            <h2 style={{ marginBottom: '0.5rem' }}>K-Means Clustering</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                Segment customers into <strong>k groups</strong> based on Price, Final Price, and Discount. Visualized using PCA.
            </p>

            <div style={{ marginBottom: '1.5rem', maxWidth: '260px' }}>
                <label style={labelStyle}>Select Number of Clusters (k)</label>
                <select
                    value={k}
                    onChange={e => setK(parseInt(e.target.value))}
                    style={inputStyle}
                >
                    <option value={2}>2 Clusters</option>
                    <option value={3}>3 Clusters</option>
                    <option value={4}>4 Clusters</option>
                    <option value={5}>5 Clusters</option>
                </select>
            </div>

            <div style={{ height: '400px', marginBottom: '1.5rem' }}>
                {data ? <Scatter data={buildChartData()} options={clusterOptions} /> : <div style={{ color: 'var(--text-muted)' }}>Loading...</div>}
            </div>

            {data && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                    {statBox('Clusters (k)', data.n_clusters, '#6366F1')}
                    {statBox('Inertia (WCSS)', data.inertia.toFixed(1), '#FBBF24')}
                </div>
            )}
        </div>
    );
};

// ----- Main Page -----
const AdvancedAnalysis = () => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={cardStyle}>
                <h2>Advanced Analysis</h2>
                <p style={{ color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Explore predictive modelling with <strong>Linear Regression</strong> and unsupervised learning with <strong>K-Means Clustering</strong>.
                </p>
            </div>
            <RegressionSection />
            <ClusteringSection />
        </div>
    );
};

export default AdvancedAnalysis;
