import React, { useState, useEffect, useMemo } from 'react';
import { useData, API_BASE } from '../hooks/useData';
import { Scatter, Bar } from 'react-chartjs-2';

const FeatureAnalysis = () => {
    const { backendData, loading: globalLoading } = useData();
    const [pcaData, setPcaData] = useState(null);
    const [chiData, setChiData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!backendData) return;

        const fetchData = async () => {
            setLoading(true);
            try {
                const [pcaRes, chiRes] = await Promise.all([
                    fetch(`${API_BASE}/api/pca`),
                    fetch(`${API_BASE}/api/chi-square`)
                ]);
                const pca = await pcaRes.json();
                const chi = await chiRes.json();
                setPcaData(pca);
                setChiData(chi);
            } catch (err) {
                console.error("Analysis error:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [backendData]);

    const chartsData = useMemo(() => {
        if (!pcaData) return null;

        return {
            scatter: {
                datasets: [{
                    label: 'PCA Components (PC1 vs PC2)',
                    data: (pcaData.components || []).map(row => ({ x: row[0], y: row[1] })),
                    backgroundColor: 'rgba(79, 70, 229, 0.6)',
                    pointRadius: 4,
                }]
            },
            variances: {
                labels: ['PC1', 'PC2'],
                datasets: [{
                    label: 'Explained Variance Ratio',
                    data: pcaData.variance || [],
                    backgroundColor: '#10B981'
                }]
            }
        };
    }, [pcaData]);

    if (globalLoading && !backendData) return <div className="card">Loading Dataset...</div>;
    if (!backendData) return <div className="card">Connecting to Python Backend...</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="card">
                <h2>Feature Selection & PCA</h2>
                <p style={{ color: 'var(--text-muted)' }}>Feature ranking and dimensionality reduction using Python-based ML libraries.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Chi-Square Scores</h3>
                    {loading ? 'Calculating scores...' : (chiData ? (
                        <table style={{ width: '100%', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <th style={{ textAlign: 'left', padding: '0.5rem', width: '60px' }}>Sl.No</th>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>Feature</th>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>Chi-Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                {chiData.features.map((f, i) => (
                                    <tr key={i}>
                                        <td style={{ padding: '0.5rem', color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td style={{ padding: '0.5rem' }}>{f}</td>
                                        <td style={{ padding: '0.5rem', fontWeight: 'bold', color: 'var(--primary)' }}>{chiData.chi_scores[i].toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : 'No data')}
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>PCA Components (2D Map)</h3>
                    {loading ? 'Calculating PCA...' : (chartsData ? (
                        <Scatter data={chartsData.scatter} options={{ responsive: true }} />
                    ) : 'No data')}
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="card">
                    <h3>Explained Variance Ratio</h3>
                    {loading ? 'Calculating...' : (chartsData ? (
                        <Bar data={chartsData.variances} options={{ responsive: true, scales: { y: { max: 1 } } }} />
                    ) : 'No data')}
                </div>
                <div className="card">
                    <h3>Python Analysis Insight</h3>
                    <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                        The Chi-Square test measures statistical dependency between numerical/categorical features and the <strong>Category</strong> target.
                        PCA reduces the feature space while retaining maximum information.
                    </p>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', fontSize: '0.85rem' }}>
                        <strong>Target:</strong> Category (Encoded) <br />
                        <strong>Scaling:</strong> MinMaxScaler + Label Encoding Applied
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FeatureAnalysis;
