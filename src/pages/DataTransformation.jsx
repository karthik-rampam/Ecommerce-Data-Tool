import React, { useState, useEffect, useMemo } from 'react';
import { useData } from '../hooks/useData';
import { Bar, Line } from 'react-chartjs-2';

const DataTransformation = () => {
    const { backendData, loading: globalLoading } = useData();
    const [transData, setTransData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!backendData) return;

        const fetchTrans = async () => {
            setLoading(true);
            try {
                const response = await fetch('http://localhost:5000/api/transformation');
                const data = await response.json();
                setTransData(data);
            } catch (err) {
                console.error("Transformation error:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchTrans();
    }, [backendData]);

    const chartsData = useMemo(() => {
        if (!transData) return null;

        const binLabels = Object.keys(transData.binning || {});
        const binCounts = Object.values(transData.binning || {});

        return {
            binning: {
                labels: binLabels,
                datasets: [{
                    label: 'Count of Products',
                    data: binCounts,
                    backgroundColor: ['#4F46E5', '#22D3EE', '#F472B6', '#10B981'],
                    borderRadius: 8
                }]
            },
            normalization: {
                labels: Array.from({ length: (transData.min_max || []).length }, (_, i) => `P${i + 1}`),
                datasets: [
                    {
                        label: 'Min-Max Scaling',
                        data: transData.min_max,
                        borderColor: '#4F46E5',
                        backgroundColor: '#4F46E5',
                        pointRadius: 4,
                        tension: 0.3
                    },
                    {
                        label: 'Z-Score (Standardized)',
                        data: transData.z_score,
                        borderColor: '#22D3EE',
                        backgroundColor: '#22D3EE',
                        pointRadius: 4,
                        tension: 0.3
                    },
                    {
                        label: 'Decimal Scaling',
                        data: transData.decimal_scaling,
                        borderColor: '#F472B6',
                        backgroundColor: '#F472B6',
                        pointRadius: 4,
                        tension: 0.3
                    }
                ]
            }
        };
    }, [transData]);

    if (globalLoading && !backendData) return <div className="card">Loading Dataset...</div>;
    if (!backendData) return <div className="card">Connecting to Python Backend...</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="card">
                <h2>Data Transformation</h2>
                <p style={{ color: 'var(--text-muted)' }}>Categorize and normalize data using the exact ranges specified in mmm.pdf.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Price Binning Distribution</h3>
                    {loading ? 'Binning data...' : (chartsData ? <Bar data={chartsData.binning} options={{ responsive: true }} /> : 'Waiting...')}
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Normalization (First 20 Samples)</h3>
                    {loading ? 'Normalizing...' : (chartsData ? <Line data={chartsData.normalization} options={{ responsive: true }} /> : 'Waiting...')}
                </div>
            </div>

            <div className="card">
                <h3>Transformation Logic Details</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1rem' }}>
                    <div>
                        <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Binning Strategy (from PDF)</h4>
                        <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                            <li><strong>Low:</strong> Rs. 0 - 100</li>
                            <li><strong>Medium:</strong> Rs. 100 - 200</li>
                            <li><strong>High:</strong> Rs. 200 - 300</li>
                            <li><strong>Premium:</strong> Rs. 300+</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--accent)', marginBottom: '0.5rem' }}>Normalization Methods</h4>
                        <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                            <li><strong>Min-Max:</strong> Rescales data to [0, 1] range.</li>
                            <li><strong>Z-Score:</strong> Standardized using mean and std.</li>
                            <li><strong>Decimal Scaling:</strong> Rescaled by 10^j factor.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DataTransformation;
