import React, { useState } from 'react';
import { useData, API_BASE } from '../hooks/useData';

const Classification = () => {
    const { backendData, loading: globalLoading } = useData();
    const [selectedModel, setSelectedModel] = useState('nb');
    const [price, setPrice] = useState(500);
    const [discount, setDiscount] = useState(10);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/classify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    price,
                    discount,
                    model: selectedModel
                })
            });
            const data = await res.json();
            setPrediction(data.prediction);
        } catch (err) {
            console.error("Prediction error:", err);
        } finally {
            setLoading(false);
        }
    };

    if (globalLoading && !backendData) return <div className="card">Loading Dataset...</div>;
    if (!backendData) return <div className="card">Connecting to Python Backend...</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="card">
                <h2>Predictive Analysis</h2>
                <p style={{ color: 'var(--text-muted)' }}>Predict the likely <strong>Payment Method</strong> for a product based on its Price and Discount.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <form className="card" onSubmit={handlePredict}>
                    <h3 style={{ marginBottom: '1.5rem' }}>Input Parameters</h3>

                    <div style={{ marginBottom: '1rem' }}>
                        <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Model Algorithm:</label>
                        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ width: '100%' }}>
                            <option value="dt">Decision Tree (Depth 4)</option>
                            <option value="nb">Naive Bayes</option>
                            <option value="knn">K-Nearest Neighbors (K=5)</option>
                        </select>
                    </div>

                    <div style={{ marginBottom: '1rem' }}>
                        <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Price (Rs.): {price}</label>
                        <input type="range" min="10" max="5000" step="10" value={price} onChange={e => setPrice(parseInt(e.target.value))} style={{ width: '100%' }} />
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Discount (%): {discount}</label>
                        <input type="range" min="0" max="100" step="1" value={discount} onChange={e => setDiscount(parseInt(e.target.value))} style={{ width: '100%' }} />
                    </div>

                    <button type="submit" disabled={loading} style={{
                        width: '100%', padding: '0.75rem', background: 'var(--primary)', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold'
                    }}>
                        {loading ? 'Processing...' : 'Run Prediction'}
                    </button>
                </form>

                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                    <h3 style={{ marginBottom: '1.5rem' }}>Resulting Prediction</h3>
                    {prediction ? (
                        <div>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Predicted Payment Method</div>
                            <div style={{ fontSize: '2.5rem', color: 'var(--primary)', fontWeight: 'bold' }}>{prediction}</div>
                            <div style={{ marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                Based on {selectedModel.toUpperCase()} model logic trained on current dataset.
                            </div>
                        </div>
                    ) : (
                        <div style={{ color: 'var(--text-muted)' }}>
                            Adjust parameters and click "Run Prediction" to see results.
                        </div>
                    )}
                </div>
            </div>

            <div className="card">
                <h3>Classification Logic Details</h3>
                <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                    The backend implements supervised learning where categorical features are encoded using `LabelEncoder`.
                    The training process involves an 80/20 split with standard scaling applied for the KNN algorithm.
                </p>
                <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(79, 70, 229, 0.1)', borderRadius: '8px' }}>
                    <strong>Features Used:</strong> Price (Rs.), Discount (%) <br />
                    <strong>Target Variable:</strong> Payment Method (Encoded)
                </div>
            </div>
        </div>
    );
};

export default Classification;
