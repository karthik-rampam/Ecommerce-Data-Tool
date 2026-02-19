import React, { useState, useEffect, useMemo } from 'react';
import { useData, API_BASE } from '../hooks/useData';

const AssociationRules = () => {
    const { backendData, loading: globalLoading } = useData();
    const [rulesData, setRulesData] = useState(null);
    const [loading, setLoading] = useState(false);

    // Mining Parameters
    const [support, setSupport] = useState(0.05);
    const [confidence, setConfidence] = useState(0.3);
    const [algorithm, setAlgorithm] = useState('apriori');
    const [groupBy, setGroupBy] = useState('User_ID');

    useEffect(() => {
        if (!backendData) return;

        const fetchRules = async () => {
            setLoading(true);
            try {
                const query = new URLSearchParams({
                    support: support,
                    confidence: confidence,
                    algo: algorithm,
                    groupby: groupBy
                });
                const res = await fetch(`${API_BASE}/api/association?${query}`);
                const data = await res.json();
                setRulesData(data);
            } catch (err) {
                console.error("MBA Error:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchRules();
    }, [backendData, support, confidence, algorithm, groupBy]);

    if (globalLoading && !backendData) return <div className="card">Loading Dataset...</div>;
    if (!backendData) return <div className="card">Connecting to Python Backend...</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div className="card">
                <h2>Association Rule Mining</h2>
                <p style={{ color: 'var(--text-muted)' }}>Mined category relationships with dynamic parameters and deep visualizations.</p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginTop: '1.5rem' }}>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>🔹 Min Support: {support}</label>
                        <input type="range" min="0.01" max="0.5" step="0.01" value={support} onChange={e => setSupport(parseFloat(e.target.value))} style={{ width: '100%' }} />
                    </div>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>🔹 Min Confidence: {confidence}</label>
                        <input type="range" min="0.1" max="1.0" step="0.1" value={confidence} onChange={e => setConfidence(parseFloat(e.target.value))} style={{ width: '100%' }} />
                    </div>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>🔹 Algorithm</label>
                        <select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ width: '100%' }}>
                            <option value="apriori">Apriori</option>
                            <option value="fpgrowth">FP-Growth</option>
                        </select>
                    </div>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem' }}>🔹 Group Transaction By</label>
                        <select value={groupBy} onChange={e => setGroupBy(e.target.value)} style={{ width: '100%' }}>
                            <option value="User_ID">User ID</option>
                            <option value="Purchase_Date">Date</option>
                        </select>
                    </div>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
                <div className="card">
                    <h3>2️⃣ Frequent Itemsets</h3>
                    <div style={{ marginTop: '1rem', maxHeight: '300px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                                    <th style={{ padding: '0.5rem', width: '60px' }}>Sl.No</th>
                                    <th style={{ padding: '0.5rem' }}>Items</th>
                                    <th style={{ padding: '0.5rem' }}>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                {(rulesData?.frequent_itemsets || []).map((item, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                        <td style={{ padding: '0.5rem', color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td style={{ padding: '0.5rem', color: 'var(--primary)' }}>{item.items.join(', ')}</td>
                                        <td style={{ padding: '0.5rem' }}>{item.support.toFixed(3)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3>4️⃣ Network Graph (Top Rules)</h3>
                    <div style={{ flex: 1, minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                        {loading ? 'Redrawing graph...' : (
                            <svg width="400" height="300" style={{ border: '1px solid var(--border)', borderRadius: '8px' }}>
                                {(rulesData?.links || []).slice(0, 10).map((link, i) => {
                                    // Static circle layout for simplicity
                                    const getN = (id) => (rulesData.nodes || []).findIndex(n => n.id === id);
                                    const getP = (idx) => {
                                        const angle = (idx / (rulesData.nodes?.length || 1)) * 2 * Math.PI;
                                        return { x: 200 + 120 * Math.sin(angle), y: 150 + 100 * Math.cos(angle) };
                                    };
                                    const s = getP(getN(link.source));
                                    const t = getP(getN(link.target));
                                    return (
                                        <g key={i}>
                                            <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="rgba(79, 70, 229, 0.4)" strokeWidth={link.value} />
                                            <circle cx={s.x} cy={s.y} r="6" fill="#4F46E5" />
                                            <circle cx={t.x} cy={t.y} r="6" fill="#10B981" />
                                            <text x={s.x} y={s.y - 10} fontSize="10" fill="white" textAnchor="middle">{link.source}</text>
                                            <text x={t.x} y={t.y - 10} fontSize="10" fill="white" textAnchor="middle">{link.target}</text>
                                        </g>
                                    );
                                })}
                            </svg>
                        )}
                    </div>
                </div>
            </div>

            <div className="card">
                <h3>3️⃣ Association Rules Table</h3>
                {loading ? <div style={{ padding: '2rem', textAlign: 'center' }}>Mining...</div> : (
                    <div style={{ maxHeight: '400px', overflowY: 'auto', marginTop: '1rem' }}>
                        <table style={{ width: '100%', color: 'var(--text-muted)', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                                    <th style={{ padding: '0.5rem', width: '60px' }}>Sl.No</th>
                                    <th style={{ padding: '0.5rem' }}>Antecedents → Consequents</th>
                                    <th style={{ padding: '0.5rem' }}>Support</th>
                                    <th style={{ padding: '0.5rem' }}>Confidence</th>
                                    <th style={{ padding: '0.5rem' }}>Lift</th>
                                </tr>
                            </thead>
                            <tbody>
                                {(rulesData?.rules || []).length > 0 ? rulesData.rules.map((rule, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                        <td style={{ padding: '0.5rem', color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td style={{ padding: '0.5rem' }}>
                                            <span style={{ color: 'var(--primary)' }}>{rule.antecedents.join(', ')}</span> →
                                            <span style={{ color: 'var(--accent)' }}> {rule.consequents.join(', ')}</span>
                                        </td>
                                        <td style={{ padding: '0.5rem' }}>{rule.support.toFixed(3)}</td>
                                        <td style={{ padding: '0.5rem' }}>{rule.confidence.toFixed(3)}</td>
                                        <td style={{ padding: '0.5rem' }}>{rule.lift.toFixed(3)}</td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan="4" style={{ padding: '2rem', textAlign: 'center' }}>No rules found matching criteria. Adjust sliders to see more.</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AssociationRules;
