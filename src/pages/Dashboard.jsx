import React, { useMemo } from 'react';
import { useData } from '../hooks/useData';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { Line, Bar, Doughnut, Scatter } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const KPI = ({ title, value, prefix = '' }) => (
    <div className="card" style={{ flex: 1, minWidth: '200px' }}>
        <h4 style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>{title}</h4>
        <h2 style={{ fontSize: '1.75rem' }}>{prefix}{value.toLocaleString()}</h2>
    </div>
);

const Dashboard = () => {
    const { kpis, chartData, loading, backendData } = useData();

    const chartsData = useMemo(() => {
        if (!chartData || !chartData.revenueByMonth) return null;

        return {
            salesTrend: {
                labels: chartData.revenueByMonth.labels,
                datasets: [{
                    label: 'Revenue (Rs.)',
                    data: chartData.revenueByMonth.data,
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            categoryData: {
                labels: chartData.categoryShare?.labels || [],
                datasets: [{
                    label: 'Revenue by Category',
                    data: chartData.categoryShare?.data || [],
                    backgroundColor: '#22D3EE',
                    borderRadius: 8
                }]
            },
            paymentData: {
                labels: chartData.paymentDistribution?.labels || [],
                datasets: [{
                    data: chartData.paymentDistribution?.data || [],
                    backgroundColor: ['#4F46E5', '#22D3EE', '#F472B6', '#FBBF24', '#10B981'],
                    borderWidth: 0
                }]
            }
        };
    }, [chartData]);

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { position: 'top', labels: { color: '#94A3B8' } },
            tooltip: { backgroundColor: '#1E293B', titleColor: '#F8FAFC', bodyColor: '#F8FAFC' }
        },
        scales: {
            x: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94A3B8' } },
            y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94A3B8' } }
        }
    };

    if (loading && !backendData) {
        return <div style={{ textAlign: 'center', marginTop: '5rem' }}>Loading Python Analytics...</div>;
    }

    if (!backendData) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
                <h2 style={{ color: 'var(--text-muted)' }}>Welcome to E-BI Dashboard</h2>
                <p style={{ color: 'var(--text-muted)' }}>Connecting to Python Backend...</p>
            </div>
        );
    }

    const topCustomers = backendData.topCustomers || [];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                <KPI title="Total Revenue" value={kpis.totalRevenue || 0} prefix="₹" />
                <KPI title="Total Transactions" value={kpis.totalTransactions || 0} />
                <KPI title="Avg. Order Value" value={kpis.avgOrderValue || 0} prefix="₹" />
                <KPI title="Unique Users" value={kpis.uniqueUsers || 0} />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Monthly Sales Trend</h3>
                    {chartsData?.salesTrend ? <Line data={chartsData.salesTrend} options={chartOptions} /> : 'No data available'}
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Revenue by Category</h3>
                    {chartsData?.categoryData ? <Bar data={chartsData.categoryData} options={chartOptions} /> : 'No data available'}
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Revenue by Payment Method</h3>
                    <div style={{ maxWidth: '300px', margin: '0 auto' }}>
                        {chartsData?.paymentData ? <Doughnut data={chartsData.paymentData} options={{ ...chartOptions, cutout: '70%' }} /> : 'No data available'}
                    </div>
                </div>
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem' }}>Top 5 High-Value Customers</h3>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                                    <th style={{ padding: '0.75rem 0.5rem', width: '60px' }}>Sl.No</th>
                                    <th style={{ padding: '0.75rem 0.5rem' }}>User ID</th>
                                    <th style={{ padding: '0.75rem 0.5rem' }}>Orders</th>
                                    <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Total Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {topCustomers.map((cust, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                        <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td style={{ padding: '0.75rem 0.5rem', color: 'var(--primary)', fontWeight: '500' }}>{cust.userId}</td>
                                        <td style={{ padding: '0.75rem 0.5rem' }}>{cust.transactions}</td>
                                        <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right', color: 'var(--accent)' }}>₹{cust.totalSpent.toLocaleString()}</td>
                                    </tr>
                                ))}
                                {topCustomers.length === 0 && (
                                    <tr>
                                        <td colSpan="3" style={{ padding: '2rem', textAlign: 'center' }}>No customer data available</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
