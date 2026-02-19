import React, { useRef } from 'react';
import { useData } from '../hooks/useData';
import { Upload, Download, Filter } from 'lucide-react';

const Navbar = ({ isCollapsed }) => {
    const { handleFileUpload, filters, setFilters, categories, paymentMethods, filteredData } = useData();
    const fileInputRef = useRef(null);

    const handleExport = () => {
        if (filteredData.length === 0) return;
        const headers = Object.keys(filteredData[0]);
        const csvContent = [
            headers.join(','),
            ...filteredData.map(row => headers.map(h => row[h]).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `exported_data_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
    };

    return (
        <div className="navbar" style={{
            height: 'var(--navbar-height)',
            background: 'rgba(15, 23, 42, 0.8)',
            backdropFilter: 'blur(12px)',
            borderBottom: '1px solid var(--border)',
            position: 'fixed',
            top: 0,
            right: 0,
            left: isCollapsed ? 'var(--sidebar-width-collapsed)' : 'var(--sidebar-width)',
            zIndex: 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 2rem',
            transition: 'var(--transition)'
        }}>
            <div className="filter-bar">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
                    <Filter size={18} />
                    <span style={{ fontSize: '0.9rem' }}>Filters</span>
                </div>

                <select
                    value={filters.category}
                    onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
                >
                    {categories.map(c => <option key={c} value={c}>{c}</option>)}
                </select>

                <select
                    value={filters.paymentMethod}
                    onChange={(e) => setFilters(prev => ({ ...prev, paymentMethod: e.target.value }))}
                >
                    {paymentMethods.map(p => <option key={p} value={p}>{p}</option>)}
                </select>

                <input
                    type="date"
                    value={filters.dateRange.start}
                    onChange={(e) => setFilters(prev => ({ ...prev, dateRange: { ...prev.dateRange, start: e.target.value } }))}
                />
                <input
                    type="date"
                    value={filters.dateRange.end}
                    onChange={(e) => setFilters(prev => ({ ...prev, dateRange: { ...prev.dateRange, end: e.target.value } }))}
                />
            </div>

            <div style={{ display: 'none' }}>
                <input
                    type="file"
                    ref={fileInputRef}
                    hidden
                    accept=".csv"
                    onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
                />
            </div>
        </div>
    );
};

export default Navbar;
