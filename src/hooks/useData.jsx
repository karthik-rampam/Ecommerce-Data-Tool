import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
export const API_BASE = import.meta.env.MODE === 'development'
    ? 'http://localhost:5000'
    : (import.meta.env.VITE_API_URL || '');

const DataContext = createContext();

export const DataProvider = ({ children }) => {
    const [backendData, setBackendData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Filters
    const [filters, setFilters] = useState({
        dateRange: { start: '', end: '' },
        category: 'All',
        paymentMethod: 'All'
    });

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        try {
            const params = new URLSearchParams({
                category: filters.category,
                paymentMethod: filters.paymentMethod,
                start: filters.dateRange.start,
                end: filters.dateRange.end
            });
            const response = await fetch(`${API_BASE}/api/data?${params}`);
            const result = await response.json();

            if (!response.ok) {
                setError(result.error || "Backend Error");
                setBackendData(null);
                return;
            }

            setBackendData(result);
        } catch (err) {
            setError("Cannot connect to Python Backend. Ensure it is running on port 5000.");
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (file) => {
        setLoading(true);
        setError(null);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error("Upload failed");
            await fetchData();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, [filters]);

    const categories = ['All', 'Toys', 'Electronics', 'Clothing', 'Books', 'Beauty', 'Sports', 'Home & Kitchen'];
    const paymentMethods = ['All', 'UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery'];
    const kpis = backendData?.kpis || {};
    const chartData = backendData?.charts || {};

    return (
        <DataContext.Provider value={{
            backendData,
            kpis,
            chartData,
            loading,
            error,
            filters,
            setFilters,
            handleFileUpload,
            categories,
            paymentMethods,
            refresh: fetchData
        }}>
            {children}
        </DataContext.Provider>
    );
};

export const useData = () => useContext(DataContext);
