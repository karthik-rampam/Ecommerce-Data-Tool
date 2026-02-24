import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TimeSeries from './pages/TimeSeries';
import DataTransformation from './pages/DataTransformation';
import AssociationRules from './pages/AssociationRules';
import FeatureAnalysis from './pages/FeatureAnalysis';
import Classification from './pages/Classification';
import AdvancedAnalysis from './pages/AdvancedAnalysis';
import { DataProvider, useData } from './hooks/useData';

const AppContent = () => {
  const { error } = useData();
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  const toggleSidebar = () => setIsCollapsed(!isCollapsed);

  return (
    <div className="app-container">
      <Sidebar isCollapsed={isCollapsed} onToggle={toggleSidebar} />
      <main className="main-content" style={{
        marginLeft: isCollapsed ? 'var(--sidebar-width-collapsed)' : 'var(--sidebar-width)',
        transition: 'var(--transition)'
      }}>
        <Navbar isCollapsed={isCollapsed} />
        {error && (
          <div className="card" style={{
            borderColor: '#EF4444',
            background: 'rgba(239, 68, 68, 0.1)',
            marginBottom: '1.5rem',
            color: '#EF4444'
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/time-series" element={<TimeSeries />} />
          <Route path="/transformation" element={<DataTransformation />} />
          <Route path="/association" element={<AssociationRules />} />
          <Route path="/pca" element={<FeatureAnalysis />} />
          <Route path="/classification" element={<Classification />} />
          <Route path="/advanced" element={<AdvancedAnalysis />} />
        </Routes>
      </main>
    </div>
  );
};

const App = () => {
  return (
    <DataProvider>
      <Router>
        <AppContent />
      </Router>
    </DataProvider>
  );
};

export default App;
