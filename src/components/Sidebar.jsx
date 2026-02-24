import React from 'react';
import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    BarChart3,
    RefreshCcw,
    FolderTree,
    Fingerprint,
    BrainCircuit,
    Box,
    Menu,
    Cpu
} from 'lucide-react';

const Sidebar = ({ isCollapsed, onToggle }) => {
    const menuItems = [
        { name: 'Dashboard', path: '/', icon: <LayoutDashboard size={20} /> },
        { name: 'Time Series', path: '/time-series', icon: <BarChart3 size={20} /> },
        { name: 'Transformation', path: '/transformation', icon: <RefreshCcw size={20} /> },
        { name: 'Association Rules', path: '/association', icon: <Box size={20} /> },
        { name: 'PCA & Features', path: '/pca', icon: <Fingerprint size={20} /> },
        { name: 'Classification', path: '/classification', icon: <BrainCircuit size={20} /> },
        { name: 'Advanced Analysis', path: '/advanced', icon: <Cpu size={20} /> },
    ];

    return (
        <div className="sidebar" style={{
            width: isCollapsed ? 'var(--sidebar-width-collapsed)' : 'var(--sidebar-width)',
            height: '100vh',
            background: 'var(--bg-card)',
            borderRight: '1px solid var(--border)',
            position: 'fixed',
            left: 0,
            top: 0,
            display: 'flex',
            flexDirection: 'column',
            padding: isCollapsed ? '1.5rem 0.5rem' : '1.5rem 1rem',
            transition: 'var(--transition)',
            zIndex: 1000
        }}>
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: isCollapsed ? 'center' : 'space-between',
                marginBottom: '2.5rem',
                padding: isCollapsed ? '0' : '0 0.5rem'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    overflow: 'hidden',
                    transition: 'var(--transition)',
                    width: isCollapsed ? '0' : 'auto',
                    opacity: isCollapsed ? 0 : 1
                }}>
                    <div style={{
                        minWidth: '32px',
                        height: '32px',
                        background: 'var(--primary)',
                        borderRadius: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        <FolderTree size={20} color="white" />
                    </div>
                    <h2 style={{ fontSize: '1.1rem', color: 'var(--accent)', whiteSpace: 'nowrap' }}>ECommerce Data Tool</h2>
                </div>

                <button
                    onClick={onToggle}
                    style={{
                        background: 'transparent',
                        color: 'var(--text-muted)',
                        padding: '0.5rem',
                        borderRadius: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        hover: { background: 'var(--border)' }
                    }}
                >
                    <Menu size={24} />
                </button>
            </div>

            <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {menuItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        title={isCollapsed ? item.name : ''}
                        style={({ isActive }) => ({
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: isCollapsed ? 'center' : 'flex-start',
                            gap: isCollapsed ? '0' : '0.75rem',
                            padding: '0.75rem',
                            borderRadius: '12px',
                            textDecoration: 'none',
                            color: isActive ? 'white' : 'var(--text-muted)',
                            background: isActive ? 'var(--primary)' : 'transparent',
                            transition: 'var(--transition)',
                            overflow: 'hidden'
                        })}
                    >
                        <div style={{ minWidth: '20px', display: 'flex', justifyContent: 'center' }}>
                            {item.icon}
                        </div>
                        <span style={{
                            fontSize: '0.9rem',
                            fontWeight: 500,
                            whiteSpace: 'nowrap',
                            opacity: isCollapsed ? 0 : 1,
                            width: isCollapsed ? 0 : 'auto',
                            transition: 'var(--transition)',
                            marginLeft: isCollapsed ? 0 : '0.5rem'
                        }}>
                            {item.name}
                        </span>
                    </NavLink>
                ))}
            </nav>
        </div>
    );
};

export default Sidebar;
