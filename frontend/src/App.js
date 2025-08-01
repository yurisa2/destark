import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import VariableEditor from './components/VariableEditor';
import RulesEditor from './components/RulesEditor';
import ExecutionPanel from './components/ExecutionPanel';
import HelpGuide from './components/HelpGuide';
import './App.css';

function App() {
  const [fisConfig, setFisConfig] = useState({
    name: "Environmental Assessment FIS",
    inputs: {
      social: {
        name: "Social Factors",
        min: 0,
        max: 10,
        step: 0.1,
        membership_functions: {
          low: { type: "trapmf", params: [0, 0, 2, 4] },
          medium: { type: "trapmf", params: [2, 4, 6, 7] },
          high: { type: "trapmf", params: [6, 7, 10, 10] }
        }
      },
      environmental: {
        name: "Environmental Factors",
        min: 0,
        max: 10,
        step: 0.1,
        membership_functions: {
          low: { type: "trapmf", params: [0, 0, 2, 5] },
          medium: { type: "trapmf", params: [2, 5, 6, 8] },
          high: { type: "trapmf", params: [6, 8, 10, 10] }
        }
      },
      strategic: {
        name: "Strategic Factors",
        min: 0,
        max: 10,
        step: 0.1,
        membership_functions: {
          low: { type: "trapmf", params: [0, 0, 3, 5] },
          medium: { type: "trapmf", params: [3, 5, 7, 8] },
          high: { type: "trapmf", params: [7, 8, 10, 10] }
        }
      }
    },
    output_variable: {
      name: "Priority Assessment",
      min: 0,
      max: 10,
      step: 0.1,
      membership_functions: {
        very_low: { type: "trimf", params: [0, 0, 2.5] },
        low: { type: "trimf", params: [0, 2.5, 5] },
        medium: { type: "trimf", params: [2.5, 5, 7.5] },
        high: { type: "trimf", params: [5, 5.5, 10] },
        very_high: { type: "trimf", params: [7.5, 10, 10] }
      }
    },
    rules: []
  });

  return (
    <Router>
      <div className="app">
        <Header title="Destark FIS Builder" />
        
        <Navigation />
        
        <main className="main-content">
          <div className="container">
            <Routes>
              <Route 
                path="/" 
                element={
                  <Dashboard 
                    fisConfig={fisConfig} 
                    setFisConfig={setFisConfig} 
                  />
                } 
              />
              <Route 
                path="/variable/:type/:name" 
                element={
                  <VariableEditor 
                    fisConfig={fisConfig} 
                    setFisConfig={setFisConfig} 
                  />
                } 
              />
              <Route 
                path="/rules" 
                element={
                  <RulesEditor 
                    fisConfig={fisConfig} 
                    setFisConfig={setFisConfig} 
                  />
                } 
              />
              <Route 
                path="/execute" 
                element={
                  <ExecutionPanel 
                    fisConfig={fisConfig} 
                  />
                } 
              />
              <Route path="/help" element={<HelpGuide />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

// Navigation component with active state
function Navigation() {
  const location = useLocation();
  
  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <nav className="main-nav">
      <div className="container">
        <Link to="/" className={`nav-link ${isActive('/') ? 'active' : ''}`}>
          🏠 Dashboard
        </Link>
        <Link to="/rules" className={`nav-link ${isActive('/rules') ? 'active' : ''}`}>
          📋 Rules
        </Link>
        <Link to="/execute" className={`nav-link ${isActive('/execute') ? 'active' : ''}`}>
          ▶️ Execute
        </Link>
        <Link to="/help" className={`nav-link ${isActive('/help') ? 'active' : ''}`}>
          ❓ Help
        </Link>
      </div>
    </nav>
  );
}

export default App; 