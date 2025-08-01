import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Header from './components/Header';
import FISBuilder from './components/FISBuilder';
import HelpGuide from './components/HelpGuide';
import ExecutionPanel from './components/ExecutionPanel';
import './App.css';

function App() {
  const [fisConfig, setFisConfig] = useState({
    name: "Environmental Assessment FIS",
    inputs: {
      social: {
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
      name: "priority",
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
        
        <nav className="main-nav">
          <div className="container">
            <Link to="/" className="nav-link">FIS Builder</Link>
            <Link to="/execute" className="nav-link">Execute System</Link>
            <Link to="/help" className="nav-link">Help Guide</Link>
          </div>
        </nav>

        <main className="main-content">
          <div className="container">
            <Routes>
              <Route 
                path="/" 
                element={
                  <FISBuilder 
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

export default App; 