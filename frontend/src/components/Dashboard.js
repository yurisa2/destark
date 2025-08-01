import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { validateFISConfig } from '../utils/fisUtils';

function Dashboard({ fisConfig, setFisConfig }) {
  const [validationErrors, setValidationErrors] = useState([]);
  const [stats, setStats] = useState({
    totalInputs: 0,
    totalOutputs: 1,
    totalRules: 0,
    totalMembershipFunctions: 0
  });

  useEffect(() => {
    // Validate configuration
    const errors = validateFISConfig(fisConfig);
    setValidationErrors(errors);

    // Calculate statistics
    const inputCount = Object.keys(fisConfig.inputs).length;
    const outputCount = fisConfig.output_variable ? 1 : 0;
    const ruleCount = fisConfig.rules.length;
    
    let membershipCount = 0;
    Object.values(fisConfig.inputs).forEach(input => {
      membershipCount += Object.keys(input.membership_functions).length;
    });
    if (fisConfig.output_variable) {
      membershipCount += Object.keys(fisConfig.output_variable.membership_functions).length;
    }

    setStats({
      totalInputs: inputCount,
      totalOutputs: outputCount,
      totalRules: ruleCount,
      totalMembershipFunctions: membershipCount
    });
  }, [fisConfig]);

  const handleConfigUpdate = (newConfig) => {
    setFisConfig(newConfig);
  };

  return (
    <div className="dashboard">
      {/* Header Section */}
      <div className="dashboard-header">
        <div className="dashboard-title">
          <h1>{fisConfig.name}</h1>
          <p className="dashboard-subtitle">Fuzzy Logic System Configuration</p>
        </div>
        
        <div className="dashboard-stats">
          <div className="stat-card">
            <div className="stat-number">{stats.totalInputs}</div>
            <div className="stat-label">Input Variables</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{stats.totalOutputs}</div>
            <div className="stat-label">Output Variables</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{stats.totalRules}</div>
            <div className="stat-label">Rules</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{stats.totalMembershipFunctions}</div>
            <div className="stat-label">Membership Functions</div>
          </div>
        </div>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="alert alert-error">
          <h4>⚠️ Configuration Issues</h4>
          <ul>
            {validationErrors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* System Configuration */}
      <div className="dashboard-section">
        <div className="section-header">
          <h2>System Configuration</h2>
          <button 
            className="btn btn-primary"
            onClick={() => {
              const newName = prompt("Enter new system name:", fisConfig.name);
              if (newName) {
                handleConfigUpdate({ ...fisConfig, name: newName });
              }
            }}
          >
            ✏️ Edit Name
          </button>
        </div>
        
        <div className="config-card">
          <div className="config-item">
            <span className="config-label">System Name:</span>
            <span className="config-value">{fisConfig.name}</span>
          </div>
          <div className="config-item">
            <span className="config-label">Status:</span>
            <span className={`config-value ${validationErrors.length === 0 ? 'status-valid' : 'status-invalid'}`}>
              {validationErrors.length === 0 ? '✅ Valid' : '❌ Invalid'}
            </span>
          </div>
        </div>
      </div>

      {/* Input Variables */}
      <div className="dashboard-section">
        <div className="section-header">
          <h2>Input Variables</h2>
          <span className="section-count">{stats.totalInputs} variables</span>
        </div>
        
        <div className="variables-grid">
          {Object.entries(fisConfig.inputs).map(([key, input]) => (
            <div key={key} className="variable-card input-card">
              <div className="variable-header">
                <h3>{input.name || key}</h3>
                <span className="variable-type">Input</span>
              </div>
              
              <div className="variable-details">
                <div className="detail-item">
                  <span className="detail-label">Range:</span>
                  <span className="detail-value">{input.min} - {input.max}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Membership Functions:</span>
                  <span className="detail-value">{Object.keys(input.membership_functions).length}</span>
                </div>
              </div>
              
              <div className="variable-actions">
                <Link 
                  to={`/variable/input/${key}`} 
                  className="btn btn-primary"
                >
                  🔧 Configure
                </Link>
                <button 
                  className="btn btn-secondary"
                  onClick={() => {
                    // Preview functionality
                    console.log('Preview variable:', key);
                  }}
                >
                  👁️ Preview
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Output Variables */}
      <div className="dashboard-section">
        <div className="section-header">
          <h2>Output Variables</h2>
          <span className="section-count">{stats.totalOutputs} variable</span>
        </div>
        
        <div className="variables-grid">
          {fisConfig.output_variable && (
            <div className="variable-card output-card">
              <div className="variable-header">
                <h3>{fisConfig.output_variable.name}</h3>
                <span className="variable-type">Output</span>
              </div>
              
              <div className="variable-details">
                <div className="detail-item">
                  <span className="detail-label">Range:</span>
                  <span className="detail-value">{fisConfig.output_variable.min} - {fisConfig.output_variable.max}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Membership Functions:</span>
                  <span className="detail-value">{Object.keys(fisConfig.output_variable.membership_functions).length}</span>
                </div>
              </div>
              
              <div className="variable-actions">
                <Link 
                  to={`/variable/output/priority`} 
                  className="btn btn-primary"
                >
                  🔧 Configure
                </Link>
                <button 
                  className="btn btn-secondary"
                  onClick={() => {
                    // Preview functionality
                    console.log('Preview output variable');
                  }}
                >
                  👁️ Preview
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="dashboard-section">
        <div className="section-header">
          <h2>Quick Actions</h2>
        </div>
        
        <div className="actions-grid">
          <Link to="/rules" className="action-card">
            <div className="action-icon">📋</div>
            <div className="action-title">Manage Rules</div>
            <div className="action-description">Configure fuzzy logic rules</div>
          </Link>
          
          <Link to="/execute" className="action-card">
            <div className="action-icon">▶️</div>
            <div className="action-title">Execute System</div>
            <div className="action-description">Run the fuzzy logic system</div>
          </Link>
          
          <button className="action-card" onClick={() => {
            // Export configuration
            const dataStr = JSON.stringify(fisConfig, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${fisConfig.name.replace(/\s+/g, '_')}.json`;
            link.click();
          }}>
            <div className="action-icon">💾</div>
            <div className="action-title">Export Config</div>
            <div className="action-description">Download configuration file</div>
          </button>
          
          <Link to="/help" className="action-card">
            <div className="action-icon">❓</div>
            <div className="action-title">Help Guide</div>
            <div className="action-description">Learn how to use the system</div>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard; 