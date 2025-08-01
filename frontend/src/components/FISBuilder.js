import React, { useState, useEffect } from 'react';
import InputsPanel from './InputsPanel';
import OutputsPanel from './OutputsPanel';
import RuleBuilder from './RuleBuilder';
import ConfigExporter from './ConfigExporter';
import { generateAllRules, validateFISConfig } from '../utils/fisUtils';

function FISBuilder({ fisConfig, setFisConfig }) {
  const [validationErrors, setValidationErrors] = useState([]);
  const [showConfig, setShowConfig] = useState(false);

  useEffect(() => {
    // Generate initial rules if none exist
    if (fisConfig.rules.length === 0) {
      const rules = generateAllRules();
      setFisConfig(prev => ({ ...prev, rules }));
    }
  }, [fisConfig.rules.length, setFisConfig]);

  useEffect(() => {
    // Validate configuration whenever it changes
    const errors = validateFISConfig(fisConfig);
    setValidationErrors(errors);
  }, [fisConfig]);

  const handleConfigUpdate = (newConfig) => {
    setFisConfig(newConfig);
  };

  return (
    <div className="fis-builder">
      <div className="card">
        <div className="form-group">
          <label className="form-label">FIS Name:</label>
          <input
            type="text"
            className="form-input"
            value={fisConfig.name}
            onChange={(e) => setFisConfig(prev => ({ ...prev, name: e.target.value }))}
            placeholder="Enter FIS name"
          />
        </div>
      </div>

      {validationErrors.length > 0 && (
        <div className="alert alert-error">
          <h4>Configuration Errors:</h4>
          <ul>
            {validationErrors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="builder-layout">
        <InputsPanel 
          inputs={fisConfig.inputs} 
          onUpdate={(inputs) => handleConfigUpdate({ ...fisConfig, inputs })}
        />
        
        <OutputsPanel 
          output={fisConfig.output_variable} 
          onUpdate={(output_variable) => handleConfigUpdate({ ...fisConfig, output_variable })}
        />
        
        <RuleBuilder 
          fisConfig={fisConfig}
          setFisConfig={setFisConfig}
        />
      </div>

      <div className="action-bar">
        <button 
          className="btn btn-success" 
          onClick={() => setShowConfig(!showConfig)}
        >
          📋 {showConfig ? 'Hide' : 'Show'} Configuration
        </button>
        
        <ConfigExporter 
          fisConfig={fisConfig} 
          isVisible={showConfig}
        />
      </div>

      {showConfig && (
        <div className="card">
          <h3>Current Configuration</h3>
          <pre>{JSON.stringify(fisConfig, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default FISBuilder; 