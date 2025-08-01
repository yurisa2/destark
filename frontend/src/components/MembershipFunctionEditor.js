import React from 'react';
import InteractiveMembershipFunction from './InteractiveMembershipFunction';

function MembershipFunctionEditor({ name, config, variableName, onUpdate }) {
  const handleTypeChange = (newType) => {
    onUpdate({ ...config, type: newType });
  };

  const handleParamChange = (index, value) => {
    const newParams = [...config.params];
    newParams[index] = parseFloat(value);
    onUpdate({ ...config, params: newParams });
  };

  const getParamLabels = () => {
    if (config.type === 'trapmf') {
      return ['a', 'b', 'c', 'd'];
    } else if (config.type === 'trimf') {
      return ['a', 'b', 'c'];
    }
    return [];
  };

  const paramLabels = getParamLabels();

  return (
    <div className="membership-function-editor" style={{ margin: '10px 0', padding: '10px', border: '1px solid #ddd', borderRadius: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <h5 style={{ margin: 0, color: '#333' }}>{name}</h5>
        <select
          value={config.type}
          onChange={(e) => handleTypeChange(e.target.value)}
          style={{ padding: '4px 8px', border: '1px solid #ddd', borderRadius: '2px' }}
        >
          <option value="trapmf">Trapezoidal (trapmf)</option>
          <option value="trimf">Triangular (trimf)</option>
        </select>
      </div>
      
      <div className="parameter-inputs">
        {paramLabels.map((label, index) => (
          <div key={index} className="parameter-input">
            <label>{label}:</label>
            <input
              type="number"
              value={config.params[index] || 0}
              onChange={(e) => handleParamChange(index, e.target.value)}
              step="0.1"
              style={{ padding: '4px 8px', border: '1px solid #ddd', borderRadius: '2px', fontSize: '12px' }}
            />
          </div>
        ))}
      </div>
      
      <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
        {config.type === 'trapmf' && 'Trapezoidal: [a, b, c, d] where a≤b≤c≤d'}
        {config.type === 'trimf' && 'Triangular: [a, b, c] where a≤b≤c'}
      </div>
      
      <InteractiveMembershipFunction 
        config={config}
        variableName={variableName || "Variable"}
        membershipName={name}
        onUpdate={onUpdate}
      />
    </div>
  );
}

export default MembershipFunctionEditor; 