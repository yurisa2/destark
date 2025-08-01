import React, { useState, useEffect } from 'react';

function RuleEditor({ rule, inputs, output, onSave, onCancel }) {
  const [editedRule, setEditedRule] = useState({
    antecedent: [
      { variable: 'social', membership: 'low' },
      { variable: 'environmental', membership: 'low' },
      { variable: 'strategic', membership: 'low' }
    ],
    consequent: 'very_low'
  });

  useEffect(() => {
    if (rule) {
      setEditedRule({ ...rule });
    }
  }, [rule]);

  const handleAntecedentChange = (index, field, value) => {
    const newAntecedent = [...editedRule.antecedent];
    newAntecedent[index] = { ...newAntecedent[index], [field]: value };
    setEditedRule({ ...editedRule, antecedent: newAntecedent });
  };

  const handleConsequentChange = (value) => {
    setEditedRule({ ...editedRule, consequent: value });
  };

  const getInputMemberships = (inputName) => {
    return inputs[inputName] ? Object.keys(inputs[inputName].membership_functions) : [];
  };

  const getOutputMemberships = () => {
    return Object.keys(output.membership_functions);
  };

  const validateRule = () => {
    // Check if all antecedent variables exist
    const validAntecedents = editedRule.antecedent.every(condition => 
      inputs[condition.variable] && 
      inputs[condition.variable].membership_functions[condition.membership]
    );
    
    // Check if consequent exists
    const validConsequent = output.membership_functions[editedRule.consequent];
    
    return validAntecedents && validConsequent;
  };

  const isValid = validateRule();

  return (
    <div>
      <h3>{rule ? 'Edit Rule' : 'Add New Rule'}</h3>
      
      <div className="form-group">
        <label className="form-label">Antecedent (IF conditions):</label>
        {editedRule.antecedent.map((condition, index) => (
          <div key={index} style={{ display: 'flex', gap: '10px', marginBottom: '10px', alignItems: 'center' }}>
            <span style={{ fontWeight: 'bold' }}>IF</span>
            
            <select
              value={condition.variable}
              onChange={(e) => handleAntecedentChange(index, 'variable', e.target.value)}
              className="form-input"
              style={{ width: '150px' }}
            >
              {Object.keys(inputs).map(inputName => (
                <option key={inputName} value={inputName}>
                  {inputName.charAt(0).toUpperCase() + inputName.slice(1)}
                </option>
              ))}
            </select>
            
            <span>IS</span>
            
            <select
              value={condition.membership}
              onChange={(e) => handleAntecedentChange(index, 'membership', e.target.value)}
              className="form-input"
              style={{ width: '120px' }}
            >
              {getInputMemberships(condition.variable).map(membership => (
                <option key={membership} value={membership}>
                  {membership}
                </option>
              ))}
            </select>
            
            {index < editedRule.antecedent.length - 1 && (
              <span style={{ fontWeight: 'bold' }}>AND</span>
            )}
          </div>
        ))}
      </div>

      <div className="form-group">
        <label className="form-label">Consequent (THEN result):</label>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <span style={{ fontWeight: 'bold' }}>THEN {output.name} IS</span>
          
          <select
            value={editedRule.consequent}
            onChange={(e) => handleConsequentChange(e.target.value)}
            className="form-input"
            style={{ width: '150px' }}
          >
            {getOutputMemberships().map(membership => (
              <option key={membership} value={membership}>
                {membership}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h4>Rule Preview:</h4>
        <div style={{ 
          padding: '10px', 
          backgroundColor: '#f8f9fa', 
          border: '1px solid #ddd', 
          borderRadius: '4px',
          fontFamily: 'monospace',
          fontSize: '14px'
        }}>
          IF {editedRule.antecedent.map(condition => 
            `${condition.variable} IS ${condition.membership}`
          ).join(' AND ')} THEN {output.name} IS {editedRule.consequent}
        </div>
      </div>

      {!isValid && (
        <div className="alert alert-warning">
          ⚠️ This rule has invalid references. Please check your selections.
        </div>
      )}

      <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
        <button 
          className="btn btn-success" 
          onClick={() => onSave(editedRule)}
          disabled={!isValid}
        >
          💾 Save Rule
        </button>
        <button className="btn btn-danger" onClick={onCancel}>
          ❌ Cancel
        </button>
      </div>
    </div>
  );
}

export default RuleEditor; 