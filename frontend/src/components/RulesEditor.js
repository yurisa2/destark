import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { generateAllRules } from '../utils/fisUtils';

function RulesEditor({ fisConfig, setFisConfig }) {
  const navigate = useNavigate();
  const [rules, setRules] = useState(fisConfig.rules || []);
  const [selectedRule, setSelectedRule] = useState(null);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    setRules(fisConfig.rules || []);
  }, [fisConfig.rules]);

  const handleRulesUpdate = (newRules) => {
    setRules(newRules);
    setFisConfig(prev => ({ ...prev, rules: newRules }));
  };

  const generateRules = () => {
    if (window.confirm('This will replace all existing rules. Are you sure?')) {
      const newRules = generateAllRules();
      handleRulesUpdate(newRules);
    }
  };

  const addRule = () => {
    const newRule = {
      id: Date.now(),
      antecedent: [],
      consequent: '',
      weight: 1.0
    };
    setSelectedRule(newRule);
    setIsEditing(true);
  };

  const editRule = (rule) => {
    setSelectedRule(rule);
    setIsEditing(true);
  };

  const deleteRule = (ruleId) => {
    if (window.confirm('Are you sure you want to delete this rule?')) {
      const updatedRules = rules.filter(rule => rule.id !== ruleId);
      handleRulesUpdate(updatedRules);
    }
  };

  const saveRule = (rule) => {
    let updatedRules;
    if (rule.id) {
      // Update existing rule
      updatedRules = rules.map(r => r.id === rule.id ? rule : r);
    } else {
      // Add new rule
      const newRule = { ...rule, id: Date.now() };
      updatedRules = [...rules, newRule];
    }
    handleRulesUpdate(updatedRules);
    setSelectedRule(null);
    setIsEditing(false);
  };

  const cancelEdit = () => {
    setSelectedRule(null);
    setIsEditing(false);
  };

  const getInputVariables = () => {
    return Object.keys(fisConfig.inputs);
  };

  const getOutputVariables = () => {
    return fisConfig.output_variable ? [fisConfig.output_variable.name] : [];
  };

  const getMembershipFunctions = (variableName) => {
    if (fisConfig.inputs[variableName]) {
      return Object.keys(fisConfig.inputs[variableName].membership_functions);
    }
    if (fisConfig.output_variable && fisConfig.output_variable.name === variableName) {
      return Object.keys(fisConfig.output_variable.membership_functions);
    }
    return [];
  };

  return (
    <div className="rules-editor">
      {/* Header */}
      <div className="editor-header">
        <div className="header-content">
          <button 
            className="btn btn-secondary back-button"
            onClick={() => navigate('/')}
          >
            ← Back to Dashboard
          </button>
          
          <div className="editor-title">
            <h1>Fuzzy Logic Rules</h1>
            <p>Configure the rules that define your fuzzy logic system</p>
          </div>
          
          <div className="header-actions">
            <button 
              className="btn btn-success"
              onClick={generateRules}
            >
              🔄 Generate Rules
            </button>
            <button 
              className="btn btn-primary"
              onClick={addRule}
            >
              ➕ Add Rule
            </button>
          </div>
        </div>
      </div>

      {/* Rules List */}
      <div className="rules-content">
        <div className="rules-list-section">
          <div className="section-header">
            <h2>Rules ({rules.length})</h2>
            <div className="rules-stats">
              <span className="stat">Total: {rules.length}</span>
              <span className="stat">Valid: {rules.filter(r => r.antecedent.length > 0 && r.consequent).length}</span>
            </div>
          </div>
          
          <div className="rules-list">
            {rules.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">📋</div>
                <h3>No Rules Defined</h3>
                <p>Start by adding rules or generate them automatically.</p>
                <div className="empty-actions">
                  <button className="btn btn-primary" onClick={addRule}>
                    ➕ Add First Rule
                  </button>
                  <button className="btn btn-secondary" onClick={generateRules}>
                    🔄 Generate Rules
                  </button>
                </div>
              </div>
            ) : (
              rules.map((rule, index) => (
                <div key={rule.id} className="rule-card">
                  <div className="rule-header">
                    <span className="rule-number">Rule {index + 1}</span>
                    <div className="rule-actions">
                      <button 
                        className="btn btn-sm btn-primary"
                        onClick={() => editRule(rule)}
                      >
                        ✏️ Edit
                      </button>
                      <button 
                        className="btn btn-sm btn-danger"
                        onClick={() => deleteRule(rule.id)}
                      >
                        🗑️ Delete
                      </button>
                    </div>
                  </div>
                  
                  <div className="rule-content">
                    <div className="rule-text">
                      <strong>IF</strong> {rule.antecedent.map(cond => 
                        `${cond.variable} is ${cond.value}`
                      ).join(' AND ')}
                      <strong> THEN </strong>
                      {rule.consequent}
                    </div>
                    
                    {rule.weight !== 1.0 && (
                      <div className="rule-weight">
                        Weight: {rule.weight}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Rule Editor Modal */}
        {isEditing && selectedRule && (
          <div className="modal-overlay">
            <div className="modal-content rule-editor-modal">
              <div className="modal-header">
                <h2>{selectedRule.id ? 'Edit Rule' : 'Add New Rule'}</h2>
                <button className="modal-close" onClick={cancelEdit}>
                  ✕
                </button>
              </div>
              
              <div className="modal-body">
                <RuleForm
                  rule={selectedRule}
                  inputVariables={getInputVariables()}
                  outputVariables={getOutputVariables()}
                  getMembershipFunctions={getMembershipFunctions}
                  onSave={saveRule}
                  onCancel={cancelEdit}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Rule Form Component
function RuleForm({ rule, inputVariables, outputVariables, getMembershipFunctions, onSave, onCancel }) {
  const [formData, setFormData] = useState({
    antecedent: [...rule.antecedent],
    consequent: rule.consequent,
    weight: rule.weight
  });

  const addCondition = () => {
    setFormData(prev => ({
      ...prev,
      antecedent: [...prev.antecedent, { variable: '', value: '' }]
    }));
  };

  const removeCondition = (index) => {
    setFormData(prev => ({
      ...prev,
      antecedent: prev.antecedent.filter((_, i) => i !== index)
    }));
  };

  const updateCondition = (index, field, value) => {
    setFormData(prev => ({
      ...prev,
      antecedent: prev.antecedent.map((cond, i) => 
        i === index ? { ...cond, [field]: value } : cond
      )
    }));
  };

  const handleSave = () => {
    const newRule = {
      ...rule,
      antecedent: formData.antecedent.filter(cond => cond.variable && cond.value),
      consequent: formData.consequent,
      weight: formData.weight
    };
    onSave(newRule);
  };

  return (
    <div className="rule-form">
      {/* Antecedent (IF conditions) */}
      <div className="form-section">
        <h3>IF Conditions</h3>
        {formData.antecedent.map((condition, index) => (
          <div key={index} className="condition-row">
            <select
              value={condition.variable}
              onChange={(e) => updateCondition(index, 'variable', e.target.value)}
              className="form-select"
            >
              <option value="">Select Variable</option>
              {inputVariables.map(varName => (
                <option key={varName} value={varName}>{varName}</option>
              ))}
            </select>
            
            <span className="condition-operator">is</span>
            
            <select
              value={condition.value}
              onChange={(e) => updateCondition(index, 'value', e.target.value)}
              className="form-select"
              disabled={!condition.variable}
            >
              <option value="">Select Value</option>
              {condition.variable && getMembershipFunctions(condition.variable).map(funcName => (
                <option key={funcName} value={funcName}>{funcName}</option>
              ))}
            </select>
            
            <button 
              className="btn btn-danger btn-sm"
              onClick={() => removeCondition(index)}
            >
              🗑️
            </button>
          </div>
        ))}
        
        <button className="btn btn-secondary" onClick={addCondition}>
          ➕ Add Condition
        </button>
      </div>

      {/* Consequent (THEN result) */}
      <div className="form-section">
        <h3>THEN Result</h3>
        <div className="consequent-row">
          <select
            value={formData.consequent.split(' is ')[0] || ''}
            onChange={(e) => {
              const outputVar = e.target.value;
              const currentValue = formData.consequent.split(' is ')[1] || '';
              setFormData(prev => ({
                ...prev,
                consequent: outputVar ? `${outputVar} is ${currentValue}` : currentValue
              }));
            }}
            className="form-select"
          >
            <option value="">Select Output Variable</option>
            {outputVariables.map(varName => (
              <option key={varName} value={varName}>{varName}</option>
            ))}
          </select>
          
          <span className="condition-operator">is</span>
          
          <select
            value={formData.consequent.split(' is ')[1] || ''}
            onChange={(e) => {
              const outputVar = formData.consequent.split(' is ')[0] || '';
              const value = e.target.value;
              setFormData(prev => ({
                ...prev,
                consequent: outputVar ? `${outputVar} is ${value}` : value
              }));
            }}
            className="form-select"
            disabled={!formData.consequent.split(' is ')[0]}
          >
            <option value="">Select Value</option>
            {formData.consequent.split(' is ')[0] && 
              getMembershipFunctions(formData.consequent.split(' is ')[0]).map(funcName => (
                <option key={funcName} value={funcName}>{funcName}</option>
              ))
            }
          </select>
        </div>
      </div>

      {/* Rule Weight */}
      <div className="form-section">
        <h3>Rule Weight</h3>
        <input
          type="number"
          min="0"
          max="1"
          step="0.1"
          value={formData.weight}
          onChange={(e) => setFormData(prev => ({ ...prev, weight: parseFloat(e.target.value) }))}
          className="form-input"
        />
        <small>Weight between 0 and 1 (default: 1.0)</small>
      </div>

      {/* Form Actions */}
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={onCancel}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={handleSave}>
          Save Rule
        </button>
      </div>
    </div>
  );
}

export default RulesEditor; 