import React, { useState } from 'react';
import { generateAllRules, formatRule } from '../utils/fisUtils';

const RuleBuilder = ({ fisConfig, setFisConfig }) => {
  // eslint-disable-next-line no-unused-vars
  const [showRuleCreator, setShowRuleCreator] = useState(false);

  const handleGenerateRules = () => {
    const rules = generateAllRules();
    setFisConfig(prev => ({ ...prev, rules }));
  };

  const handleRuleDelete = (ruleIndex) => {
    const newRules = fisConfig.rules.filter((_, index) => index !== ruleIndex);
    setFisConfig(prev => ({ ...prev, rules: newRules }));
  };

  const handleRuleWeightChange = (ruleIndex, newWeight) => {
    const newRules = [...fisConfig.rules];
    newRules[ruleIndex] = { ...newRules[ruleIndex], weight: newWeight };
    setFisConfig(prev => ({ ...prev, rules: newRules }));
  };

  return (
    <div className="panel" style={{ minHeight: '600px' }}>
      <div className="panel-header">
        📋 Fuzzy Rules ({fisConfig.rules.length} total)
      </div>
      
      <div className="rule-controls">
        <button className="btn btn-primary" onClick={handleGenerateRules}>
          🔄 Generate All 27 Rules
        </button>
        <button className="btn btn-secondary" onClick={() => setShowRuleCreator(true)}>
          Add Custom Rule
        </button>
      </div>

      <div className="rule-stats">
        <p># Total rules: {fisConfig.rules.length}</p>
        <p># Valid Rules: {fisConfig.rules.filter(rule => rule.weight > 0).length}</p>
        <p># Zero Weight Rules: {fisConfig.rules.filter(rule => rule.weight === 0).length}</p>
      </div>

      <div className="rules-list">
        {fisConfig.rules.length === 0 ? (
          <div className="empty-state">
            <p>No rules defined yet. Click "Generate All 27 Rules" to create a complete rule set.</p>
          </div>
        ) : (
          fisConfig.rules.map((rule, index) => (
            <div key={index} className="rule-item">
              <div className="rule-content">
                <div className="rule-text">
                  {formatRule(rule)}
                </div>
                <div className="rule-weight">
                  <label>Weight:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={rule.weight}
                    onChange={(e) => handleRuleWeightChange(index, parseFloat(e.target.value))}
                    className="weight-slider"
                  />
                  <span>{rule.weight}</span>
                </div>
              </div>
              <button 
                className="btn btn-danger btn-sm"
                onClick={() => handleRuleDelete(index)}
              >
                Delete
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default RuleBuilder; 