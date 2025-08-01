import React, { useState } from 'react';

function InteractiveRuleBuilder({ rules, inputs, output, onRuleUpdate, onRuleDelete, onAddRule }) {
  const [selectedRule, setSelectedRule] = useState(null);
  const [showRuleCreator, setShowRuleCreator] = useState(false);

  const inputNames = {
    social: "Social Factors",
    environmental: "Environmental Factors", 
    strategic: "Strategic Factors"
  };

  const getMembershipOptions = (inputName) => {
    return inputs[inputName] ? Object.keys(inputs[inputName].membership_functions) : [];
  };

  const getOutputMembershipOptions = () => {
    return Object.keys(output.membership_functions);
  };

  const formatRule = (rule) => {
    const antecedentStr = rule.antecedent
      .map(condition => `${inputNames[condition.variable]} IS ${condition.membership}`)
      .join(' AND ');
    
    return `IF ${antecedentStr} THEN Priority IS ${rule.consequent}`;
  };

  const createRuleFromSelection = (selections) => {
    const antecedent = Object.entries(selections.antecedent).map(([variable, membership]) => ({
      variable,
      membership
    }));

    return {
      antecedent,
      consequent: selections.consequent
    };
  };

  const RuleCreator = () => {
    const [selections, setSelections] = useState({
      antecedent: {
        social: 'low',
        environmental: 'low',
        strategic: 'low'
      },
      consequent: 'very_low'
    });

    const handleAntecedentChange = (variable, membership) => {
      setSelections(prev => ({
        ...prev,
        antecedent: {
          ...prev.antecedent,
          [variable]: membership
        }
      }));
    };

    const handleConsequentChange = (membership) => {
      setSelections(prev => ({
        ...prev,
        consequent: membership
      }));
    };

    const handleSave = () => {
      const newRule = createRuleFromSelection(selections);
      onAddRule(newRule);
      setShowRuleCreator(false);
    };

    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.7)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000
      }}>
        <div style={{
          background: 'white',
          padding: '30px',
          borderRadius: '12px',
          maxWidth: '600px',
          width: '90%',
          maxHeight: '80vh',
          overflow: 'auto'
        }}>
          <h3 style={{ marginTop: 0, color: '#333' }}>🎯 Create New Fuzzy Rule</h3>
          
          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ color: '#555', marginBottom: '15px' }}>Input Conditions (IF):</h4>
            <div style={{ display: 'grid', gap: '15px' }}>
              {Object.entries(inputNames).map(([variable, displayName]) => (
                <div key={variable} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  backgroundColor: '#f9f9f9'
                }}>
                  <span style={{ fontWeight: 'bold', minWidth: '120px' }}>{displayName}:</span>
                  <select
                    value={selections.antecedent[variable]}
                    onChange={(e) => handleAntecedentChange(variable, e.target.value)}
                    style={{
                      padding: '8px 12px',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      flex: 1
                    }}
                  >
                    {getMembershipOptions(variable).map(membership => (
                      <option key={membership} value={membership}>
                        {membership.charAt(0).toUpperCase() + membership.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ color: '#555', marginBottom: '15px' }}>Output Result (THEN):</h4>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '15px',
              border: '1px solid #ddd',
              borderRadius: '6px',
              backgroundColor: '#f0f8ff'
            }}>
              <span style={{ fontWeight: 'bold' }}>Priority IS:</span>
              <select
                value={selections.consequent}
                onChange={(e) => handleConsequentChange(e.target.value)}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  flex: 1
                }}
              >
                {getOutputMembershipOptions().map(membership => (
                  <option key={membership} value={membership}>
                    {membership.replace('_', ' ').split(' ').map(word => 
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ')}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div style={{
            padding: '15px',
            backgroundColor: '#f8f9fa',
            border: '1px solid #e9ecef',
            borderRadius: '6px',
            marginBottom: '25px'
          }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#333' }}>Rule Preview:</h5>
            <div style={{
              fontFamily: 'monospace',
              fontSize: '14px',
              color: '#007bff',
              fontWeight: 'bold'
            }}>
              {formatRule(createRuleFromSelection(selections))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setShowRuleCreator(false)}
              style={{
                padding: '10px 20px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                backgroundColor: '#f8f9fa',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '4px',
                backgroundColor: '#28a745',
                color: 'white',
                cursor: 'pointer'
              }}
            >
              Create Rule
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="interactive-rule-builder">
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={() => setShowRuleCreator(true)}
          style={{
            padding: '12px 20px',
            border: 'none',
            borderRadius: '6px',
            backgroundColor: '#007bff',
            color: 'white',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold'
          }}
        >
          ➕ Create New Rule
        </button>
      </div>

      <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
        {rules.map((rule, index) => (
          <div
            key={index}
            style={{
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '15px',
              marginBottom: '10px',
              backgroundColor: selectedRule === index ? '#f0f8ff' : '#f9f9f9',
              transition: 'all 0.2s ease'
            }}
            onClick={() => setSelectedRule(index)}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <div style={{
                  fontFamily: 'monospace',
                  fontSize: '14px',
                  lineHeight: '1.4',
                  color: '#333',
                  marginBottom: '8px'
                }}>
                  {formatRule(rule)}
                </div>
                <div style={{
                  fontSize: '12px',
                  color: '#666',
                  fontStyle: 'italic'
                }}>
                  Rule #{index + 1}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRuleUpdate(index, rule);
                  }}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #007bff',
                    borderRadius: '4px',
                    backgroundColor: 'white',
                    color: '#007bff',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  ✏️ Edit
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRuleDelete(index);
                  }}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #dc3545',
                    borderRadius: '4px',
                    backgroundColor: 'white',
                    color: '#dc3545',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {showRuleCreator && <RuleCreator />}
    </div>
  );
}

export default InteractiveRuleBuilder; 