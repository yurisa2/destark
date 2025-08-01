import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import MembershipFunctionEditor from './MembershipFunctionEditor';
import MembershipFunctionVisual from './MembershipFunctionVisual';

function VariableEditor({ fisConfig, setFisConfig }) {
  const { type, name } = useParams();
  const navigate = useNavigate();
  const [variable, setVariable] = useState(null);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    // Get the variable based on type and name
    if (type === 'input') {
      setVariable(fisConfig.inputs[name]);
    } else if (type === 'output') {
      setVariable(fisConfig.output_variable);
    }
  }, [type, name, fisConfig]);

  const handleVariableUpdate = (updatedVariable) => {
    if (type === 'input') {
      setFisConfig(prev => ({
        ...prev,
        inputs: {
          ...prev.inputs,
          [name]: updatedVariable
        }
      }));
    } else if (type === 'output') {
      setFisConfig(prev => ({
        ...prev,
        output_variable: updatedVariable
      }));
    }
  };

  const handleMembershipFunctionUpdate = (membershipName, updatedConfig) => {
    const updatedVariable = {
      ...variable,
      membership_functions: {
        ...variable.membership_functions,
        [membershipName]: updatedConfig
      }
    };
    handleVariableUpdate(updatedVariable);
  };

  const addMembershipFunction = () => {
    const newName = prompt("Enter membership function name:");
    if (newName && newName.trim()) {
      const newConfig = {
        type: "trimf",
        params: [0, 5, 10]
      };
      
      const updatedVariable = {
        ...variable,
        membership_functions: {
          ...variable.membership_functions,
          [newName]: newConfig
        }
      };
      handleVariableUpdate(updatedVariable);
    }
  };

  const removeMembershipFunction = (membershipName) => {
    if (window.confirm(`Are you sure you want to remove "${membershipName}"?`)) {
      const updatedMembershipFunctions = { ...variable.membership_functions };
      delete updatedMembershipFunctions[membershipName];
      
      const updatedVariable = {
        ...variable,
        membership_functions: updatedMembershipFunctions
      };
      handleVariableUpdate(updatedVariable);
    }
  };

  if (!variable) {
    return (
      <div className="variable-editor">
        <div className="error-message">
          <h2>Variable Not Found</h2>
          <p>The requested variable could not be found.</p>
          <button className="btn btn-primary" onClick={() => navigate('/')}>
            ← Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="variable-editor">
      {/* Header */}
      <div className="editor-header">
        <div className="header-content">
          <button 
            className="btn btn-secondary back-button"
            onClick={() => navigate('/')}
          >
            ← Back to Dashboard
          </button>
          
          <div className="variable-info">
            <h1>{variable.name || name}</h1>
            <span className={`variable-type-badge ${type}`}>
              {type === 'input' ? 'Input Variable' : 'Output Variable'}
            </span>
          </div>
          
          <div className="header-actions">
            <button 
              className="btn btn-primary"
              onClick={() => setIsEditing(!isEditing)}
            >
              {isEditing ? '✏️ Save Changes' : '✏️ Edit Variable'}
            </button>
          </div>
        </div>
      </div>

      {/* Variable Configuration */}
      <div className="editor-content">
        <div className="variable-config-section">
          <div className="section-header">
            <h2>Variable Configuration</h2>
          </div>
          
          <div className="config-grid">
            <div className="config-item">
              <label>Variable Name:</label>
              <input
                type="text"
                value={variable.name || name}
                onChange={(e) => {
                  const updatedVariable = { ...variable, name: e.target.value };
                  handleVariableUpdate(updatedVariable);
                }}
                disabled={!isEditing}
                className="form-input"
              />
            </div>
            
            <div className="config-item">
              <label>Minimum Value:</label>
              <input
                type="number"
                value={variable.min}
                onChange={(e) => {
                  const updatedVariable = { ...variable, min: parseFloat(e.target.value) };
                  handleVariableUpdate(updatedVariable);
                }}
                disabled={!isEditing}
                className="form-input"
              />
            </div>
            
            <div className="config-item">
              <label>Maximum Value:</label>
              <input
                type="number"
                value={variable.max}
                onChange={(e) => {
                  const updatedVariable = { ...variable, max: parseFloat(e.target.value) };
                  handleVariableUpdate(updatedVariable);
                }}
                disabled={!isEditing}
                className="form-input"
              />
            </div>
            
            <div className="config-item">
              <label>Step Size:</label>
              <input
                type="number"
                value={variable.step}
                onChange={(e) => {
                  const updatedVariable = { ...variable, step: parseFloat(e.target.value) };
                  handleVariableUpdate(updatedVariable);
                }}
                disabled={!isEditing}
                className="form-input"
              />
            </div>
          </div>
        </div>

        {/* Membership Functions */}
        <div className="membership-functions-section">
          <div className="section-header">
            <h2>Membership Functions</h2>
            <button 
              className="btn btn-success"
              onClick={addMembershipFunction}
              disabled={!isEditing}
            >
              ➕ Add Function
            </button>
          </div>
          
          <div className="membership-functions-grid">
            {Object.entries(variable.membership_functions).map(([membershipName, config]) => (
              <div key={membershipName} className="membership-function-card">
                <div className="function-header">
                  <h3>{membershipName}</h3>
                  <div className="function-actions">
                    <button 
                      className="btn btn-danger btn-sm"
                      onClick={() => removeMembershipFunction(membershipName)}
                      disabled={!isEditing}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
                
                <div className="function-content">
                  <MembershipFunctionVisual 
                    config={config}
                    variableName={variable.name || name}
                    membershipName={membershipName}
                  />
                  
                  <MembershipFunctionEditor
                    name={membershipName}
                    config={config}
                    variableName={variable.name || name}
                    onUpdate={(updatedConfig) => 
                      handleMembershipFunctionUpdate(membershipName, updatedConfig)
                    }
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Variable Overview */}
        <div className="variable-overview-section">
          <div className="section-header">
            <h2>Variable Overview</h2>
          </div>
          
          <div className="overview-card">
            <div className="overview-item">
              <span className="overview-label">Type:</span>
              <span className="overview-value">{type === 'input' ? 'Input Variable' : 'Output Variable'}</span>
            </div>
            <div className="overview-item">
              <span className="overview-label">Range:</span>
              <span className="overview-value">{variable.min} - {variable.max}</span>
            </div>
            <div className="overview-item">
              <span className="overview-label">Membership Functions:</span>
              <span className="overview-value">{Object.keys(variable.membership_functions).length}</span>
            </div>
            <div className="overview-item">
              <span className="overview-label">Step Size:</span>
              <span className="overview-value">{variable.step}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VariableEditor; 