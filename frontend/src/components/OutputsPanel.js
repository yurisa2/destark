import React from 'react';
import MembershipFunctionEditor from './MembershipFunctionEditor';

function OutputsPanel({ output, onUpdate }) {
  const handleRangeUpdate = (field, value) => {
    const updatedOutput = { ...output, [field]: parseFloat(value) };
    onUpdate(updatedOutput);
  };

  const handleNameUpdate = (value) => {
    const updatedOutput = { ...output, name: value };
    onUpdate(updatedOutput);
  };

  const handleMembershipUpdate = (membershipName, updatedMembership) => {
    const updatedOutput = {
      ...output,
      membership_functions: {
        ...output.membership_functions,
        [membershipName]: updatedMembership
      }
    };
    onUpdate(updatedOutput);
  };

  return (
    <div className="panel">
      <div className="panel-header">
        🎯 Output Variable
      </div>
      
      <div className="membership-function-editor">
        <h4>Priority Assessment</h4>
        
        <div className="form-group">
          <label className="form-label">Variable Name:</label>
          <input
            type="text"
            className="form-input"
            value={output.name}
            onChange={(e) => handleNameUpdate(e.target.value)}
            placeholder="Enter output variable name"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Value Range:</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <div className="parameter-input">
              <label>Min:</label>
              <input
                type="number"
                value={output.min}
                onChange={(e) => handleRangeUpdate('min', e.target.value)}
                step="0.1"
              />
            </div>
            <div className="parameter-input">
              <label>Max:</label>
              <input
                type="number"
                value={output.max}
                onChange={(e) => handleRangeUpdate('max', e.target.value)}
                step="0.1"
              />
            </div>
            <div className="parameter-input">
              <label>Step:</label>
              <input
                type="number"
                value={output.step}
                onChange={(e) => handleRangeUpdate('step', e.target.value)}
                step="0.1"
              />
            </div>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Membership Functions:</label>
                      {Object.entries(output.membership_functions).map(([membershipName, membershipConfig]) => (
              <MembershipFunctionEditor
                key={membershipName}
                name={membershipName}
                config={membershipConfig}
                variableName="Priority"
                onUpdate={(updatedConfig) => handleMembershipUpdate(membershipName, updatedConfig)}
              />
            ))}
        </div>
      </div>
      
      <div className="tooltip">
        <span className="tooltiptext">
          The output variable represents the final priority assessment.
          It combines all input factors using fuzzy logic rules to produce a priority score.
        </span>
        <button className="btn btn-primary" style={{ marginTop: '10px' }}>
          ℹ️ Help
        </button>
      </div>
    </div>
  );
}

export default OutputsPanel; 