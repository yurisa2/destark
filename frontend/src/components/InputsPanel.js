import React from 'react';
import MembershipFunctionEditor from './MembershipFunctionEditor';

function InputsPanel({ inputs, onUpdate }) {
  const inputNames = {
    social: "Social Factors",
    environmental: "Environmental Factors", 
    strategic: "Strategic Factors"
  };

  const handleInputUpdate = (inputName, updatedInput) => {
    const newInputs = { ...inputs, [inputName]: updatedInput };
    onUpdate(newInputs);
  };

  const handleRangeUpdate = (inputName, field, value) => {
    const updatedInput = { ...inputs[inputName], [field]: parseFloat(value) };
    handleInputUpdate(inputName, updatedInput);
  };

  const handleMembershipUpdate = (inputName, membershipName, updatedMembership) => {
    const updatedInput = {
      ...inputs[inputName],
      membership_functions: {
        ...inputs[inputName].membership_functions,
        [membershipName]: updatedMembership
      }
    };
    handleInputUpdate(inputName, updatedInput);
  };

  return (
    <div className="panel">
      <div className="panel-header">
        📊 Input Variables
      </div>
      
      {Object.entries(inputs).map(([inputName, inputConfig]) => (
        <div key={inputName} className="membership-function-editor">
          <h4>{inputNames[inputName]}</h4>
          
          <div className="form-group">
            <label className="form-label">Value Range:</label>
            <div style={{ display: 'flex', gap: '10px' }}>
              <div className="parameter-input">
                <label>Min:</label>
                <input
                  type="number"
                  value={inputConfig.min}
                  onChange={(e) => handleRangeUpdate(inputName, 'min', e.target.value)}
                  step="0.1"
                />
              </div>
              <div className="parameter-input">
                <label>Max:</label>
                <input
                  type="number"
                  value={inputConfig.max}
                  onChange={(e) => handleRangeUpdate(inputName, 'max', e.target.value)}
                  step="0.1"
                />
              </div>
              <div className="parameter-input">
                <label>Step:</label>
                <input
                  type="number"
                  value={inputConfig.step}
                  onChange={(e) => handleRangeUpdate(inputName, 'step', e.target.value)}
                  step="0.1"
                />
              </div>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Membership Functions:</label>
            {Object.entries(inputConfig.membership_functions).map(([membershipName, membershipConfig]) => (
              <MembershipFunctionEditor
                key={membershipName}
                name={membershipName}
                config={membershipConfig}
                variableName={inputNames[inputName]}
                onUpdate={(updatedConfig) => handleMembershipUpdate(inputName, membershipName, updatedConfig)}
              />
            ))}
          </div>
        </div>
      ))}
      
      <div className="tooltip">
        <span className="tooltiptext">
          Input variables represent the factors that influence environmental assessment.
          Each variable has membership functions that define how values belong to linguistic terms.
        </span>
        <button className="btn btn-primary" style={{ marginTop: '10px' }}>
          ℹ️ Help
        </button>
      </div>
    </div>
  );
}

export default InputsPanel; 