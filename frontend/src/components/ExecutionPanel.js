import React, { useState } from 'react';
import axios from 'axios';

function ExecutionPanel({ fisConfig }) {
  const [inputValues, setInputValues] = useState({});
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [testCases, setTestCases] = useState([]);

  // Initialize input values
  React.useEffect(() => {
    const initialValues = {};
    Object.keys(fisConfig.inputs).forEach(inputName => {
      initialValues[inputName] = fisConfig.inputs[inputName].min;
    });
    setInputValues(initialValues);
  }, [fisConfig]);

  const handleInputChange = (inputName, value) => {
    setInputValues(prev => ({
      ...prev,
      [inputName]: parseFloat(value)
    }));
  };

  const executeFIS = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/execute-fis', {
        config: fisConfig,
        inputs: inputValues
      });
      
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to execute FIS. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const addTestCase = () => {
    const newTestCase = {
      id: Date.now(),
      name: `Test Case ${testCases.length + 1}`,
      inputs: { ...inputValues },
      expectedOutput: null
    };
    setTestCases([...testCases, newTestCase]);
  };

  const runTestCase = async (testCase) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/execute-fis', {
        config: fisConfig,
        inputs: testCase.inputs
      });
      
      const updatedTestCases = testCases.map(tc => 
        tc.id === testCase.id 
          ? { ...tc, expectedOutput: response.data.output }
          : tc
      );
      setTestCases(updatedTestCases);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to execute test case.');
    } finally {
      setLoading(false);
    }
  };

  const deleteTestCase = (testCaseId) => {
    setTestCases(testCases.filter(tc => tc.id !== testCaseId));
  };

  return (
    <div className="execution-panel">
      <div className="card">
        <h2>FIS Execution Panel</h2>
        <p>Test your Fuzzy Inference System with different input values.</p>
      </div>

      <div className="execution-layout">
        {/* Input Configuration */}
        <div className="card">
          <h3>Input Values</h3>
          <div className="input-grid">
            {Object.keys(fisConfig.inputs).map(inputName => {
              const input = fisConfig.inputs[inputName];
              return (
                <div key={inputName} className="form-group">
                  <label className="form-label">{inputName}:</label>
                  <input
                    type="range"
                    min={input.min}
                    max={input.max}
                    step={input.step}
                    value={inputValues[inputName] || input.min}
                    onChange={(e) => handleInputChange(inputName, e.target.value)}
                    className="form-range"
                  />
                  <div className="range-value">
                    {inputValues[inputName] || input.min}
                  </div>
                </div>
              );
            })}
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={executeFIS}
            disabled={loading}
          >
            {loading ? 'Executing...' : 'Execute FIS'}
          </button>
        </div>

        {/* Results Display */}
        <div className="card">
          <h3>Results</h3>
          {error && (
            <div className="alert alert-error">
              <strong>Error:</strong> {error}
            </div>
          )}
          
          {results && (
            <div className="results-display">
              <div className="result-item">
                <strong>Output Value:</strong> {results.output.toFixed(3)}
              </div>
              <div className="result-item">
                <strong>Membership Degrees:</strong>
                <ul>
                  {Object.entries(results.membership_degrees || {}).map(([term, degree]) => (
                    <li key={term}>{term}: {(degree * 100).toFixed(1)}%</li>
                  ))}
                </ul>
              </div>
              {results.rule_activations && (
                <div className="result-item">
                  <strong>Rule Activations:</strong>
                  <ul>
                    {results.rule_activations.map((activation, index) => (
                      <li key={index}>
                        Rule {index + 1}: {(activation * 100).toFixed(1)}%
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Test Cases */}
      <div className="card">
        <div className="test-cases-header">
          <h3>Test Cases</h3>
          <button className="btn btn-secondary" onClick={addTestCase}>
            Add Test Case
          </button>
        </div>
        
        {testCases.length > 0 && (
          <div className="test-cases-grid">
            {testCases.map(testCase => (
              <div key={testCase.id} className="test-case-card">
                <div className="test-case-header">
                  <h4>{testCase.name}</h4>
                  <button 
                    className="btn btn-danger btn-sm"
                    onClick={() => deleteTestCase(testCase.id)}
                  >
                    Delete
                  </button>
                </div>
                
                <div className="test-case-inputs">
                  {Object.entries(testCase.inputs).map(([inputName, value]) => (
                    <div key={inputName} className="test-input">
                      <strong>{inputName}:</strong> {value}
                    </div>
                  ))}
                </div>
                
                {testCase.expectedOutput !== null && (
                  <div className="test-case-output">
                    <strong>Output:</strong> {testCase.expectedOutput.toFixed(3)}
                  </div>
                )}
                
                <button 
                  className="btn btn-primary btn-sm"
                  onClick={() => runTestCase(testCase)}
                  disabled={loading}
                >
                  Run Test
                </button>
              </div>
            ))}
          </div>
        )}
        
        {testCases.length === 0 && (
          <p className="text-muted">No test cases yet. Add one to get started.</p>
        )}
      </div>

      {/* Configuration Summary */}
      <div className="card">
        <h3>Current Configuration</h3>
        <div className="config-summary">
          <div className="config-item">
            <strong>FIS Name:</strong> {fisConfig.name}
          </div>
          <div className="config-item">
            <strong>Input Variables:</strong> {Object.keys(fisConfig.inputs).length}
          </div>
          <div className="config-item">
            <strong>Rules:</strong> {fisConfig.rules.length}
          </div>
          <div className="config-item">
            <strong>Output Variable:</strong> {fisConfig.output_variable.name}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ExecutionPanel; 