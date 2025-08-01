import React from 'react';

function HelpGuide() {
  return (
    <div className="help-guide">
      <div className="card">
        <h2>Fuzzy Inference System (FIS) Builder Help Guide</h2>
        
        <section className="help-section">
          <h3>What is a Fuzzy Inference System?</h3>
          <p>
            A Fuzzy Inference System (FIS) is a computational framework that uses fuzzy logic 
            to map inputs to outputs. It's particularly useful for environmental assessment 
            where precise measurements may not be available or where human expertise needs 
            to be incorporated into decision-making processes.
          </p>
        </section>

        <section className="help-section">
          <h3>Components Overview</h3>
          
          <div className="component-help">
            <h4>1. Input Variables</h4>
            <p>
              Define the input variables for your FIS. Each input has:
            </p>
            <ul>
              <li><strong>Range:</strong> Minimum and maximum values</li>
              <li><strong>Step:</strong> Granularity for calculations</li>
              <li><strong>Membership Functions:</strong> Fuzzy sets that define how values belong to linguistic terms</li>
            </ul>
          </div>

          <div className="component-help">
            <h4>2. Output Variable</h4>
            <p>
              Define the output variable that represents the result of your FIS:
            </p>
            <ul>
              <li><strong>Name:</strong> Descriptive name for the output</li>
              <li><strong>Range:</strong> Minimum and maximum output values</li>
              <li><strong>Membership Functions:</strong> Fuzzy sets for output categories</li>
            </ul>
          </div>

          <div className="component-help">
            <h4>3. Rules</h4>
            <p>
              Define the fuzzy rules that connect inputs to outputs:
            </p>
            <ul>
              <li><strong>IF-THEN statements:</strong> "IF input1 IS low AND input2 IS high THEN output IS medium"</li>
              <li><strong>Rule weight:</strong> Importance of each rule (0-1)</li>
              <li><strong>Connectors:</strong> AND, OR operators between conditions</li>
            </ul>
          </div>
        </section>

        <section className="help-section">
          <h3>How to Use the Builder</h3>
          
          <div className="step-guide">
            <h4>Step 1: Configure Inputs</h4>
            <ol>
              <li>Set the range (min/max) for each input variable</li>
              <li>Define membership functions for each linguistic term</li>
              <li>Adjust the step size for calculation precision</li>
            </ol>
          </div>

          <div className="step-guide">
            <h4>Step 2: Configure Output</h4>
            <ol>
              <li>Name your output variable</li>
              <li>Set the output range</li>
              <li>Define membership functions for output categories</li>
            </ol>
          </div>

          <div className="step-guide">
            <h4>Step 3: Generate Rules</h4>
            <ol>
              <li>Click "Generate All 27 Rules" to create a complete rule set</li>
              <li>Review and edit individual rules as needed</li>
              <li>Adjust rule weights to reflect importance</li>
            </ol>
          </div>

          <div className="step-guide">
            <h4>Step 4: Export and Execute</h4>
            <ol>
              <li>Export your configuration as JSON</li>
              <li>Use the Execution Panel to test your FIS</li>
              <li>Analyze results and refine as needed</li>
            </ol>
          </div>
        </section>

        <section className="help-section">
          <h3>Membership Function Types</h3>
          
          <div className="mf-types">
            <div className="mf-type">
              <h4>Trapezoidal (trapmf)</h4>
              <p>Defined by four points: [a, b, c, d] where a≤b≤c≤d</p>
              <p>Good for: Wide, flat membership regions</p>
            </div>
            
            <div className="mf-type">
              <h4>Triangular (trimf)</h4>
              <p>Defined by three points: [a, b, c] where a≤b≤c</p>
              <p>Good for: Sharp, focused membership regions</p>
            </div>
          </div>
        </section>

        <section className="help-section">
          <h3>Tips for Better Results</h3>
          <ul>
            <li>Ensure membership functions overlap appropriately</li>
            <li>Cover the entire input range with membership functions</li>
            <li>Use meaningful linguistic terms (low, medium, high)</li>
            <li>Test your FIS with various input combinations</li>
            <li>Validate that outputs make sense for your domain</li>
          </ul>
        </section>

        <section className="help-section">
          <h3>Environmental Assessment Example</h3>
          <p>
            For environmental assessment, you might have inputs like:
          </p>
          <ul>
            <li><strong>Social Impact:</strong> Community acceptance, stakeholder satisfaction</li>
            <li><strong>Environmental Impact:</strong> Biodiversity loss, pollution levels</li>
            <li><strong>Strategic Value:</strong> Economic benefits, long-term sustainability</li>
          </ul>
          <p>
            The output could be a <strong>Priority Score</strong> indicating the overall 
            assessment of a project or decision.
          </p>
        </section>
      </div>
    </div>
  );
}

export default HelpGuide; 