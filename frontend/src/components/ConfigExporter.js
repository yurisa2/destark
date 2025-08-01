import React from 'react';

function ConfigExporter({ fisConfig, isVisible }) {
  const downloadConfig = () => {
    const config = {
      description: "Environmental Assessment FIS Configuration",
      input_variables: fisConfig.inputs,
      output_variable: fisConfig.output_variable,
      rules: fisConfig.rules
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fisConfig.name.replace(/\s+/g, '_')}_config.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="config-exporter">
      <button 
        onClick={downloadConfig} 
        className="btn btn-success"
        style={{ marginLeft: '10px' }}
      >
        📥 Download FIS Configuration (JSON)
      </button>
      
      <div className="card" style={{ marginTop: '20px' }}>
        <h3>Configuration Preview</h3>
        <div style={{ 
          backgroundColor: '#f8f9fa', 
          padding: '15px', 
          borderRadius: '4px',
          maxHeight: '400px',
          overflow: 'auto',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}>
          <pre>{JSON.stringify({
            description: "Environmental Assessment FIS Configuration",
            input_variables: fisConfig.inputs,
            output_variable: fisConfig.output_variable,
            rules: fisConfig.rules
          }, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}

export default ConfigExporter; 