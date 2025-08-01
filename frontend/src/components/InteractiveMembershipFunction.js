import React, { useState, useEffect } from 'react';

// Triangular membership function - matches C++ implementation exactly
function triangularMF(x, a, b, c) {
  // Handle NaN input
  if (isNaN(x)) return NaN;
  
  const minimum = a;
  const maximum = c;
  
  if (x === b) {
    return 1.0;
  } else if (x <= minimum || x >= maximum) {
    return 0.0;
  } else if (x < b) {
    return (x - minimum) / (b - minimum);
  } else {
    return (maximum - x) / (maximum - b);
  }
}

// Trapezoidal membership function - matches C++ implementation exactly
function trapezoidalMF(x, a, b, c, d) {
  // Handle NaN input
  if (isNaN(x)) return NaN;
  
  const minimum = a;
  const maximum = d;
  
  if (x <= minimum || x >= maximum) {
    return 0.0;
  } else if (x >= b && x <= c) {
    return 1.0;
  } else if (x < b) {
    return (x - minimum) / (b - minimum);
  } else {
    return (maximum - x) / (maximum - c);
  }
}

function InteractiveMembershipFunction({ config, variableName, membershipName, onUpdate }) {
  const [localParams, setLocalParams] = useState([...config.params]);
  const [sliderValues, setSliderValues] = useState([...config.params]);

  useEffect(() => {
    setLocalParams([...config.params]);
    setSliderValues([...config.params]);
  }, [config.params]);

  const handleParamChange = (index, value) => {
    const newValue = parseFloat(value);
    const newParams = [...localParams];
    let constrainedValue = newValue;
    
    // Constrain the value to valid ranges before setting it
    if (config.type === 'trimf') {
      if (index === 0) { // parameter a
        // A must be <= B
        const maxA = newParams[1] || 10;
        constrainedValue = Math.max(0, Math.min(maxA, newValue));
      } else if (index === 1) { // parameter b
        // B must be >= A and <= C
        const minB = newParams[0] || 0;
        const maxB = newParams[2] || 10;
        constrainedValue = Math.max(minB, Math.min(maxB, newValue));
      } else if (index === 2) { // parameter c
        // C must be >= B
        const minC = newParams[1] || 0;
        constrainedValue = Math.max(minC, Math.min(10, newValue));
      }
    } else if (config.type === 'trapmf') {
      if (index === 0) { // parameter a
        // A must be <= B
        const maxA = newParams[1] || 10;
        constrainedValue = Math.max(0, Math.min(maxA, newValue));
      } else if (index === 1) { // parameter b
        const minB = newParams[0] || 0;
        const maxB = newParams[2] || 10;
        constrainedValue = Math.max(minB, Math.min(maxB, newValue));
      } else if (index === 2) { // parameter c
        const minC = newParams[1] || 0;
        const maxC = newParams[3] || 10;
        constrainedValue = Math.max(minC, Math.min(maxC, newValue));
      } else if (index === 3) { // parameter d
        const minD = newParams[2] || 0;
        constrainedValue = Math.max(minD, Math.min(10, newValue));
      }
    }
    
    // Update slider value to the constrained value
    const newSliderValues = [...sliderValues];
    newSliderValues[index] = constrainedValue;
    setSliderValues(newSliderValues);
    
    // Update the actual parameter
    newParams[index] = constrainedValue;
    setLocalParams(newParams);
    onUpdate({ ...config, params: newParams });
  };

  const getParamLabels = () => {
    if (config.type === 'trapmf') {
      return ['a', 'b', 'c', 'd'];
    } else if (config.type === 'trimf') {
      return ['a', 'b', 'c'];
    }
    return [];
  };

  const paramLabels = getParamLabels();

  // Generate SVG path for membership function
  const generateSVGPath = () => {
    const width = 300;
    const height = 150;
    const padding = 30;
    const graphWidth = width - 2 * padding;

    if (config.type === 'trapmf') {
      const [a, b, c, d] = localParams;
      const yTop = padding;
      const yBottom = height - padding;
      
      // Handle degenerate cases properly
      if (a === b && b === c && c === d) {
        // All points are the same - draw a vertical line
        const x = padding + (a / 10) * graphWidth;
        return `M ${x} ${yBottom} L ${x} ${yTop} L ${x} ${yBottom} Z`;
      } else if (a === b && c === d) {
        // a = b and c = d, create a rectangle
        const xA = padding + (a / 10) * graphWidth;
        const xC = padding + (c / 10) * graphWidth;
        return `M ${xA} ${yBottom} L ${xA} ${yTop} L ${xC} ${yTop} L ${xC} ${yBottom} Z`;
      } else if (a === b) {
        // a = b, create a right trapezoid
        const xA = padding + (a / 10) * graphWidth;
        const xC = padding + (c / 10) * graphWidth;
        const xD = padding + (d / 10) * graphWidth;
        return `M ${xA} ${yBottom} L ${xA} ${yTop} L ${xC} ${yTop} L ${xD} ${yBottom} Z`;
      } else if (c === d) {
        // c = d, create a left trapezoid
        const xA = padding + (a / 10) * graphWidth;
        const xB = padding + (b / 10) * graphWidth;
        const xC = padding + (c / 10) * graphWidth;
        return `M ${xA} ${yBottom} L ${xB} ${yTop} L ${xC} ${yTop} L ${xC} ${yBottom} Z`;
      } else {
        // Normal trapezoid: sample the membership function at many points
        const divisions = 50;
        const dx = 10 / divisions;
        const points = [];
        
        for (let i = 0; i <= divisions; i++) {
          const x = i * dx;
          const membership = trapezoidalMF(x, a, b, c, d);
          
          const svgX = padding + (x / 10) * graphWidth;
          const svgY = yBottom - (membership * (yBottom - yTop));
          
          points.push({ x: svgX, y: svgY });
        }
        
        // Create SVG path
        let path = `M ${points[0].x} ${yBottom}`;
        
        // Draw the membership function line
        for (let i = 0; i < points.length; i++) {
          path += ` L ${points[i].x} ${points[i].y}`;
        }
        
        // Close the polygon to the bottom
        path += ` L ${points[points.length - 1].x} ${yBottom} Z`;
        
        return path;
      }
    } else if (config.type === 'trimf') {
      const [a, b, c] = localParams;
      const yTop = padding;
      const yBottom = height - padding;
      
      // Handle degenerate cases properly
      if (a === b && b === c) {
        // All points are the same - draw a vertical line
        const x = padding + (a / 10) * graphWidth;
        return `M ${x} ${yBottom} L ${x} ${yTop} L ${x} ${yBottom} Z`;
      } else if (a === b) {
        // a = b, create a right triangle with vertical line from a to b
        const xA = padding + (a / 10) * graphWidth;
        const xC = padding + (c / 10) * graphWidth;
        return `M ${xA} ${yBottom} L ${xA} ${yTop} L ${xC} ${yBottom} Z`;
      } else if (b === c) {
        // b = c, create a right triangle with vertical line from b to c
        const xA = padding + (a / 10) * graphWidth;
        const xB = padding + (b / 10) * graphWidth;
        return `M ${xA} ${yBottom} L ${xB} ${yTop} L ${xB} ${yBottom} Z`;
      } else {
        // Normal triangle: sample the membership function at many points
        const divisions = 50;
        const dx = 10 / divisions;
        const points = [];
        
        for (let i = 0; i <= divisions; i++) {
          const x = i * dx;
          const membership = triangularMF(x, a, b, c);
          
          const svgX = padding + (x / 10) * graphWidth;
          const svgY = yBottom - (membership * (yBottom - yTop));
          
          points.push({ x: svgX, y: svgY });
        }
        
        // Create SVG path
        let path = `M ${points[0].x} ${yBottom}`;
        
        // Draw the membership function line
        for (let i = 0; i < points.length; i++) {
          path += ` L ${points[i].x} ${points[i].y}`;
        }
        
        // Close the polygon to the bottom
        path += ` L ${points[points.length - 1].x} ${yBottom} Z`;
        
        return path;
      }
    }
    return '';
  };

  // Generate parameter markers
  const generateMarkers = () => {
    const width = 300;
    const height = 150;
    const padding = 30;
    const graphWidth = width - 2 * padding;
    const yBottom = height - padding;

    // Only show markers for the parameters that exist for this function type
    const numMarkers = config.type === 'trimf' ? 3 : 4;
    
    return localParams.slice(0, numMarkers).map((param, index) => {
      const x = padding + (param / 10) * graphWidth;
      return (
        <g key={index}>
          <circle cx={x} cy={yBottom} r="4" fill="#ef4444" />
          <text x={x} y={yBottom + 20} textAnchor="middle" className="text-xs font-bold fill-gray-700">
            {paramLabels[index]}
          </text>
        </g>
      );
    });
  };

  return (
    <div className="bg-white border-2 border-gray-200 rounded-lg p-6 shadow-sm">
      {/* Visual Preview */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-3 text-center">
          {variableName} - {membershipName}
        </h4>
        <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
          <svg width="300" height="150" className="w-full h-32">
            {/* Grid lines */}
            {Array.from({ length: 11 }, (_, i) => (
              <line
                key={`v${i}`}
                x1={30 + (i / 10) * 240}
                y1="30"
                x2={30 + (i / 10) * 240}
                y2="120"
                stroke="#e5e7eb"
                strokeWidth="1"
              />
            ))}
            {Array.from({ length: 6 }, (_, i) => (
              <line
                key={`h${i}`}
                x1="30"
                y1={30 + (i / 5) * 90}
                x2="270"
                y2={30 + (i / 5) * 90}
                stroke="#e5e7eb"
                strokeWidth="1"
              />
            ))}
            
            {/* Axes */}
            <line x1="30" y1="30" x2="30" y2="120" stroke="#374151" strokeWidth="2" />
            <line x1="30" y1="120" x2="270" y2="120" stroke="#374151" strokeWidth="2" />
            
            {/* Axis labels */}
            {Array.from({ length: 6 }, (_, i) => (
              <text
                key={`label${i}`}
                x="15"
                y={125 - (i / 5) * 90}
                className="text-xs fill-gray-600"
                textAnchor="middle"
              >
                {(i / 5).toFixed(1)}
              </text>
            ))}
            {Array.from({ length: 6 }, (_, i) => (
              <text
                key={`xlabel${i}`}
                x={30 + (i * 2 / 5) * 240}
                y="135"
                className="text-xs fill-gray-600"
                textAnchor="middle"
              >
                {i * 2}
              </text>
            ))}
            
            {/* Membership function */}
            <path
              d={generateSVGPath()}
              fill="rgba(59, 130, 246, 0.1)"
              stroke="#3b82f6"
              strokeWidth="2"
            />
            
            {/* Parameter markers */}
            {generateMarkers()}
          </svg>
        </div>
      </div>

      {/* Function Type Selector */}
      <div className="flex justify-between items-center mb-6">
        <h5 className="text-md font-medium text-gray-700">Function Type:</h5>
        <select
          value={config.type}
          onChange={(e) => {
            const newType = e.target.value;
            let newParams;
            if (newType === 'trapmf') {
              newParams = [0, 2, 4, 6];
            } else if (newType === 'trimf') {
              newParams = [0, 5, 10];
            }
            onUpdate({ ...config, type: newType, params: newParams });
          }}
          className="px-3 py-2 border border-gray-300 rounded-md bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="trapmf">Trapezoidal (trapmf)</option>
          <option value="trimf">Triangular (trimf)</option>
        </select>
      </div>

      {/* Parameter Sliders */}
      <div className="space-y-4">
        {paramLabels.map((label, index) => {
          // Use fixed ranges for all sliders to prevent visual movement
          const minValue = 0;
          const maxValue = 10;
          
          return (
            <div key={index} className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-gray-700">
                  Parameter {label.toUpperCase()}
                </label>
                <span className={`text-sm font-bold transition-colors text-blue-600`}>
                  {sliderValues[index].toFixed(1)}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min={minValue}
                  max={maxValue}
                  step="0.1"
                  value={sliderValues[index]}
                  onChange={(e) => handleParamChange(index, e.target.value)}
                  className={`flex-1 h-2 rounded-lg appearance-none cursor-pointer slider transition-colors bg-gray-200`}
                  style={{
                    background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(sliderValues[index] / maxValue) * 100}%, #e5e7eb ${(sliderValues[index] / maxValue) * 100}%, #e5e7eb 100%)`
                  }}
                />
                <div className="flex space-x-1">
                  <button
                    onClick={() => handleParamChange(index, Math.max(minValue, sliderValues[index] - 0.1))}
                    className="w-8 h-8 flex items-center justify-center bg-gray-100 hover:bg-gray-200 rounded-md text-gray-600 font-bold transition-colors"
                  >
                    −
                  </button>
                  <button
                    onClick={() => handleParamChange(index, Math.min(maxValue, sliderValues[index] + 0.1))}
                    className="w-8 h-8 flex items-center justify-center bg-gray-100 hover:bg-gray-200 rounded-md text-gray-600 font-bold transition-colors"
                  >
                    +
                  </button>
                </div>
              </div>
              {/* Show constraint information */}
              <div className="text-xs text-gray-500">
                {config.type === 'trimf' && (
                  <>
                    {index === 0 && `A must be ≤ ${localParams[1].toFixed(1)}`}
                    {index === 1 && `B must be ≥ ${localParams[0].toFixed(1)} and ≤ ${localParams[2].toFixed(1)}`}
                    {index === 2 && `C must be ≥ ${localParams[1].toFixed(1)}`}
                  </>
                )}
                {config.type === 'trapmf' && (
                  <>
                    {index === 0 && `A must be ≤ ${localParams[1].toFixed(1)}`}
                    {index === 1 && `B must be ≥ ${localParams[0].toFixed(1)} and ≤ ${localParams[2].toFixed(1)}`}
                    {index === 2 && `C must be ≥ ${localParams[1].toFixed(1)} and ≤ ${localParams[3].toFixed(1)}`}
                    {index === 3 && `D must be ≥ ${localParams[2].toFixed(1)}`}
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Constraint Info */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-sm text-blue-800 text-center font-medium">
          {config.type === 'trapmf' && 'Trapezoidal: a ≤ b ≤ c ≤ d'}
          {config.type === 'trimf' && 'Triangular: a ≤ b ≤ c'}
        </p>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
      `}</style>
    </div>
  );
}

export default InteractiveMembershipFunction; 