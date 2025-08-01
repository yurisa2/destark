import React, { useEffect, useRef } from 'react';

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

function MembershipFunctionVisual({ config, variableName, membershipName }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !config) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up coordinate system
    const padding = 20;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    // Draw axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Membership Degree', width / 2, height - 5);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Value', 0, 0);
    ctx.restore();

    // Draw value range (assuming 0-10)
    const minValue = 0;
    const maxValue = 10;
    const valueRange = maxValue - minValue;

    // Draw membership function
    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(0, 123, 255, 0.1)';

    if (config.type === 'trapmf') {
      const [a, b, c, d] = config.params;
      
      // Convert to canvas coordinates
      const xA = padding + (a / valueRange) * graphWidth;
      const xB = padding + (b / valueRange) * graphWidth;
      const xC = padding + (c / valueRange) * graphWidth;
      const xD = padding + (d / valueRange) * graphWidth;
      const yTop = padding;
      const yBottom = height - padding;

      // Handle degenerate cases properly
      if (a === b && b === c && c === d) {
        // All points are the same - draw a vertical line
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xA, yTop);
        ctx.lineTo(xA, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (a === b && c === d) {
        // a = b and c = d, create a rectangle
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xA, yTop);
        ctx.lineTo(xC, yTop);
        ctx.lineTo(xC, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (a === b) {
        // a = b, create a right trapezoid
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xA, yTop);
        ctx.lineTo(xC, yTop);
        ctx.lineTo(xD, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (c === d) {
        // c = d, create a left trapezoid
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xB, yTop);
        ctx.lineTo(xC, yTop);
        ctx.lineTo(xC, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else {
        // Normal trapezoid: sample the membership function at many points
        ctx.beginPath();
        
        // Sample the membership function at many points (like C++ does with 1000 divisions)
        const divisions = 100;
        const dx = valueRange / divisions;
        const points = [];
        
        for (let i = 0; i <= divisions; i++) {
          const x = minValue + i * dx;
          const membership = trapezoidalMF(x, a, b, c, d);
          
          const canvasX = padding + (x / valueRange) * graphWidth;
          const canvasY = yBottom - (membership * (yBottom - yTop));
          
          points.push({ x: canvasX, y: canvasY });
        }
        
        // Create polygon path
        ctx.moveTo(points[0].x, yBottom); // Start at bottom
        
        // Draw the membership function line
        for (let i = 0; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        
        // Close the polygon to the bottom
        ctx.lineTo(points[points.length - 1].x, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Draw parameter labels
      ctx.fillStyle = '#333';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('a', xA, yBottom + 15);
      ctx.fillText('b', xB, yBottom + 15);
      ctx.fillText('c', xC, yBottom + 15);
      ctx.fillText('d', xD, yBottom + 15);

    } else if (config.type === 'trimf') {
      const [a, b, c] = config.params;
      
      // Convert to canvas coordinates
      const xA = padding + (a / valueRange) * graphWidth;
      const xB = padding + (b / valueRange) * graphWidth;
      const xC = padding + (c / valueRange) * graphWidth;
      
      // Map membership degrees to y-coordinates
      const yTop = padding;
      const yBottom = height - padding;

      // Handle degenerate cases properly
      if (a === b && b === c) {
        // All points are the same - draw a vertical line
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xA, yTop);
        ctx.lineTo(xA, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (a === b) {
        // a = b, create a right triangle with vertical line from a to b
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xA, yTop);
        ctx.lineTo(xC, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (b === c) {
        // b = c, create a right triangle with vertical line from b to c
        ctx.beginPath();
        ctx.moveTo(xA, yBottom);
        ctx.lineTo(xB, yTop);
        ctx.lineTo(xB, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else {
        // Normal triangle: sample the membership function at many points
        ctx.beginPath();
        
        // Sample the membership function at many points (like C++ does with 1000 divisions)
        const divisions = 100;
        const dx = valueRange / divisions;
        const points = [];
        
        for (let i = 0; i <= divisions; i++) {
          const x = minValue + i * dx;
          const membership = triangularMF(x, a, b, c);
          
          const canvasX = padding + (x / valueRange) * graphWidth;
          const canvasY = yBottom - (membership * (yBottom - yTop));
          
          points.push({ x: canvasX, y: canvasY });
        }
        
        // Create polygon path
        ctx.moveTo(points[0].x, yBottom); // Start at bottom
        
        // Draw the membership function line
        for (let i = 0; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        
        // Close the polygon to the bottom
        ctx.lineTo(points[points.length - 1].x, yBottom);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Draw parameter labels
      ctx.fillStyle = '#333';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('a', xA, yBottom + 15);
      ctx.fillText('b', xB, yBottom + 15);
      ctx.fillText('c', xC, yBottom + 15);
    }

    // Draw title
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${variableName} - ${membershipName}`, width / 2, 15);

  }, [config, variableName, membershipName]);

  return (
    <div className="membership-function-visual">
      <canvas
        ref={canvasRef}
        width={300}
        height={120}
        style={{ width: '100%', height: '120px' }}
      />
    </div>
  );
}

export default MembershipFunctionVisual; 