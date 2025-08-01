# Coding Standards

## Overview
This document outlines the coding standards and best practices for the frontend application, ensuring consistency, maintainability, and quality across the codebase.

## JavaScript/React Standards

### Code Style

#### Naming Conventions
```javascript
// Components: PascalCase
function MembershipFunctionEditor() { }

// Functions: camelCase
function calculateMembershipDegree() { }

// Constants: UPPER_SNAKE_CASE
const MAX_PARAMETER_VALUE = 10;

// Variables: camelCase
const membershipValue = 0.5;

// Files: camelCase for utilities, PascalCase for components
// fisUtils.js, MembershipFunctionEditor.js
```

#### Component Structure
```javascript
// 1. Imports
import React, { useState, useEffect } from 'react';

// 2. Helper functions (if any)
function helperFunction() { }

// 3. Main component
function ComponentName({ prop1, prop2, onUpdate }) {
  // 4. State declarations
  const [state1, setState1] = useState(initialValue);
  
  // 5. Effects
  useEffect(() => {
    // effect logic
  }, [dependencies]);
  
  // 6. Event handlers
  const handleEvent = () => {
    // handler logic
  };
  
  // 7. Render logic
  return (
    <div>
      {/* JSX */}
    </div>
  );
}

// 8. Export
export default ComponentName;
```

### Membership Function Standards

#### Mathematical Functions
```javascript
// Always include NaN handling
function membershipFunction(x, a, b, c) {
  if (isNaN(x)) return NaN;
  
  // Use descriptive variable names
  const minimum = a;
  const maximum = c;
  
  // Check conditions in logical order
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
```

#### Constraint Logic
```javascript
// Constrain values before setting
const handleParamChange = (index, value) => {
  const newValue = parseFloat(value);
  let constrainedValue = newValue;
  
  // Apply constraints
  if (index === 0) {
    const maxA = newParams[1] || 10;
    constrainedValue = Math.max(0, Math.min(maxA, newValue));
  }
  
  // Update state with constrained value
  setLocalParams(prev => {
    const updated = [...prev];
    updated[index] = constrainedValue;
    return updated;
  });
};
```

### State Management

#### Local State
```javascript
// Use descriptive state names
const [localParams, setLocalParams] = useState([...config.params]);
const [sliderValues, setSliderValues] = useState([...config.params]);

// Update state immutably
setLocalParams(prev => [...prev, newValue]);
```

#### Props and Callbacks
```javascript
// Destructure props at the top
function Component({ config, onUpdate, variableName }) {
  // Use props directly
  const { type, params } = config;
}

// Pass callbacks with descriptive names
<ChildComponent 
  onParameterChange={handleParamChange}
  onConfigUpdate={handleConfigUpdate}
/>
```

### Error Handling

#### Input Validation
```javascript
// Validate inputs early
function membershipFunction(x, a, b, c) {
  // Check for invalid inputs
  if (isNaN(x) || isNaN(a) || isNaN(b) || isNaN(c)) {
    return NaN;
  }
  
  // Check for logical constraints
  if (a > b || b > c) {
    console.warn('Invalid parameter order: a <= b <= c');
    return NaN;
  }
}
```

#### Component Error Boundaries
```javascript
// Wrap components that might fail
<ErrorBoundary>
  <MembershipFunctionEditor 
    config={config}
    onUpdate={handleUpdate}
  />
</ErrorBoundary>
```

## CSS/Styling Standards

### Class Naming
```css
/* Use BEM methodology */
.membership-function-editor { }
.membership-function-editor__slider { }
.membership-function-editor__slider--active { }
.membership-function-editor__parameter-label { }
```

### Inline Styles
```javascript
// Use objects for inline styles
const style = {
  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`
};

// Apply styles conditionally
const sliderStyle = {
  ...baseStyle,
  ...(isRejected && rejectedStyle)
};
```

## Performance Standards

### Optimization Techniques
```javascript
// Memoize expensive calculations
const membershipValue = useMemo(() => {
  return triangularMF(x, a, b, c);
}, [x, a, b, c]);

// Use React.memo for pure components
const ParameterSlider = React.memo(({ value, onChange }) => {
  return <input type="range" value={value} onChange={onChange} />;
});

// Debounce frequent updates
const debouncedUpdate = useCallback(
  debounce((value) => {
    onUpdate(value);
  }, 100),
  [onUpdate]
);
```

### Rendering Optimization
```javascript
// Avoid inline functions in render
// ❌ Bad
<button onClick={() => handleClick(index)}>

// ✅ Good
<button onClick={handleClick} data-index={index}>
```

## Testing Standards

### Unit Tests
```javascript
// Test membership functions
describe('triangularMF', () => {
  test('returns 1.0 when x equals b', () => {
    expect(triangularMF(5, 0, 5, 10)).toBe(1.0);
  });
  
  test('returns 0.0 when x is outside bounds', () => {
    expect(triangularMF(-1, 0, 5, 10)).toBe(0.0);
    expect(triangularMF(11, 0, 5, 10)).toBe(0.0);
  });
  
  test('handles degenerate cases', () => {
    expect(triangularMF(0, 0, 0, 0)).toBe(1.0);
  });
});
```

### Component Tests
```javascript
// Test component behavior
describe('InteractiveMembershipFunction', () => {
  test('constrains parameter values', () => {
    render(<InteractiveMembershipFunction config={config} />);
    
    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '15' } });
    
    expect(slider.value).toBe('10'); // Constrained to max
  });
});
```

## Documentation Standards

### Code Comments
```javascript
/**
 * Calculates membership degree for triangular function
 * @param {number} x - Input value
 * @param {number} a - Left boundary
 * @param {number} b - Peak point
 * @param {number} c - Right boundary
 * @returns {number} Membership degree (0.0 to 1.0)
 */
function triangularMF(x, a, b, c) {
  // Implementation
}
```

### Component Documentation
```javascript
/**
 * Interactive membership function component with real-time parameter adjustment
 * 
 * @param {Object} props
 * @param {Object} props.config - Membership function configuration
 * @param {string} props.variableName - Name of the variable
 * @param {string} props.membershipName - Name of the membership function
 * @param {Function} props.onUpdate - Callback for parameter updates
 */
function InteractiveMembershipFunction({ config, variableName, membershipName, onUpdate }) {
  // Implementation
}
```

## Git Standards

### Commit Messages
```
feat: add trapezoidal membership function support
fix: constrain parameter A to prevent invalid values
docs: update membership function rules
test: add degenerate case tests
refactor: improve constraint logic
```

### Branch Naming
```
feature/membership-function-types
fix/parameter-constraints
docs/component-architecture
test/coverage-improvement
```

## Code Review Checklist

### Before Submitting
- [ ] Code follows naming conventions
- [ ] Functions are properly documented
- [ ] Error handling is implemented
- [ ] Performance considerations addressed
- [ ] Tests are written and passing
- [ ] No console.log statements in production code
- [ ] No unused imports or variables
- [ ] Code is properly formatted

### Review Criteria
- [ ] Functionality works as expected
- [ ] Code is readable and maintainable
- [ ] Performance is acceptable
- [ ] Security considerations addressed
- [ ] Accessibility requirements met
- [ ] Documentation is updated

---

**Last Updated**: January 2025  
**Maintainer**: AI Assistant  
**Status**: Active Standards ✅ 