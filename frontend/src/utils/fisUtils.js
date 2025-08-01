// Utility functions for Fuzzy Inference System operations

/**
 * Generate all possible rules for a 3-input FIS with 3 membership functions each
 * This creates 3^3 = 27 rules covering all combinations
 */
export function generateAllRules() {
  const membershipLevels = ['low', 'medium', 'high'];
  const rules = [];
  
  membershipLevels.forEach(social => {
    membershipLevels.forEach(environmental => {
      membershipLevels.forEach(strategic => {
        // Determine output based on combination
        let output;
        let weight = 1.0;
        
        // Simple rule logic - can be customized
        if (social === 'high' && environmental === 'high' && strategic === 'high') {
          output = 'very_high';
        } else if (social === 'low' && environmental === 'low' && strategic === 'low') {
          output = 'very_low';
        } else if (social === 'high' || environmental === 'high' || strategic === 'high') {
          output = 'high';
        } else if (social === 'low' || environmental === 'low' || strategic === 'low') {
          output = 'low';
        } else {
          output = 'medium';
        }
        
        rules.push({
          conditions: [
            { variable: 'social', membership: social },
            { variable: 'environmental', membership: environmental },
            { variable: 'strategic', membership: strategic }
          ],
          output: output,
          weight: weight,
          connector: 'AND'
        });
      });
    });
  });
  
  return rules;
}

/**
 * Validate FIS configuration for completeness and correctness
 */
export function validateFISConfig(config) {
  const errors = [];
  
  // Check basic structure
  if (!config.name || config.name.trim() === '') {
    errors.push('FIS name is required');
  }
  
  if (!config.inputs || Object.keys(config.inputs).length === 0) {
    errors.push('At least one input variable is required');
  }
  
  if (!config.output_variable) {
    errors.push('Output variable is required');
  }
  
  // Validate inputs
  Object.entries(config.inputs).forEach(([inputName, input]) => {
    if (!input.min || !input.max) {
      errors.push(`Input '${inputName}' must have min and max values`);
    }
    
    if (input.min >= input.max) {
      errors.push(`Input '${inputName}' min value must be less than max value`);
    }
    
    if (!input.step || input.step <= 0) {
      errors.push(`Input '${inputName}' must have a positive step value`);
    }
    
    if (!input.membership_functions || Object.keys(input.membership_functions).length === 0) {
      errors.push(`Input '${inputName}' must have at least one membership function`);
    }
    
    // Validate membership functions
    Object.entries(input.membership_functions).forEach(([mfName, mf]) => {
      if (!mf.type || !mf.params) {
        errors.push(`Membership function '${mfName}' in input '${inputName}' must have type and params`);
      }
      
      if (mf.type === 'trapmf' && mf.params.length !== 4) {
        errors.push(`Trapezoidal membership function '${mfName}' in input '${inputName}' must have exactly 4 parameters`);
      }
      
      if (mf.type === 'trimf' && mf.params.length !== 3) {
        errors.push(`Triangular membership function '${mfName}' in input '${inputName}' must have exactly 3 parameters`);
      }
    });
  });
  
  // Validate output variable
  if (config.output_variable) {
    const output = config.output_variable;
    
    if (!output.name || output.name.trim() === '') {
      errors.push('Output variable must have a name');
    }
    
    if (!output.min || !output.max) {
      errors.push('Output variable must have min and max values');
    }
    
    if (output.min >= output.max) {
      errors.push('Output variable min value must be less than max value');
    }
    
    if (!output.step || output.step <= 0) {
      errors.push('Output variable must have a positive step value');
    }
    
    if (!output.membership_functions || Object.keys(output.membership_functions).length === 0) {
      errors.push('Output variable must have at least one membership function');
    }
    
    // Validate output membership functions
    Object.entries(output.membership_functions).forEach(([mfName, mf]) => {
      if (!mf.type || !mf.params) {
        errors.push(`Membership function '${mfName}' in output must have type and params`);
      }
      
      if (mf.type === 'trapmf' && mf.params.length !== 4) {
        errors.push(`Trapezoidal membership function '${mfName}' in output must have exactly 4 parameters`);
      }
      
      if (mf.type === 'trimf' && mf.params.length !== 3) {
        errors.push(`Triangular membership function '${mfName}' in output must have exactly 3 parameters`);
      }
    });
  }
  
  // Validate rules
  if (!config.rules || config.rules.length === 0) {
    errors.push('At least one rule is required');
  } else {
    config.rules.forEach((rule, index) => {
      if (!rule.conditions || rule.conditions.length === 0) {
        errors.push(`Rule ${index + 1} must have at least one condition`);
      }
      
      if (!rule.output) {
        errors.push(`Rule ${index + 1} must have an output`);
      }
      
      if (rule.weight === undefined || rule.weight < 0 || rule.weight > 1) {
        errors.push(`Rule ${index + 1} must have a weight between 0 and 1`);
      }
      
      // Validate conditions reference valid inputs
      rule.conditions.forEach(condition => {
        if (!config.inputs[condition.variable]) {
          errors.push(`Rule ${index + 1} references unknown input variable '${condition.variable}'`);
        } else {
          const input = config.inputs[condition.variable];
          if (!input.membership_functions[condition.membership]) {
            errors.push(`Rule ${index + 1} references unknown membership function '${condition.membership}' for input '${condition.variable}'`);
          }
        }
      });
      
      // Validate output references valid membership function
      if (config.output_variable && !config.output_variable.membership_functions[rule.output]) {
        errors.push(`Rule ${index + 1} references unknown output membership function '${rule.output}'`);
      }
    });
  }
  
  return errors;
}

/**
 * Calculate membership degree for a given value and membership function
 */
export function calculateMembershipDegree(value, membershipFunction) {
  const { type, params } = membershipFunction;
  
  if (type === 'trimf') {
    return triangularMF(value, params[0], params[1], params[2]);
  } else if (type === 'trapmf') {
    return trapezoidalMF(value, params[0], params[1], params[2], params[3]);
  }
  
  return 0;
}

/**
 * Triangular membership function - matches C++ implementation exactly
 */
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

/**
 * Trapezoidal membership function - matches C++ implementation exactly
 */
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

/**
 * Format a rule for display
 */
export function formatRule(rule) {
  const conditions = rule.conditions.map(cond => 
    `${cond.variable} IS ${cond.membership}`
  ).join(` ${rule.connector} `);
  
  return `IF ${conditions} THEN output IS ${rule.output} (weight: ${rule.weight})`;
}

/**
 * Check if a rule is valid for the current configuration
 */
export function isRuleValid(rule, config) {
  // Check if all referenced inputs exist
  const validInputs = rule.conditions.every(condition => 
    config.inputs[condition.variable]
  );
  
  // Check if all referenced membership functions exist
  const validMemberships = rule.conditions.every(condition => {
    const input = config.inputs[condition.variable];
    return input && input.membership_functions[condition.membership];
  });
  
  // Check if output membership function exists
  const validOutput = config.output_variable && 
    config.output_variable.membership_functions[rule.output];
  
  return validInputs && validMemberships && validOutput;
} 