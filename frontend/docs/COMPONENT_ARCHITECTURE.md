# Component Architecture

## Overview
The frontend application is built with React and focuses on providing an intuitive interface for configuring Fuzzy Logic System (FIS) membership functions.

## Component Hierarchy

```
App.js
├── FISBuilder.js (Main container)
│   ├── MembershipFunctionEditor.js (Parameter editing)
│   │   └── InteractiveMembershipFunction.js (Interactive visualization)
│   ├── MembershipFunctionVisual.js (Canvas visualization)
│   ├── ConfigExporter.js (Configuration export)
│   └── ExecutionPanel.js (Execution controls)
```

## Core Components

### FISBuilder.js
**Purpose**: Main container component that orchestrates the entire FIS configuration interface.

**Responsibilities**:
- Manages overall application state
- Coordinates between different components
- Handles configuration validation
- Manages membership function definitions

**Key Props**:
- `config`: Complete FIS configuration object
- `onConfigChange`: Callback for configuration updates

### MembershipFunctionEditor.js
**Purpose**: Provides parameter input interface for membership functions.

**Responsibilities**:
- Displays parameter input fields
- Handles function type selection
- Manages parameter validation
- Integrates interactive visualization

**Key Props**:
- `name`: Membership function name
- `config`: Membership function configuration
- `variableName`: Associated variable name
- `onUpdate`: Callback for parameter updates

### InteractiveMembershipFunction.js
**Purpose**: Interactive visualization with real-time parameter adjustment.

**Responsibilities**:
- Renders SVG-based membership function visualization
- Provides interactive sliders for parameter adjustment
- Enforces parameter constraints
- Shows real-time constraint information

**Key Features**:
- Constraint-based parameter adjustment
- Real-time visual feedback
- Degenerate case handling
- Smooth user interactions

### MembershipFunctionVisual.js
**Purpose**: Canvas-based visualization for membership functions.

**Responsibilities**:
- Renders membership functions on HTML5 canvas
- Handles different membership function types
- Provides static visualization
- Supports export functionality

**Key Features**:
- High-quality rendering
- Sampling-based drawing
- Degenerate case support
- Responsive design

## State Management

### Local State
Each component manages its own local state for:
- UI interactions
- Temporary parameter values
- Visual feedback states

### Props-Based Communication
- Parent components pass configuration data down
- Child components communicate changes via callbacks
- No global state management (keeps it simple)

### State Flow
```
User Interaction → Component State → Props Update → Parent State → Re-render
```

## Data Flow

### Configuration Updates
1. User adjusts parameter in `InteractiveMembershipFunction`
2. Constraint validation occurs
3. Valid changes update `localParams` state
4. `onUpdate` callback triggers parent update
5. Parent updates configuration
6. All components re-render with new data

### Visual Updates
1. Parameter changes trigger re-calculation
2. Membership function samples are generated
3. SVG path or canvas drawing is updated
4. Visual representation reflects new parameters

## Key Patterns

### Constraint Enforcement
- Parameters are constrained before setting
- No rejection feedback (prevention over rejection)
- Real-time constraint information display

### Sampling-Based Rendering
- All membership functions use sampling approach
- Consistent with C++ implementation
- Handles degenerate cases properly

### Responsive Design
- Components adapt to different screen sizes
- Touch-friendly interactions
- Accessible interface elements

## File Organization

```
src/
├── components/
│   ├── FISBuilder.js
│   ├── MembershipFunctionEditor.js
│   ├── InteractiveMembershipFunction.js
│   ├── MembershipFunctionVisual.js
│   ├── ConfigExporter.js
│   └── ExecutionPanel.js
├── utils/
│   └── fisUtils.js
└── App.js
```

## Best Practices

### Component Design
- Single responsibility principle
- Clear prop interfaces
- Consistent error handling
- Proper TypeScript types (if using TS)

### Performance
- Memoize expensive calculations
- Avoid unnecessary re-renders
- Use React.memo for pure components
- Optimize canvas/SVG rendering

### Accessibility
- Proper ARIA labels
- Keyboard navigation support
- Screen reader compatibility
- High contrast support

## Future Enhancements

### Planned Improvements
1. Add more membership function types
2. Implement undo/redo functionality
3. Add configuration templates
4. Support for custom membership functions
5. Enhanced export/import capabilities

### Architecture Considerations
- Consider state management library for complex state
- Implement proper error boundaries
- Add comprehensive logging
- Consider micro-frontend architecture for scalability

---

**Last Updated**: January 2025  
**Maintainer**: AI Assistant  
**Status**: Current Architecture ✅ 