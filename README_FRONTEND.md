# Destark FIS Frontend System

## Overview

The Destark FIS Frontend is a comprehensive, didactic web application for building and executing Fuzzy Inference Systems (FIS) for environmental assessment. This system provides an intuitive, patent-worthy interface that makes fuzzy logic accessible to users while providing powerful functionality.

## Features

### 🏗️ FIS Builder Interface
- **Visual Configuration**: Drag-and-drop interface for building fuzzy logic systems
- **Input Variables**: Configure social, environmental, and strategic factors
- **Output Variables**: Define priority assessment with 5 membership levels
- **Membership Functions**: Visual editor for trapezoidal and triangular functions
- **Real-time Validation**: Instant feedback on configuration errors

### 📋 Rule Base Management
- **Auto-Generation**: Generate all 27 possible fuzzy rules automatically
- **Editable Rules**: Click-to-edit interface for customizing rules
- **Rule Validation**: Real-time syntax and logic checking
- **Visual Rule Builder**: Drag-and-drop antecedent/consequent selection
- **Rule Statistics**: Live count of valid/invalid rules

### 📥 Configuration Export
- **JSON Export**: Download complete FIS configuration
- **Configuration Preview**: Real-time preview of generated config
- **Template Support**: Save/load configuration templates
- **Validation**: Ensure config compatibility with backend

### 🚀 System Execution
- **File Upload**: Drag-and-drop GeoTIFF upload interface
- **Processing Status**: Real-time execution progress
- **Results Display**: Map visualization and statistics
- **Download Results**: Export processed GeoTIFF files

### 📚 Educational Content
- **Interactive Tutorials**: Step-by-step guidance
- **Fuzzy Logic Education**: Clear explanations of concepts
- **Context-Sensitive Help**: Help that appears based on user actions
- **Tooltips**: Hover explanations for all interface elements

## Architecture

### Frontend Structure
```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── FISBuilder.js     # Main FIS builder interface
│   │   ├── InputsPanel.js    # Input variables configuration
│   │   ├── OutputsPanel.js   # Output variable configuration
│   │   ├── RuleBuilder.js    # Fuzzy rules management
│   │   ├── RuleEditor.js     # Individual rule editing
│   │   ├── ConfigExporter.js # Configuration export
│   │   ├── ExecutionPanel.js # System execution interface
│   │   ├── FileUpload.js     # File upload component
│   │   ├── ExecutionStatus.js # Processing status display
│   │   └── HelpGuide.js      # Educational content
│   ├── utils/
│   │   └── fisUtils.js       # FIS utility functions
│   ├── App.js               # Main application component
│   └── index.js             # Application entry point
├── public/                  # Static assets
└── package.json            # Dependencies and scripts
```

### Component Architecture

#### FISBuilder
The main component that orchestrates the entire FIS building process:
- Manages global FIS configuration state
- Coordinates between input, output, and rule panels
- Handles validation and error display
- Provides action buttons for generation and export

#### InputsPanel
Manages the three input variables (social, environmental, strategic):
- Configures value ranges (min, max, step)
- Manages membership functions for each variable
- Provides visual membership function editor
- Real-time validation of input parameters

#### OutputsPanel
Manages the priority output variable:
- Configures output variable name and range
- Manages 5 membership levels (very_low to very_high)
- Provides membership function editing
- Ensures output compatibility with rules

#### RuleBuilder
Handles the 27 fuzzy logic rules:
- Auto-generates all possible rule combinations
- Provides editable rule interface
- Validates rule syntax and references
- Displays rule statistics and validation status

#### ExecutionPanel
Manages the FIS system execution:
- File upload interface for GeoTIFF inputs
- Processing status and progress display
- Results visualization and download
- Educational content about FIS processing

## Installation and Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Modern web browser

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd destark/frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Development
```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Build for production
npm run build
```

## Usage Guide

### 1. Building a FIS System

#### Step 1: Configure Input Variables
1. Navigate to the FIS Builder tab
2. In the Input Variables panel, configure:
   - **Social Factors**: Socioeconomic indicators (0-10 scale)
   - **Environmental Factors**: Environmental impact data (0-10 scale)
   - **Strategic Factors**: Strategic importance data (0-10 scale)
3. For each variable, set:
   - Value range (min, max, step)
   - Membership functions (low, medium, high)
   - Membership function parameters

#### Step 2: Configure Output Variable
1. In the Output Variables panel, configure:
   - Variable name (default: "priority")
   - Value range (0-10)
   - 5 membership levels: very_low, low, medium, high, very_high
   - Membership function parameters for each level

#### Step 3: Generate Rules
1. Click "Generate All 27 Rules" to create all possible combinations
2. Review generated rules in the Rules panel
3. Edit individual rules by clicking the "Edit" button
4. Add custom rules using "Add Custom Rule"

#### Step 4: Export Configuration
1. Click "Show Configuration" to preview the JSON
2. Click "Download FIS Configuration" to save the config file
3. The configuration can be used with the backend API

### 2. Executing the FIS System

#### Step 1: Upload Input Files
1. Navigate to the Execute System tab
2. Upload three GeoTIFF files:
   - Social factors raster
   - Environmental factors raster
   - Strategic factors raster
3. Ensure all files have the same spatial extent and resolution

#### Step 2: Execute Processing
1. Review the FIS configuration summary
2. Click "Execute FIS System" to start processing
3. Monitor progress through the status display
4. Wait for completion (processing time depends on file size)

#### Step 3: Download Results
1. Once processing is complete, view the results summary
2. Download the processed GeoTIFF file
3. Review statistics (mean, std, min, max values)

### 3. Educational Resources

#### Help Guide
1. Navigate to the Help Guide tab
2. Explore different topics:
   - Getting Started
   - Understanding Fuzzy Logic
   - Building Fuzzy Rules
   - FIS Processing Steps
   - Configuration Guide
   - Troubleshooting

#### Interactive Features
- Hover over interface elements for tooltips
- Click help buttons for context-sensitive assistance
- Use the educational content to understand fuzzy logic concepts

## Technical Details

### Fuzzy Logic Implementation

#### Membership Functions
- **Trapezoidal (trapmf)**: Defined by 4 parameters [a, b, c, d] where a≤b≤c≤d
- **Triangular (trimf)**: Defined by 3 parameters [a, b, c] where a≤b≤c

#### Rule Structure
Each fuzzy rule follows the format:
```
IF social IS high AND environmental IS high AND strategic IS high 
THEN priority IS very_high
```

#### Processing Pipeline
1. **Fuzzification**: Convert input values to membership degrees
2. **Rule Evaluation**: Apply fuzzy rules to determine output membership
3. **Defuzzification**: Convert fuzzy output to crisp numerical values

### API Integration

#### Backend Communication
The frontend communicates with the FastAPI backend through:
- File upload endpoints for GeoTIFF processing
- Configuration validation and processing
- Real-time status updates
- Result download functionality

#### Error Handling
- Comprehensive error validation for all inputs
- User-friendly error messages
- Graceful handling of API failures
- Retry mechanisms for failed operations

### Testing

#### Test Coverage
- Unit tests for all components
- Integration tests for API communication
- End-to-end tests for complete workflows
- Validation tests for fuzzy logic operations

#### Running Tests
```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test -- FISBuilder.test.js
```

## Deployment

### Production Build
```bash
# Create production build
npm run build

# The build output will be in the build/ directory
```

### Docker Deployment
```bash
# Build Docker image
docker build -t destark-fis-frontend .

# Run container
docker run -p 3000:3000 destark-fis-frontend
```

### Environment Variables
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8000)

## Contributing

### Development Guidelines
1. Follow React best practices
2. Use functional components with hooks
3. Implement comprehensive error handling
4. Write unit tests for all components
5. Maintain responsive design
6. Follow accessibility guidelines

### Code Style
- Use ESLint configuration
- Follow Prettier formatting
- Use meaningful component and variable names
- Add JSDoc comments for complex functions

## Troubleshooting

### Common Issues

#### File Upload Problems
- Ensure files are valid GeoTIFF format
- Check file size limits
- Verify all three input files are uploaded
- Ensure files have matching spatial extents

#### Configuration Errors
- Validate membership function parameters
- Check rule syntax and references
- Ensure all variables are properly defined
- Verify output membership functions exist

#### Processing Issues
- Check backend API connectivity
- Verify file format compatibility
- Monitor processing time for large files
- Check system memory availability

### Performance Optimization
- Use appropriate file resolutions
- Implement file compression for uploads
- Optimize component rendering
- Use lazy loading for large datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Check the Help Guide within the application
- Review the troubleshooting section
- Contact the development team
- Submit issues through the project repository 