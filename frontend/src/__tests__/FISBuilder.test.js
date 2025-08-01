import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import FISBuilder from '../components/FISBuilder';

// Mock the utility functions
jest.mock('../utils/fisUtils', () => ({
  generateAllRules: jest.fn(() => [
    {
      antecedent: [
        { variable: 'social', membership: 'low' },
        { variable: 'environmental', membership: 'low' },
        { variable: 'strategic', membership: 'low' }
      ],
      consequent: 'very_low'
    }
  ]),
  validateFISConfig: jest.fn(() => [])
}));

const mockFisConfig = {
  name: "Test FIS",
  inputs: {
    social: {
      min: 0,
      max: 10,
      step: 0.1,
      membership_functions: {
        low: { type: "trapmf", params: [0, 0, 2, 4] },
        medium: { type: "trapmf", params: [2, 4, 6, 7] },
        high: { type: "trapmf", params: [6, 7, 10, 10] }
      }
    },
    environmental: {
      min: 0,
      max: 10,
      step: 0.1,
      membership_functions: {
        low: { type: "trapmf", params: [0, 0, 2, 5] },
        medium: { type: "trapmf", params: [2, 5, 6, 8] },
        high: { type: "trapmf", params: [6, 8, 10, 10] }
      }
    },
    strategic: {
      min: 0,
      max: 10,
      step: 0.1,
      membership_functions: {
        low: { type: "trapmf", params: [0, 0, 3, 5] },
        medium: { type: "trapmf", params: [3, 5, 7, 8] },
        high: { type: "trapmf", params: [7, 8, 10, 10] }
      }
    }
  },
  output_variable: {
    name: "priority",
    min: 0,
    max: 10,
    step: 0.1,
    membership_functions: {
      very_low: { type: "trimf", params: [0, 0, 2.5] },
      low: { type: "trimf", params: [0, 2.5, 5] },
      medium: { type: "trimf", params: [2.5, 5, 7.5] },
      high: { type: "trimf", params: [5, 5.5, 10] },
      very_high: { type: "trimf", params: [7.5, 10, 10] }
    }
  },
  rules: []
};

const mockSetFisConfig = jest.fn();

describe('FISBuilder Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders FIS name input field', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const nameInput = screen.getByLabelText(/FIS Name/i);
    expect(nameInput).toBeInTheDocument();
    expect(nameInput).toHaveValue('Test FIS');
  });

  test('allows editing FIS name', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const nameInput = screen.getByLabelText(/FIS Name/i);
    fireEvent.change(nameInput, { target: { value: 'New FIS Name' } });
    
    expect(mockSetFisConfig).toHaveBeenCalledWith(expect.objectContaining({
      name: 'New FIS Name'
    }));
  });

  test('renders input variables panel', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    expect(screen.getByText('📊 Input Variables')).toBeInTheDocument();
    expect(screen.getByText('Social Factors')).toBeInTheDocument();
    expect(screen.getByText('Environmental Factors')).toBeInTheDocument();
    expect(screen.getByText('Strategic Factors')).toBeInTheDocument();
  });

  test('renders output variables panel', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    expect(screen.getByText('🎯 Output Variable')).toBeInTheDocument();
    expect(screen.getByText('Priority Assessment')).toBeInTheDocument();
  });

  test('renders rules panel', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    expect(screen.getByText(/📋 Fuzzy Rules/)).toBeInTheDocument();
  });

  test('shows generate rules button', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const generateButton = screen.getByText(/🔄 Generate All 27 Rules/i);
    expect(generateButton).toBeInTheDocument();
  });

  test('shows configuration toggle button', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const configButton = screen.getByText(/📋 Show Configuration/i);
    expect(configButton).toBeInTheDocument();
  });

  test('toggles configuration display', () => {
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const configButton = screen.getByText(/📋 Show Configuration/i);
    fireEvent.click(configButton);
    
    expect(screen.getByText(/📋 Hide Configuration/i)).toBeInTheDocument();
    expect(screen.getByText('Current Configuration')).toBeInTheDocument();
  });

  test('displays validation errors when present', () => {
    const { validateFISConfig } = require('../utils/fisUtils');
    validateFISConfig.mockReturnValue(['Error 1', 'Error 2']);
    
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    expect(screen.getByText('Configuration Errors:')).toBeInTheDocument();
    expect(screen.getByText('Error 1')).toBeInTheDocument();
    expect(screen.getByText('Error 2')).toBeInTheDocument();
  });

  test('generates rules when button is clicked', () => {
    const { generateAllRules } = require('../utils/fisUtils');
    
    render(<FISBuilder fisConfig={mockFisConfig} setFisConfig={mockSetFisConfig} />);
    
    const generateButton = screen.getByText(/🔄 Generate All 27 Rules/i);
    fireEvent.click(generateButton);
    
    expect(generateAllRules).toHaveBeenCalled();
    expect(mockSetFisConfig).toHaveBeenCalledWith(expect.objectContaining({
      rules: expect.arrayContaining([
        expect.objectContaining({
          antecedent: expect.arrayContaining([
            expect.objectContaining({ variable: 'social', membership: 'low' })
          ]),
          consequent: 'very_low'
        })
      ])
    }));
  });

  test('initializes with generated rules if none exist', () => {
    const configWithoutRules = { ...mockFisConfig, rules: [] };
    const { generateAllRules } = require('../utils/fisUtils');
    
    render(<FISBuilder fisConfig={configWithoutRules} setFisConfig={mockSetFisConfig} />);
    
    expect(generateAllRules).toHaveBeenCalled();
  });

  test('does not generate rules if they already exist', () => {
    const configWithRules = { ...mockFisConfig, rules: [{ antecedent: [], consequent: 'test' }] };
    const { generateAllRules } = require('../utils/fisUtils');
    
    render(<FISBuilder fisConfig={configWithRules} setFisConfig={mockSetFisConfig} />);
    
    expect(generateAllRules).not.toHaveBeenCalled();
  });
}); 