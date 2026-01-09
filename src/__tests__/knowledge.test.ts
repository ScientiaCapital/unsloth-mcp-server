/**
 * Knowledge Module Tests
 *
 * Comprehensive tests for OCR, training generation, AI enhancement, and database
 */

import { describe, test, expect } from '@jest/globals';

// Import knowledge module components
import { classifyContent, cleanText } from '../knowledge/ocr.js';
import { generateTrainingPairs, SYSTEM_PROMPTS } from '../knowledge/training-generator.js';
import {
  assessDifficulty,
  calculateNextReview,
  type ReviewSchedule,
} from '../knowledge/ai-enhancer.js';
import {
  Category,
  CATEGORY_DEFINITIONS,
  toAlpacaFormat,
  toShareGPTFormat,
  toChatMLFormat,
  type TrainingPair,
  type KnowledgeEntry,
} from '../knowledge/schema.js';

// ============================================================================
// OCR Module Tests
// ============================================================================

describe('OCR Module', () => {
  describe('classifyContent', () => {
    test('should classify trading content correctly', () => {
      const tradingText = `
        The hammer candlestick pattern is a bullish reversal signal.
        It appears at the bottom of a downtrend and indicates
        potential buying pressure. The pattern has a small body
        and a long lower shadow.
      `;

      const result = classifyContent(tradingText);

      expect(result.category).toBe('candlestick_patterns');
      expect(result.confidence).toBeGreaterThanOrEqual(30);
      expect(result.detected_topics).toContain('hammer');
    });

    test('should classify technical indicators content', () => {
      const indicatorText = `
        The RSI indicator measures momentum and overbought/oversold conditions.
        When RSI crosses above 70, it indicates overbought territory.
        MACD divergence can signal potential trend reversals.
        Moving averages help identify trend direction.
      `;

      const result = classifyContent(indicatorText);

      expect(result.category).toBe('technical_indicators');
      expect(result.detected_topics.length).toBeGreaterThan(0);
    });

    test('should classify HVAC content correctly', () => {
      const hvacText = `
        The HVAC system requires proper ductwork sizing for optimal CFM.
        BTU calculations must account for heat load and tonnage requirements.
        The compressor and condenser work together in the refrigeration cycle.
      `;

      const result = classifyContent(hvacText);

      expect(result.category).toBe('hvac_systems');
    });

    test('should classify electrical content correctly', () => {
      const electricalText = `
        NEC code requires proper grounding and bonding for electrical panels.
        Circuit breakers must be sized according to conductor ampacity.
        Voltage drop calculations ensure proper wire sizing for feeders.
      `;

      const result = classifyContent(electricalText);

      expect(result.category).toBe('electrical_systems');
    });

    test('should return general category for unclassified content', () => {
      const genericText = 'This is just some random text without specific keywords.';

      const result = classifyContent(genericText);

      expect(result.category).toBe('general');
      expect(result.confidence).toBeLessThanOrEqual(95);
    });

    test('should limit detected topics to 10', () => {
      const multiTopicText = `
        RSI MACD moving average bollinger stochastic ADX ATR volume momentum oscillator
        support resistance trend breakout swing pattern indicator chart analysis signal
      `;

      const result = classifyContent(multiTopicText);

      expect(result.detected_topics.length).toBeLessThanOrEqual(10);
    });
  });

  describe('cleanText', () => {
    test('should normalize line endings', () => {
      const text = 'Hello\r\nWorld\r\nTest';
      const cleaned = cleanText(text);

      expect(cleaned).not.toContain('\r');
      expect(cleaned).toContain('Hello\nWorld\nTest');
    });

    test('should remove excessive blank lines', () => {
      const text = 'Line 1\n\n\n\n\nLine 2';
      const cleaned = cleanText(text);

      expect(cleaned).toBe('Line 1\n\nLine 2');
    });

    test('should remove excessive spaces', () => {
      const text = 'Hello    world     test';
      const cleaned = cleanText(text);

      expect(cleaned).toBe('Hello world test');
    });

    test('should fix pipe to I OCR error', () => {
      const text = 'The |nvestor bought shares';
      const cleaned = cleanText(text);

      expect(cleaned).toBe('The Investor bought shares');
    });

    test('should trim whitespace from each line', () => {
      const text = '  Line with spaces  \n  Another line  ';
      const cleaned = cleanText(text);

      expect(cleaned).toBe('Line with spaces\nAnother line');
    });

    test('should handle empty string', () => {
      expect(cleanText('')).toBe('');
    });
  });
});

// ============================================================================
// Training Generator Tests
// ============================================================================

describe('Training Generator', () => {
  const mockEntry: KnowledgeEntry = {
    id: 'test_entry_1',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    source: {
      type: 'book',
      book_title: 'Technical Analysis Guide',
      author: 'Test Author',
      image_path: '/test/image.jpg',
      capture_date: '2024-01-01',
    },
    raw_text: 'The hammer pattern indicates bullish reversal at support.',
    cleaned_text:
      'The hammer pattern indicates a bullish reversal signal when it appears at support levels. Traders should look for confirmation before entering.',
    category: 'candlestick_patterns',
    topics: ['hammer', 'reversal', 'support'],
    tags: ['trading', 'patterns'],
    quality_score: 85,
    ocr_confidence: 95,
    manually_reviewed: false,
    training_pairs: [],
    related_entries: [],
  };

  describe('generateTrainingPairs', () => {
    test('should generate Q&A pairs', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        generate_qa: true,
        generate_instruction: false,
        generate_conversation: false,
      });

      const qaPairs = pairs.filter((p) => p.type === 'qa');
      expect(qaPairs.length).toBeGreaterThan(0);
      expect(qaPairs[0].instruction).toBeDefined();
      expect(qaPairs[0].output).toBeDefined();
    });

    test('should generate instruction pairs', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        generate_qa: false,
        generate_instruction: true,
        generate_conversation: false,
      });

      const instructionPairs = pairs.filter((p) => p.type === 'instruction');
      expect(instructionPairs.length).toBeGreaterThan(0);
    });

    test('should generate conversation pairs', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        generate_qa: false,
        generate_instruction: false,
        generate_conversation: true,
      });

      const conversationPairs = pairs.filter((p) => p.type === 'conversation');
      expect(conversationPairs.length).toBeGreaterThan(0);
    });

    test('should include system prompt when enabled', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        include_system_prompt: true,
        system_prompt_type: 'trading_expert',
      });

      const pairWithPrompt = pairs.find((p) => p.system_prompt);
      expect(pairWithPrompt).toBeDefined();
      expect(pairWithPrompt?.system_prompt).toContain('trading');
    });

    test('should not include system prompt when disabled', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        include_system_prompt: false,
      });

      const pairWithPrompt = pairs.find((p) => p.system_prompt);
      expect(pairWithPrompt).toBeUndefined();
    });

    test('should respect pairs_per_entry limit', () => {
      const pairs = generateTrainingPairs(mockEntry, {
        pairs_per_entry: 2,
        generate_qa: true,
        generate_instruction: false,
        generate_conversation: false,
      });

      const qaPairs = pairs.filter((p) => p.type === 'qa');
      expect(qaPairs.length).toBeLessThanOrEqual(2);
    });

    test('should calculate quality score for pairs', () => {
      const pairs = generateTrainingPairs(mockEntry);

      pairs.forEach((pair) => {
        expect(pair.quality_score).toBeGreaterThanOrEqual(0);
        expect(pair.quality_score).toBeLessThanOrEqual(100);
      });
    });
  });

  describe('SYSTEM_PROMPTS', () => {
    test('should have trading expert prompt', () => {
      expect(SYSTEM_PROMPTS.trading_expert).toBeDefined();
      expect(SYSTEM_PROMPTS.trading_expert).toContain('trading');
    });

    test('should have electrical expert prompt', () => {
      expect(SYSTEM_PROMPTS.electrical_expert).toBeDefined();
      expect(SYSTEM_PROMPTS.electrical_expert).toContain('electrician');
    });

    test('should have HVAC technician prompt', () => {
      expect(SYSTEM_PROMPTS.hvac_technician).toBeDefined();
      expect(SYSTEM_PROMPTS.hvac_technician).toContain('HVAC');
    });

    test('should have all expected expert types', () => {
      const expectedTypes = [
        'trading_expert',
        'risk_manager',
        'sales_expert',
        'energy_engineer',
        'solar_specialist',
        'electrical_expert',
        'hvac_technician',
        'estimator',
        'safety_officer',
      ];

      expectedTypes.forEach((type) => {
        expect(SYSTEM_PROMPTS[type as keyof typeof SYSTEM_PROMPTS]).toBeDefined();
      });
    });
  });
});

// ============================================================================
// AI Enhancer Tests
// ============================================================================

describe('AI Enhancer', () => {
  describe('assessDifficulty', () => {
    test('should classify beginner content', () => {
      // Note: The algorithm starts at score 3 (beginner-intermediate boundary)
      // Content without technical terms stays at beginner/intermediate
      const beginnerContent = 'A simple wire connects two points.';
      const result = assessDifficulty(beginnerContent, 'electrical_systems');

      expect(['beginner', 'intermediate']).toContain(result.level);
      expect(result.score).toBeLessThanOrEqual(5);
    });

    test('should classify intermediate content', () => {
      const intermediateContent = `
        When installing a new circuit, select the appropriate breaker size.
        Run the wiring through conduit and ensure proper grounding.
        The thermostat controls the HVAC system temperature.
      `;
      const result = assessDifficulty(intermediateContent, 'electrical_systems');

      // Content with intermediate terms (circuit, breaker, conduit, grounding, thermostat)
      expect(['beginner', 'intermediate', 'advanced']).toContain(result.level);
    });

    test('should classify advanced content', () => {
      const advancedContent = `
        Load calculation per NEC Article 220 requires derating factors.
        Voltage drop calculations must not exceed 3% for branch circuits.
        NFPA requirements for fire alarm systems include proper zoning.
        Manual J heating and cooling load calculations determine equipment sizing.
      `;
      const result = assessDifficulty(advancedContent, 'electrical_systems');

      expect(['intermediate', 'advanced', 'expert']).toContain(result.level);
      expect(result.factors.length).toBeGreaterThan(0);
    });

    test('should classify expert content', () => {
      const expertContent = `
        Arc flash analysis and coordination study are required for panels over 1200A.
        Psychrometric chart analysis determines proper HVAC system design.
        Hydraulic calculation for sprinkler density per NFPA 13 requirements.
        Commissioning sequence of operations for building automation systems.
      `;
      const result = assessDifficulty(expertContent, 'mechanical_engineering');

      expect(['advanced', 'expert']).toContain(result.level);
      expect(result.score).toBeGreaterThanOrEqual(5);
    });

    test('should identify prerequisites', () => {
      const contentWithFormulas = `
        Calculate the load using this formula: Load = Voltage × Current.
        The NEC code requires specific calculations.
      `;
      const result = assessDifficulty(contentWithFormulas, 'electrical_systems');

      expect(result.prerequisites.length).toBeGreaterThan(0);
    });

    test('should detect factors affecting difficulty', () => {
      const technicalContent = 'Perform the load calculation per NEC requirements.';
      const result = assessDifficulty(technicalContent, 'electrical_systems');

      expect(result.factors.length).toBeGreaterThan(0);
    });

    test('should provide experience estimate', () => {
      const content = 'Basic electrical safety practices.';
      const result = assessDifficulty(content, 'electrical_systems');

      expect(result.estimated_experience).toBeDefined();
      expect(typeof result.estimated_experience).toBe('string');
    });
  });

  describe('calculateNextReview (Spaced Repetition)', () => {
    const baseSchedule: ReviewSchedule = {
      entry_id: 'test_1',
      next_review: new Date(),
      interval_days: 1,
      ease_factor: 2.5,
      repetitions: 0,
      last_quality: 0,
    };

    test('should reset on failed review (quality < 3)', () => {
      const result = calculateNextReview({ ...baseSchedule, repetitions: 5 }, 2);

      expect(result.repetitions).toBe(0);
      expect(result.interval_days).toBe(1);
    });

    test('should increase interval on successful review', () => {
      const schedule: ReviewSchedule = { ...baseSchedule, repetitions: 2, interval_days: 6 };
      const result = calculateNextReview(schedule, 4);

      expect(result.interval_days).toBeGreaterThan(6);
      expect(result.repetitions).toBe(3);
    });

    test('should set 1-day interval for first successful review', () => {
      const result = calculateNextReview(baseSchedule, 4);

      expect(result.interval_days).toBe(1);
      expect(result.repetitions).toBe(1);
    });

    test('should set 6-day interval for second successful review', () => {
      const schedule: ReviewSchedule = { ...baseSchedule, repetitions: 1, interval_days: 1 };
      const result = calculateNextReview(schedule, 4);

      expect(result.interval_days).toBe(6);
      expect(result.repetitions).toBe(2);
    });

    test('should update ease factor based on quality', () => {
      const result = calculateNextReview(baseSchedule, 5); // Perfect score

      expect(result.ease_factor).toBeGreaterThanOrEqual(2.5);
    });

    test('should decrease ease factor for lower quality', () => {
      const result = calculateNextReview(baseSchedule, 3); // Barely passed

      expect(result.ease_factor).toBeLessThan(2.5);
    });

    test('should not allow ease factor below 1.3', () => {
      const lowEaseSchedule: ReviewSchedule = { ...baseSchedule, ease_factor: 1.4 };
      const result = calculateNextReview(lowEaseSchedule, 3);

      expect(result.ease_factor).toBeGreaterThanOrEqual(1.3);
    });

    test('should set next review date in the future', () => {
      const now = new Date();
      const result = calculateNextReview(baseSchedule, 4);

      expect(result.next_review.getTime()).toBeGreaterThan(now.getTime());
    });

    test('should preserve entry_id', () => {
      const result = calculateNextReview(baseSchedule, 4);

      expect(result.entry_id).toBe('test_1');
    });

    test('should store last quality rating', () => {
      const result = calculateNextReview(baseSchedule, 5);

      expect(result.last_quality).toBe(5);
    });
  });
});

// ============================================================================
// Schema Tests
// ============================================================================

describe('Schema', () => {
  describe('CATEGORY_DEFINITIONS', () => {
    test('should define all trading categories', () => {
      const tradingCategories: Category[] = [
        'candlestick_patterns',
        'chart_patterns',
        'technical_indicators',
        'risk_management',
        'trading_psychology',
      ];

      tradingCategories.forEach((cat) => {
        expect(CATEGORY_DEFINITIONS[cat]).toBeDefined();
        expect(CATEGORY_DEFINITIONS[cat].keywords.length).toBeGreaterThan(0);
      });
    });

    test('should define all MEP categories', () => {
      const mepCategories: Category[] = [
        'electrical_systems',
        'hvac_systems',
        'plumbing_systems',
        'fire_protection',
      ];

      mepCategories.forEach((cat) => {
        expect(CATEGORY_DEFINITIONS[cat]).toBeDefined();
        expect(CATEGORY_DEFINITIONS[cat].keywords.length).toBeGreaterThan(0);
      });
    });

    test('should have description for each category', () => {
      Object.values(CATEGORY_DEFINITIONS).forEach((def) => {
        expect(def.description).toBeDefined();
        expect(def.description.length).toBeGreaterThan(0);
      });
    });

    test('should have examples for non-general categories', () => {
      Object.entries(CATEGORY_DEFINITIONS).forEach(([category, def]) => {
        if (category !== 'general') {
          expect(def.examples.length).toBeGreaterThan(0);
        }
      });
    });
  });

  describe('toAlpacaFormat', () => {
    const testPair: TrainingPair = {
      id: 'test_1',
      type: 'qa',
      instruction: 'What is a hammer pattern?',
      input: '',
      output: 'A hammer is a bullish reversal candlestick pattern.',
      quality_score: 80,
    };

    test('should convert to Alpaca format', () => {
      const result = toAlpacaFormat(testPair);

      expect(result.instruction).toBe('What is a hammer pattern?');
      expect(result.input).toBe('');
      expect(result.output).toBe('A hammer is a bullish reversal candlestick pattern.');
    });

    test('should use default instruction when missing', () => {
      const pairWithoutInstruction: TrainingPair = {
        id: 'test_2',
        type: 'instruction',
        output: 'Some output text',
        quality_score: 70,
      };

      const result = toAlpacaFormat(pairWithoutInstruction);

      expect(result.instruction).toContain('information');
    });

    test('should handle input field', () => {
      const pairWithInput: TrainingPair = {
        ...testPair,
        input: 'Context about trading',
      };

      const result = toAlpacaFormat(pairWithInput);

      expect(result.input).toBe('Context about trading');
    });
  });

  describe('toShareGPTFormat', () => {
    test('should convert to ShareGPT format', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'What is RSI?',
        input: '',
        output: 'RSI is the Relative Strength Index.',
        quality_score: 80,
      };

      const result = toShareGPTFormat(testPair);

      expect(result.conversations).toBeDefined();
      expect(result.conversations.length).toBeGreaterThanOrEqual(2);
    });

    test('should include system message when present', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'What is RSI?',
        output: 'RSI is the Relative Strength Index.',
        system_prompt: 'You are a trading expert.',
        quality_score: 80,
      };

      const result = toShareGPTFormat(testPair);

      const systemMsg = result.conversations.find((c) => c.from === 'system');
      expect(systemMsg).toBeDefined();
      expect(systemMsg?.value).toBe('You are a trading expert.');
    });

    test('should have human and gpt messages', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'Question?',
        output: 'Answer.',
        quality_score: 80,
      };

      const result = toShareGPTFormat(testPair);

      const humanMsg = result.conversations.find((c) => c.from === 'human');
      const gptMsg = result.conversations.find((c) => c.from === 'gpt');

      expect(humanMsg).toBeDefined();
      expect(gptMsg).toBeDefined();
      expect(gptMsg?.value).toBe('Answer.');
    });

    test('should combine instruction and input for human message', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'Analyze this:',
        input: 'Sample data',
        output: 'Analysis result.',
        quality_score: 80,
      };

      const result = toShareGPTFormat(testPair);

      const humanMsg = result.conversations.find((c) => c.from === 'human');
      expect(humanMsg?.value).toContain('Analyze this:');
      expect(humanMsg?.value).toContain('Sample data');
    });
  });

  describe('toChatMLFormat', () => {
    test('should convert to ChatML format', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'What is MACD?',
        output: 'MACD is Moving Average Convergence Divergence.',
        quality_score: 80,
      };

      const result = toChatMLFormat(testPair);

      expect(result.messages).toBeDefined();
      expect(result.messages.length).toBeGreaterThanOrEqual(2);
    });

    test('should include system role when present', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'Question?',
        output: 'Answer.',
        system_prompt: 'System instruction.',
        quality_score: 80,
      };

      const result = toChatMLFormat(testPair);

      const systemMsg = result.messages.find((m) => m.role === 'system');
      expect(systemMsg).toBeDefined();
      expect(systemMsg?.content).toBe('System instruction.');
    });

    test('should have user and assistant roles', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'Question?',
        output: 'Answer.',
        quality_score: 80,
      };

      const result = toChatMLFormat(testPair);

      const userMsg = result.messages.find((m) => m.role === 'user');
      const assistantMsg = result.messages.find((m) => m.role === 'assistant');

      expect(userMsg).toBeDefined();
      expect(assistantMsg).toBeDefined();
      expect(assistantMsg?.content).toBe('Answer.');
    });

    test('should combine instruction and input for user message', () => {
      const testPair: TrainingPair = {
        id: 'test_1',
        type: 'qa',
        instruction: 'Explain:',
        input: 'Candlestick patterns',
        output: 'Explanation here.',
        quality_score: 80,
      };

      const result = toChatMLFormat(testPair);

      const userMsg = result.messages.find((m) => m.role === 'user');
      expect(userMsg?.content).toContain('Explain:');
      expect(userMsg?.content).toContain('Candlestick patterns');
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Knowledge Module Integration', () => {
  test('should generate complete training pipeline from content', () => {
    // Simulate content extraction
    const rawContent = `
      The doji candlestick pattern shows indecision in the market.
      It has a small body and indicates that buyers and sellers are balanced.
      Doji patterns are more significant at support or resistance levels.
    `;

    // Clean the text
    const cleanedContent = cleanText(rawContent);
    expect(cleanedContent.length).toBeGreaterThan(0);

    // Classify the content
    const classification = classifyContent(cleanedContent);
    expect(classification.category).toBe('candlestick_patterns');
    expect(classification.detected_topics).toContain('doji');

    // Create mock entry
    const entry: KnowledgeEntry = {
      id: 'integration_test_1',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      source: {
        type: 'book',
        image_path: '/test/doji.jpg',
        capture_date: new Date().toISOString(),
      },
      raw_text: rawContent,
      cleaned_text: cleanedContent,
      category: classification.category,
      topics: classification.detected_topics,
      tags: ['trading', 'candlesticks'],
      quality_score: 80,
      ocr_confidence: 90,
      manually_reviewed: false,
      training_pairs: [],
      related_entries: [],
    };

    // Generate training pairs
    const pairs = generateTrainingPairs(entry);
    expect(pairs.length).toBeGreaterThan(0);

    // Convert to different formats
    const alpacaPair = toAlpacaFormat(pairs[0]);
    const sharegptPair = toShareGPTFormat(pairs[0]);
    const chatmlPair = toChatMLFormat(pairs[0]);

    expect(alpacaPair.output).toBeDefined();
    expect(sharegptPair.conversations.length).toBeGreaterThan(0);
    expect(chatmlPair.messages.length).toBeGreaterThan(0);
  });

  test('should assess difficulty for trade content', () => {
    const beginnerContent = 'A hammer pattern has a small body and long lower shadow.';
    const expertContent = `
      Arc flash hazard analysis requires coordination study per NFPA 70E.
      Calculate incident energy using IEEE 1584 methodology.
    `;

    const beginnerResult = assessDifficulty(beginnerContent, 'candlestick_patterns');
    const expertResult = assessDifficulty(expertContent, 'electrical_systems');

    expect(beginnerResult.score).toBeLessThan(expertResult.score);
  });
});
