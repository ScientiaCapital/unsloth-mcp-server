/**
 * Knowledge Base Schema for Financial Book Training Pipeline
 *
 * This schema supports:
 * - Cataloguing content from book photos
 * - Organizing by category, topic, and source
 * - Generating training data for fine-tuning
 */

// ============================================================================
// Core Types
// ============================================================================

export interface KnowledgeEntry {
  id: string;
  created_at: string;
  updated_at: string;

  // Source Information
  source: SourceInfo;

  // Content
  raw_text: string;
  cleaned_text: string;

  // Classification
  category: Category;
  topics: string[];
  tags: string[];

  // Quality Metadata
  quality_score: number; // 0-100
  ocr_confidence: number; // 0-100
  manually_reviewed: boolean;

  // Training Data Generation
  training_pairs: TrainingPair[];

  // Relationships
  related_entries: string[]; // IDs of related entries
}

export interface SourceInfo {
  type: 'book' | 'article' | 'notes' | 'chart' | 'other';
  book_title?: string;
  author?: string;
  chapter?: string;
  page_numbers?: string;
  image_path: string;
  capture_date: string;
}

export type Category =
  // Trading & Finance
  | 'candlestick_patterns'
  | 'chart_patterns'
  | 'technical_indicators'
  | 'risk_management'
  | 'trading_psychology'
  | 'market_structure'
  | 'options_strategies'
  | 'fundamental_analysis'
  | 'order_flow'
  | 'volume_analysis'
  // Sales & Persuasion
  | 'sales_techniques'
  | 'negotiation'
  | 'persuasion'
  | 'closing'
  // Business & Entrepreneurship
  | 'business_strategy'
  | 'marketing'
  | 'leadership'
  | 'management'
  | 'startups'
  // Self-Help & Personal Development
  | 'mindset'
  | 'habits'
  | 'productivity'
  | 'motivation'
  | 'success_principles'
  // Wealth & Investing
  | 'wealth_building'
  | 'real_estate'
  | 'passive_income'
  // Communication & Influence
  | 'communication'
  | 'public_speaking'
  | 'networking'
  // General
  | 'general';

export interface TrainingPair {
  id: string;
  type: 'qa' | 'instruction' | 'conversation';
  instruction?: string;
  input?: string;
  output: string;
  system_prompt?: string;
  quality_score: number;
}

// ============================================================================
// Database Schema (SQLite)
// ============================================================================

export const DB_SCHEMA = `
-- Main knowledge entries table
CREATE TABLE IF NOT EXISTS knowledge_entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  -- Source info (JSON)
  source_type TEXT NOT NULL,
  source_book_title TEXT,
  source_author TEXT,
  source_chapter TEXT,
  source_page_numbers TEXT,
  source_image_path TEXT NOT NULL,
  source_capture_date TEXT NOT NULL,

  -- Content
  raw_text TEXT NOT NULL,
  cleaned_text TEXT NOT NULL,

  -- Classification
  category TEXT NOT NULL DEFAULT 'general',
  quality_score REAL DEFAULT 0,
  ocr_confidence REAL DEFAULT 0,
  manually_reviewed INTEGER DEFAULT 0
);

-- Topics table (many-to-many)
CREATE TABLE IF NOT EXISTS topics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS entry_topics (
  entry_id TEXT NOT NULL,
  topic_id INTEGER NOT NULL,
  PRIMARY KEY (entry_id, topic_id),
  FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

-- Tags table (many-to-many)
CREATE TABLE IF NOT EXISTS tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS entry_tags (
  entry_id TEXT NOT NULL,
  tag_id INTEGER NOT NULL,
  PRIMARY KEY (entry_id, tag_id),
  FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
  FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Training pairs generated from entries
CREATE TABLE IF NOT EXISTS training_pairs (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('qa', 'instruction', 'conversation')),
  instruction TEXT,
  input TEXT,
  output TEXT NOT NULL,
  system_prompt TEXT,
  quality_score REAL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
);

-- Related entries (self-referential many-to-many)
CREATE TABLE IF NOT EXISTS related_entries (
  entry_id TEXT NOT NULL,
  related_entry_id TEXT NOT NULL,
  relationship_type TEXT DEFAULT 'related',
  PRIMARY KEY (entry_id, related_entry_id),
  FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
  FOREIGN KEY (related_entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_entries_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_entries_quality ON knowledge_entries(quality_score);
CREATE INDEX IF NOT EXISTS idx_entries_source_type ON knowledge_entries(source_type);
CREATE INDEX IF NOT EXISTS idx_training_pairs_entry ON training_pairs(entry_id);
CREATE INDEX IF NOT EXISTS idx_training_pairs_type ON training_pairs(type);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
  id,
  raw_text,
  cleaned_text,
  content='knowledge_entries',
  content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_entries BEGIN
  INSERT INTO knowledge_fts(id, raw_text, cleaned_text)
  VALUES (new.id, new.raw_text, new.cleaned_text);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_entries BEGIN
  INSERT INTO knowledge_fts(knowledge_fts, id, raw_text, cleaned_text)
  VALUES('delete', old.id, old.raw_text, old.cleaned_text);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_entries BEGIN
  INSERT INTO knowledge_fts(knowledge_fts, id, raw_text, cleaned_text)
  VALUES('delete', old.id, old.raw_text, old.cleaned_text);
  INSERT INTO knowledge_fts(id, raw_text, cleaned_text)
  VALUES (new.id, new.raw_text, new.cleaned_text);
END;
`;

// ============================================================================
// Category Definitions (for classification)
// ============================================================================

export const CATEGORY_DEFINITIONS: Record<
  Category,
  {
    description: string;
    keywords: string[];
    examples: string[];
  }
> = {
  candlestick_patterns: {
    description: 'Japanese candlestick patterns for price action trading',
    keywords: [
      'doji',
      'hammer',
      'engulfing',
      'morning star',
      'evening star',
      'harami',
      'shooting star',
      'hanging man',
      'marubozu',
      'spinning top',
    ],
    examples: ['bullish engulfing pattern', 'bearish harami', 'three white soldiers'],
  },
  chart_patterns: {
    description: 'Technical chart patterns for trend analysis',
    keywords: [
      'head and shoulders',
      'double top',
      'double bottom',
      'triangle',
      'wedge',
      'flag',
      'pennant',
      'cup and handle',
      'channel',
    ],
    examples: [
      'ascending triangle breakout',
      'head and shoulders reversal',
      'bull flag continuation',
    ],
  },
  technical_indicators: {
    description: 'Mathematical indicators derived from price/volume data',
    keywords: [
      'RSI',
      'MACD',
      'moving average',
      'bollinger',
      'stochastic',
      'ADX',
      'ATR',
      'volume',
      'momentum',
      'oscillator',
    ],
    examples: ['RSI divergence', 'MACD crossover', 'moving average convergence'],
  },
  risk_management: {
    description: 'Position sizing, stop losses, and capital preservation',
    keywords: [
      'stop loss',
      'position size',
      'risk reward',
      'drawdown',
      'kelly criterion',
      'portfolio',
      'hedge',
      'diversification',
    ],
    examples: ['2% risk rule', 'trailing stop strategy', 'position sizing formula'],
  },
  trading_psychology: {
    description: 'Mental aspects of trading and emotional control',
    keywords: [
      'fear',
      'greed',
      'discipline',
      'patience',
      'bias',
      'emotion',
      'mindset',
      'journal',
      'routine',
    ],
    examples: ['overcoming FOMO', 'trading journal practice', 'emotional detachment'],
  },
  market_structure: {
    description: 'How markets move, trends, and price levels',
    keywords: [
      'support',
      'resistance',
      'trend',
      'range',
      'breakout',
      'breakdown',
      'higher high',
      'lower low',
      'swing',
    ],
    examples: ['break of structure', 'trend reversal', 'range-bound trading'],
  },
  options_strategies: {
    description: 'Options trading strategies and Greeks',
    keywords: [
      'call',
      'put',
      'spread',
      'straddle',
      'strangle',
      'iron condor',
      'butterfly',
      'theta',
      'delta',
      'gamma',
      'vega',
      'IV',
    ],
    examples: ['covered call strategy', 'iron condor setup', 'theta decay play'],
  },
  fundamental_analysis: {
    description: 'Company and economic analysis for valuation',
    keywords: [
      'earnings',
      'revenue',
      'P/E ratio',
      'balance sheet',
      'cash flow',
      'valuation',
      'sector',
      'economic',
    ],
    examples: ['earnings surprise trade', 'sector rotation', 'value investing criteria'],
  },
  order_flow: {
    description: 'Reading order books, tape, and institutional activity',
    keywords: [
      'order book',
      'tape reading',
      'bid ask',
      'spread',
      'liquidity',
      'dark pool',
      'block trade',
      'imbalance',
    ],
    examples: ['order flow imbalance', 'tape reading signals', 'dark pool prints'],
  },
  volume_analysis: {
    description: 'Volume-based analysis and indicators',
    keywords: [
      'volume',
      'OBV',
      'VWAP',
      'volume profile',
      'accumulation',
      'distribution',
      'climax',
      'exhaustion',
    ],
    examples: ['volume climax reversal', 'VWAP bounce', 'accumulation zone'],
  },

  // ==================== SALES & PERSUASION ====================
  sales_techniques: {
    description: 'Sales methodologies and closing techniques',
    keywords: [
      'prospect',
      'qualify',
      'objection',
      'close',
      'pitch',
      'cold call',
      'follow up',
      'pipeline',
      'conversion',
      'upsell',
    ],
    examples: ['handling objections', 'qualifying leads', 'sales pitch structure'],
  },
  negotiation: {
    description: 'Negotiation tactics and deal-making',
    keywords: [
      'negotiate',
      'leverage',
      'BATNA',
      'concession',
      'anchor',
      'compromise',
      'win-win',
      'deadlock',
      'counter offer',
    ],
    examples: ['anchoring technique', 'creating leverage', 'win-win negotiation'],
  },
  persuasion: {
    description: 'Psychology of influence and persuasion',
    keywords: [
      'influence',
      'reciprocity',
      'scarcity',
      'authority',
      'social proof',
      'liking',
      'commitment',
      'consistency',
      'cialdini',
    ],
    examples: ['reciprocity principle', 'social proof in marketing', 'scarcity tactics'],
  },
  closing: {
    description: 'Closing techniques and deal finalization',
    keywords: [
      'assumptive close',
      'trial close',
      'urgency',
      'now or never',
      'summary close',
      'alternative close',
      'puppy dog close',
    ],
    examples: ['assumptive close technique', 'creating urgency', 'trial closing'],
  },

  // ==================== BUSINESS & ENTREPRENEURSHIP ====================
  business_strategy: {
    description: 'Business strategy and competitive advantage',
    keywords: [
      'strategy',
      'competitive advantage',
      'moat',
      'disruption',
      'scale',
      'pivot',
      'business model',
      'value proposition',
      'differentiation',
    ],
    examples: ['building competitive moat', 'business model innovation', 'strategic pivot'],
  },
  marketing: {
    description: 'Marketing strategies and customer acquisition',
    keywords: [
      'marketing',
      'branding',
      'positioning',
      'funnel',
      'acquisition',
      'retention',
      'SEO',
      'content',
      'viral',
      'growth hacking',
    ],
    examples: ['marketing funnel optimization', 'brand positioning', 'growth hacking tactics'],
  },
  leadership: {
    description: 'Leadership principles and team management',
    keywords: [
      'leader',
      'vision',
      'inspire',
      'delegate',
      'empower',
      'culture',
      'team building',
      'servant leadership',
      'executive',
    ],
    examples: ['servant leadership', 'building team culture', 'inspiring vision'],
  },
  management: {
    description: 'Management techniques and organizational skills',
    keywords: [
      'manage',
      'organize',
      'prioritize',
      'delegate',
      'KPI',
      'metrics',
      'performance',
      'feedback',
      'one on one',
    ],
    examples: ['effective delegation', 'KPI tracking', 'performance reviews'],
  },
  startups: {
    description: 'Startup building and entrepreneurship',
    keywords: [
      'startup',
      'founder',
      'MVP',
      'product market fit',
      'fundraising',
      'venture',
      'bootstrap',
      'iterate',
      'pivot',
      'traction',
    ],
    examples: ['finding product market fit', 'MVP development', 'fundraising strategies'],
  },

  // ==================== SELF-HELP & PERSONAL DEVELOPMENT ====================
  mindset: {
    description: 'Mental frameworks and belief systems',
    keywords: [
      'mindset',
      'belief',
      'growth mindset',
      'fixed mindset',
      'abundance',
      'scarcity',
      'limiting belief',
      'reframe',
      'perspective',
    ],
    examples: ['growth vs fixed mindset', 'overcoming limiting beliefs', 'abundance mindset'],
  },
  habits: {
    description: 'Habit formation and behavior change',
    keywords: [
      'habit',
      'routine',
      'trigger',
      'cue',
      'reward',
      'habit loop',
      'atomic habits',
      'keystone habit',
      'habit stacking',
    ],
    examples: ['habit loop mechanics', 'keystone habits', 'habit stacking technique'],
  },
  productivity: {
    description: 'Time management and personal productivity',
    keywords: [
      'productivity',
      'time management',
      'focus',
      'deep work',
      'pomodoro',
      'batch',
      'prioritize',
      'eliminate',
      '80/20',
      'pareto',
    ],
    examples: ['deep work practice', 'Pomodoro technique', '80/20 principle'],
  },
  motivation: {
    description: 'Motivation and drive',
    keywords: [
      'motivation',
      'drive',
      'purpose',
      'why',
      'passion',
      'intrinsic',
      'extrinsic',
      'goal',
      'vision',
      'inspire',
    ],
    examples: ['finding your why', 'intrinsic motivation', 'setting compelling goals'],
  },
  success_principles: {
    description: 'Universal principles of success',
    keywords: [
      'success',
      'principle',
      'law',
      'secret',
      'millionaire',
      'wealthy',
      'achieve',
      'excellence',
      'mastery',
      'compound',
    ],
    examples: ['compound effect', 'laws of success', 'mastery principles'],
  },

  // ==================== WEALTH & INVESTING ====================
  wealth_building: {
    description: 'Long-term wealth accumulation strategies',
    keywords: [
      'wealth',
      'rich',
      'millionaire',
      'billionaire',
      'compound',
      'invest',
      'asset',
      'liability',
      'net worth',
      'financial freedom',
    ],
    examples: ['assets vs liabilities', 'compound interest', 'building net worth'],
  },
  real_estate: {
    description: 'Real estate investing strategies',
    keywords: [
      'real estate',
      'property',
      'rental',
      'cash flow',
      'appreciation',
      'leverage',
      'mortgage',
      'flip',
      'landlord',
      'tenant',
    ],
    examples: ['rental property cash flow', 'house flipping', 'real estate leverage'],
  },
  passive_income: {
    description: 'Passive income streams and systems',
    keywords: [
      'passive income',
      'residual',
      'royalty',
      'dividend',
      'automation',
      'system',
      'cash flow',
      'multiple streams',
    ],
    examples: ['building passive income', 'dividend investing', 'income automation'],
  },

  // ==================== COMMUNICATION & INFLUENCE ====================
  communication: {
    description: 'Effective communication skills',
    keywords: [
      'communicate',
      'listen',
      'empathy',
      'rapport',
      'body language',
      'tone',
      'clarity',
      'message',
      'feedback',
    ],
    examples: ['active listening', 'building rapport', 'clear communication'],
  },
  public_speaking: {
    description: 'Public speaking and presentation skills',
    keywords: [
      'speech',
      'presentation',
      'audience',
      'stage',
      'storytelling',
      'delivery',
      'confidence',
      'TED',
      'keynote',
    ],
    examples: ['storytelling in speeches', 'confident delivery', 'engaging audiences'],
  },
  networking: {
    description: 'Professional networking and relationship building',
    keywords: [
      'network',
      'connection',
      'relationship',
      'referral',
      'introduction',
      'value',
      'follow up',
      'mastermind',
    ],
    examples: ['strategic networking', 'providing value first', 'mastermind groups'],
  },

  // ==================== GENERAL ====================
  general: {
    description: 'General knowledge and miscellaneous topics',
    keywords: [],
    examples: [],
  },
};

// ============================================================================
// Training Data Formats
// ============================================================================

export interface AlpacaFormat {
  instruction: string;
  input: string;
  output: string;
}

export interface ShareGPTFormat {
  conversations: Array<{
    from: 'human' | 'gpt' | 'system';
    value: string;
  }>;
}

export interface ChatMLFormat {
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
}

// ============================================================================
// Export Utilities
// ============================================================================

export function toAlpacaFormat(pair: TrainingPair): AlpacaFormat {
  return {
    instruction: pair.instruction || 'Provide information about the following trading concept.',
    input: pair.input || '',
    output: pair.output,
  };
}

export function toShareGPTFormat(pair: TrainingPair): ShareGPTFormat {
  const conversations: ShareGPTFormat['conversations'] = [];

  if (pair.system_prompt) {
    conversations.push({ from: 'system', value: pair.system_prompt });
  }

  if (pair.instruction) {
    const humanMessage = pair.input ? `${pair.instruction}\n\n${pair.input}` : pair.instruction;
    conversations.push({ from: 'human', value: humanMessage });
  }

  conversations.push({ from: 'gpt', value: pair.output });

  return { conversations };
}

export function toChatMLFormat(pair: TrainingPair): ChatMLFormat {
  const messages: ChatMLFormat['messages'] = [];

  if (pair.system_prompt) {
    messages.push({ role: 'system', content: pair.system_prompt });
  }

  if (pair.instruction) {
    const userMessage = pair.input ? `${pair.instruction}\n\n${pair.input}` : pair.instruction;
    messages.push({ role: 'user', content: userMessage });
  }

  messages.push({ role: 'assistant', content: pair.output });

  return { messages };
}
