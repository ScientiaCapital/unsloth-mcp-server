/**
 * Construction & Trades Dataset Builder
 *
 * Tools for building high-quality training datasets for construction,
 * MEP, and skilled trades domains.
 *
 * Data Sources:
 * 1. Your book photos (via PWA)
 * 2. Public forum Q&A (Reddit, trade forums)
 * 3. Government/public documents (OSHA, public codes)
 * 4. Manufacturer documentation (public PDFs)
 * 5. Synthetic generation from seed content
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// Simple UUID generator without external dependency
function generateId(): string {
  return crypto.randomUUID();
}
import { Category, TrainingPair } from './schema.js';
import { SYSTEM_PROMPTS } from './training-generator.js';
import logger from '../utils/logger.js';

// ============================================================================
// Dataset Entry Types
// ============================================================================

export interface DatasetEntry {
  id: string;
  source: DataSource;
  category: Category;
  subcategory?: string;
  content: ContentBlock;
  training_pairs: TrainingPair[];
  metadata: EntryMetadata;
  quality: QualityMetrics;
}

export interface DataSource {
  type:
    | 'book_photo'
    | 'forum_qa'
    | 'government_doc'
    | 'manufacturer_doc'
    | 'synthetic'
    | 'manual_entry';
  name: string;
  url?: string;
  author?: string;
  license?: string;
  date_collected: string;
}

export interface ContentBlock {
  raw_text: string;
  cleaned_text: string;
  format: 'prose' | 'qa' | 'list' | 'table' | 'code_snippet' | 'procedure';
  language: string;
}

export interface EntryMetadata {
  topics: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  trade_relevance: string[]; // e.g., ['electrician', 'HVAC tech', 'plumber']
  code_references?: string[]; // e.g., ['NEC 210.8', 'NFPA 13']
  tools_mentioned?: string[];
  materials_mentioned?: string[];
}

export interface QualityMetrics {
  accuracy_score: number; // 0-100
  completeness_score: number; // 0-100
  clarity_score: number; // 0-100
  practical_value: number; // 0-100
  verified: boolean;
  verified_by?: string;
  verification_date?: string;
}

// ============================================================================
// Public Domain Sources for Construction Data
// ============================================================================

export const PUBLIC_DATA_SOURCES = {
  // Government Sources (Public Domain)
  government: [
    {
      name: 'OSHA eTool - Electrical',
      url: 'https://www.osha.gov/electrical',
      type: 'safety',
      license: 'Public Domain',
    },
    {
      name: 'OSHA Construction Standards',
      url: 'https://www.osha.gov/construction',
      type: 'safety',
      license: 'Public Domain',
    },
    {
      name: 'EPA Energy Star',
      url: 'https://www.energystar.gov',
      type: 'energy',
      license: 'Public Domain',
    },
    {
      name: 'DOE Building Technologies',
      url: 'https://www.energy.gov/eere/buildings',
      type: 'energy',
      license: 'Public Domain',
    },
    {
      name: 'HUD Housing Guidelines',
      url: 'https://www.hud.gov',
      type: 'construction',
      license: 'Public Domain',
    },
  ],

  // Open Educational Resources
  education: [
    {
      name: 'MIT OpenCourseWare - Building Technology',
      url: 'https://ocw.mit.edu',
      type: 'engineering',
      license: 'CC BY-NC-SA',
    },
    {
      name: 'NCCER Open Resources',
      url: 'https://www.nccer.org',
      type: 'trades',
      license: 'Educational',
    },
  ],

  // Community Q&A (User-Generated, Check ToS)
  forums: [
    {
      name: 'r/electricians',
      url: 'https://reddit.com/r/electricians',
      type: 'electrical',
      license: 'Reddit ToS - Check before scraping',
    },
    {
      name: 'r/HVAC',
      url: 'https://reddit.com/r/HVAC',
      type: 'hvac',
      license: 'Reddit ToS - Check before scraping',
    },
    {
      name: 'r/Plumbing',
      url: 'https://reddit.com/r/Plumbing',
      type: 'plumbing',
      license: 'Reddit ToS - Check before scraping',
    },
    {
      name: 'Contractor Talk Forum',
      url: 'https://www.contractortalk.com',
      type: 'general',
      license: 'Check ToS',
    },
    {
      name: 'HVAC-Talk',
      url: 'https://hvac-talk.com',
      type: 'hvac',
      license: 'Check ToS',
    },
  ],

  // Manufacturer Resources (Many have public installation guides)
  manufacturers: [
    { name: 'Carrier HVAC Guides', type: 'hvac', license: 'Check individual docs' },
    { name: 'Trane Installation Manuals', type: 'hvac', license: 'Check individual docs' },
    { name: 'Square D/Schneider Electric', type: 'electrical', license: 'Check individual docs' },
    { name: 'Rheem Water Heaters', type: 'plumbing', license: 'Check individual docs' },
  ],
};

// ============================================================================
// Dataset Templates for Each Trade
// ============================================================================

export const TRADE_TEMPLATES: Record<
  string,
  {
    qa_patterns: string[];
    instruction_patterns: string[];
    topics: string[];
  }
> = {
  electrical: {
    qa_patterns: [
      'What size wire do I need for {amperage}A at {distance} feet?',
      'How do I calculate the load for {equipment}?',
      'What does NEC {code_section} require for {application}?',
      'How do I properly ground {equipment}?',
      'What is the derating factor for {condition}?',
      'How many outlets can I put on a {amperage}A circuit?',
      'What is the difference between {term1} and {term2}?',
      'How do I troubleshoot {problem}?',
    ],
    instruction_patterns: [
      'Explain how to install {equipment} according to NEC code.',
      'Calculate the service entrance size for a {size} square foot home.',
      'Describe the proper procedure for {task}.',
      'List the tools needed for {job}.',
    ],
    topics: [
      'wire sizing',
      'load calculations',
      'grounding',
      'bonding',
      'GFCI',
      'AFCI',
      'panel installation',
      'conduit bending',
      'voltage drop',
      'motor circuits',
      'lighting circuits',
      'service entrance',
    ],
  },

  hvac: {
    qa_patterns: [
      'How do I size a system for {square_feet} square feet?',
      'What is the proper superheat for {refrigerant}?',
      'How do I calculate CFM for {application}?',
      'What causes {symptom} in a {system_type}?',
      'How do I charge a system with {refrigerant}?',
      'What is the difference between {term1} and {term2}?',
      'How do I troubleshoot {problem}?',
    ],
    instruction_patterns: [
      'Explain the refrigeration cycle.',
      'Describe how to perform a Manual J load calculation.',
      'Walk through the steps to replace {component}.',
      'Explain proper duct sizing for {CFM} CFM.',
    ],
    topics: [
      'load calculations',
      'refrigerant charging',
      'superheat',
      'subcooling',
      'duct sizing',
      'airflow',
      'heat pumps',
      'furnaces',
      'air handlers',
      'condensers',
      'evaporators',
      'thermostats',
      'controls',
    ],
  },

  plumbing: {
    qa_patterns: [
      'What size drain do I need for {fixture}?',
      'How do I properly vent {fixture}?',
      'What is the proper slope for a {size} drain?',
      'How do I size a water heater for {application}?',
      'What causes {symptom}?',
      'How do I install {fixture}?',
    ],
    instruction_patterns: [
      'Explain the DWV system design principles.',
      'Describe how to properly install a {fixture}.',
      'Walk through sizing a water supply system.',
      'Explain backflow prevention requirements.',
    ],
    topics: [
      'drain sizing',
      'venting',
      'water supply',
      'water heaters',
      'fixtures',
      'traps',
      'backflow prevention',
      'pressure',
      'pipe materials',
      'soldering',
      'PEX installation',
    ],
  },

  estimating: {
    qa_patterns: [
      'How do I estimate {work_type} for {quantity}?',
      'What is a typical labor rate for {trade}?',
      'How do I calculate markup for {project_type}?',
      'What production rate should I use for {task}?',
      'How do I account for {factor} in my estimate?',
    ],
    instruction_patterns: [
      'Walk through a takeoff for {scope}.',
      'Explain how to build a bid for {project_type}.',
      'Describe the components of construction overhead.',
      'Calculate the total cost for {scope}.',
    ],
    topics: [
      'quantity takeoff',
      'labor rates',
      'material pricing',
      'overhead',
      'profit',
      'markup',
      'production rates',
      'unit costs',
      'bid preparation',
    ],
  },

  safety: {
    qa_patterns: [
      'What PPE is required for {task}?',
      'What are OSHA requirements for {hazard}?',
      'How do I properly set up {equipment}?',
      'What is the proper procedure for {safety_task}?',
      'What training is required for {activity}?',
    ],
    instruction_patterns: [
      'Explain OSHA fall protection requirements.',
      'Describe the lockout/tagout procedure.',
      'List the components of a Job Hazard Analysis.',
      'Explain confined space entry requirements.',
    ],
    topics: [
      'fall protection',
      'PPE',
      'lockout tagout',
      'scaffolding',
      'excavation',
      'electrical safety',
      'confined spaces',
      'hazard communication',
      'first aid',
    ],
  },
};

// ============================================================================
// Dataset Builder Class
// ============================================================================

export class DatasetBuilder {
  private entries: DatasetEntry[] = [];
  private outputDir: string;

  constructor(outputDir: string = './dataset') {
    this.outputDir = outputDir;
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
  }

  /**
   * Add a manual Q&A entry
   */
  addQAPair(
    question: string,
    answer: string,
    category: Category,
    metadata: Partial<EntryMetadata> = {}
  ): string {
    const id = generateId();

    const entry: DatasetEntry = {
      id,
      source: {
        type: 'manual_entry',
        name: 'Manual Entry',
        date_collected: new Date().toISOString(),
      },
      category,
      content: {
        raw_text: `Q: ${question}\nA: ${answer}`,
        cleaned_text: `Q: ${question}\nA: ${answer}`,
        format: 'qa',
        language: 'en',
      },
      training_pairs: [
        {
          id: `${id}_qa`,
          type: 'qa',
          instruction: question,
          input: '',
          output: answer,
          system_prompt: this.getSystemPromptForCategory(category),
          quality_score: 70,
        },
      ],
      metadata: {
        topics: metadata.topics || [],
        difficulty: metadata.difficulty || 'intermediate',
        trade_relevance: metadata.trade_relevance || [],
        code_references: metadata.code_references,
        tools_mentioned: metadata.tools_mentioned,
        materials_mentioned: metadata.materials_mentioned,
      },
      quality: {
        accuracy_score: 0,
        completeness_score: 0,
        clarity_score: 0,
        practical_value: 0,
        verified: false,
      },
    };

    this.entries.push(entry);
    return id;
  }

  /**
   * Add content from a book photo (processed by OCR)
   */
  addFromBookPhoto(
    content: string,
    bookTitle: string,
    author: string,
    category: Category,
    topics: string[]
  ): string {
    const id = generateId();

    const entry: DatasetEntry = {
      id,
      source: {
        type: 'book_photo',
        name: bookTitle,
        author,
        date_collected: new Date().toISOString(),
      },
      category,
      content: {
        raw_text: content,
        cleaned_text: content.trim(),
        format: 'prose',
        language: 'en',
      },
      training_pairs: this.generatePairsFromContent(content, category, topics),
      metadata: {
        topics,
        difficulty: 'intermediate',
        trade_relevance: this.inferTradeRelevance(category),
      },
      quality: {
        accuracy_score: 80,
        completeness_score: 70,
        clarity_score: 75,
        practical_value: 80,
        verified: false,
      },
    };

    this.entries.push(entry);
    return id;
  }

  /**
   * Generate synthetic Q&A from seed topics
   */
  generateSyntheticEntries(trade: string, count: number = 10): string[] {
    const template = TRADE_TEMPLATES[trade];
    if (!template) {
      throw new Error(`Unknown trade: ${trade}`);
    }

    const ids: string[] = [];

    for (let i = 0; i < count; i++) {
      const topic = template.topics[i % template.topics.length];
      const qaPattern = template.qa_patterns[i % template.qa_patterns.length];

      // Create a placeholder Q&A that needs expert completion
      const question = qaPattern.replace(/{[^}]+}/g, `[${topic}]`);

      const id = this.addQAPair(
        question,
        `[NEEDS EXPERT ANSWER - Topic: ${topic}]`,
        this.tradeToCategory(trade),
        {
          topics: [topic],
          difficulty: 'intermediate',
          trade_relevance: [trade],
        }
      );

      ids.push(id);
    }

    return ids;
  }

  /**
   * Export dataset in various formats
   */
  exportDataset(format: 'alpaca' | 'sharegpt' | 'chatml' | 'huggingface'): string {
    const timestamp = new Date().toISOString().split('T')[0];
    let outputPath: string;
    let content: string;

    switch (format) {
      case 'alpaca':
        outputPath = path.join(this.outputDir, `trades_dataset_alpaca_${timestamp}.json`);
        content = JSON.stringify(this.toAlpacaFormat(), null, 2);
        break;

      case 'sharegpt':
        outputPath = path.join(this.outputDir, `trades_dataset_sharegpt_${timestamp}.json`);
        content = JSON.stringify(this.toShareGPTFormat(), null, 2);
        break;

      case 'chatml':
        outputPath = path.join(this.outputDir, `trades_dataset_chatml_${timestamp}.json`);
        content = JSON.stringify(this.toChatMLFormat(), null, 2);
        break;

      case 'huggingface':
        outputPath = path.join(this.outputDir, `trades_dataset_hf_${timestamp}.jsonl`);
        content = this.entries
          .flatMap((e) =>
            e.training_pairs.map((p) =>
              JSON.stringify({
                instruction: p.instruction,
                input: p.input || '',
                output: p.output,
                category: e.category,
                source: e.source.type,
              })
            )
          )
          .join('\n');
        break;

      default:
        throw new Error(`Unknown format: ${format}`);
    }

    fs.writeFileSync(outputPath, content);
    logger.info(`Dataset exported to ${outputPath}`);
    return outputPath;
  }

  /**
   * Get dataset statistics
   */
  getStats(): {
    total_entries: number;
    total_pairs: number;
    by_category: Record<string, number>;
    by_source: Record<string, number>;
    verified_count: number;
  } {
    const byCategory: Record<string, number> = {};
    const bySource: Record<string, number> = {};
    let verifiedCount = 0;

    for (const entry of this.entries) {
      byCategory[entry.category] = (byCategory[entry.category] || 0) + 1;
      bySource[entry.source.type] = (bySource[entry.source.type] || 0) + 1;
      if (entry.quality.verified) verifiedCount++;
    }

    return {
      total_entries: this.entries.length,
      total_pairs: this.entries.reduce((sum, e) => sum + e.training_pairs.length, 0),
      by_category: byCategory,
      by_source: bySource,
      verified_count: verifiedCount,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private getSystemPromptForCategory(category: Category): string {
    const categoryToPrompt: Record<string, keyof typeof SYSTEM_PROMPTS> = {
      electrical_systems: 'electrical_expert',
      hvac_systems: 'hvac_technician',
      plumbing_systems: 'mep_engineer',
      solar_power: 'solar_specialist',
      energy_systems: 'energy_engineer',
      mechanical_engineering: 'mep_engineer',
      fire_protection: 'code_inspector',
      building_automation: 'mep_engineer',
      estimating: 'estimator',
      project_management_construction: 'project_manager',
      building_codes: 'code_inspector',
      safety_compliance: 'safety_officer',
      blueprints_drawings: 'project_manager',
      contractor_business: 'contractor_coach',
      bidding_proposals: 'estimator',
      client_management: 'contractor_coach',
    };

    const promptKey = categoryToPrompt[category] || 'trading_expert';
    return SYSTEM_PROMPTS[promptKey] || SYSTEM_PROMPTS.trading_expert;
  }

  private inferTradeRelevance(category: Category): string[] {
    const categoryToTrades: Record<string, string[]> = {
      electrical_systems: ['electrician', 'electrical contractor'],
      hvac_systems: ['HVAC technician', 'mechanical contractor'],
      plumbing_systems: ['plumber', 'mechanical contractor'],
      solar_power: ['solar installer', 'electrician'],
      fire_protection: ['fire protection contractor', 'sprinkler fitter'],
      estimating: ['estimator', 'project manager'],
      safety_compliance: ['safety officer', 'all trades'],
    };

    return categoryToTrades[category] || ['general contractor'];
  }

  private tradeToCategory(trade: string): Category {
    const tradeToCategory: Record<string, Category> = {
      electrical: 'electrical_systems',
      hvac: 'hvac_systems',
      plumbing: 'plumbing_systems',
      estimating: 'estimating',
      safety: 'safety_compliance',
    };

    return tradeToCategory[trade] || 'general';
  }

  private generatePairsFromContent(
    content: string,
    category: Category,
    topics: string[]
  ): TrainingPair[] {
    const pairs: TrainingPair[] = [];
    const systemPrompt = this.getSystemPromptForCategory(category);

    // Generate Q&A pair for each topic
    for (const topic of topics.slice(0, 3)) {
      pairs.push({
        id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        type: 'qa',
        instruction: `Explain ${topic} based on this content.`,
        input: content.substring(0, 500),
        output: this.extractRelevantContent(content, topic),
        system_prompt: systemPrompt,
        quality_score: 65,
      });
    }

    // Add summarization pair
    pairs.push({
      id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      type: 'instruction',
      instruction: 'Summarize the key points from this content.',
      input: content,
      output: this.generateSummary(content),
      system_prompt: systemPrompt,
      quality_score: 60,
    });

    return pairs;
  }

  private extractRelevantContent(content: string, topic: string): string {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    const relevant = sentences.filter((s) => s.toLowerCase().includes(topic.toLowerCase()));

    if (relevant.length > 0) {
      return relevant.slice(0, 3).join('. ').trim() + '.';
    }

    return sentences.slice(0, 3).join('. ').trim() + '.';
  }

  private generateSummary(content: string): string {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    return sentences.slice(0, 4).join('. ').trim() + '.';
  }

  private toAlpacaFormat(): Array<{ instruction: string; input: string; output: string }> {
    return this.entries.flatMap((e) =>
      e.training_pairs.map((p) => ({
        instruction: p.instruction || '',
        input: p.input || '',
        output: p.output,
      }))
    );
  }

  private toShareGPTFormat(): Array<{ conversations: Array<{ from: string; value: string }> }> {
    return this.entries.flatMap((e) =>
      e.training_pairs.map((p) => ({
        conversations: [
          ...(p.system_prompt ? [{ from: 'system', value: p.system_prompt }] : []),
          { from: 'human', value: p.instruction || '' },
          { from: 'gpt', value: p.output },
        ],
      }))
    );
  }

  private toChatMLFormat(): Array<{
    messages: Array<{ role: string; content: string }>;
  }> {
    return this.entries.flatMap((e) =>
      e.training_pairs.map((p) => ({
        messages: [
          ...(p.system_prompt ? [{ role: 'system', content: p.system_prompt }] : []),
          { role: 'user', content: p.instruction || '' },
          { role: 'assistant', content: p.output },
        ],
      }))
    );
  }
}

// ============================================================================
// Dataset Contribution Guide
// ============================================================================

export const CONTRIBUTION_GUIDE = `
# Contributing to the Trades & Construction Dataset

## Goal
Build the first comprehensive open-source training dataset for construction,
MEP, and skilled trades knowledge.

## How to Contribute

### 1. Book Photos (via PWA)
- Capture pages from your trade manuals and code books
- Add proper metadata (title, author, chapter)
- Review OCR accuracy before submitting

### 2. Q&A Pairs
- Submit real questions from your work experience
- Provide detailed, accurate answers
- Include code references where applicable

### 3. Verification
- Review and verify existing entries
- Flag inaccurate or outdated information
- Add code citations and references

### 4. Synthetic Completion
- Complete placeholder entries with expert knowledge
- Add real-world examples and scenarios

## Quality Guidelines

### Required for Each Entry:
- [ ] Technically accurate
- [ ] Current (codes/practices not outdated)
- [ ] Clear and understandable
- [ ] Practical application included
- [ ] Code references where applicable

### Format Requirements:
- Use proper trade terminology
- Include units (ft, in, AWG, CFM, etc.)
- Reference specific code sections (NEC 210.8, NFPA 13, etc.)
- Describe safety considerations

## License
All contributions should be:
- Original content you created, OR
- From public domain sources (government, expired copyright)

Do NOT submit:
- Copyrighted textbook content
- Code book text (NEC, IBC are copyrighted)
- Proprietary manufacturer specifications
`;
