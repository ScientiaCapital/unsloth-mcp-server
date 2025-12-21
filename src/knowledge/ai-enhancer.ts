/**
 * AI-Powered Training Data Enhancement
 *
 * Uses Claude to generate high-quality training pairs from captured content.
 * Includes quality scoring, difficulty assessment, and expert validation.
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { Category, TrainingPair, CATEGORY_DEFINITIONS } from './schema.js';
import { SYSTEM_PROMPTS } from './training-generator.js';
import logger from '../utils/logger.js';

const execPromise = promisify(exec);

// ============================================================================
// Quality Criteria
// ============================================================================

export interface QualityReport {
  overall_score: number; // 0-100
  dimensions: {
    accuracy: { score: number; feedback: string };
    completeness: { score: number; feedback: string };
    clarity: { score: number; feedback: string };
    practicality: { score: number; feedback: string };
    code_compliance: { score: number; feedback: string };
  };
  suggestions: string[];
  ready_for_training: boolean;
}

export interface EnhancedTrainingPair extends TrainingPair {
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  prerequisites: string[];
  related_topics: string[];
  code_references: string[];
  safety_notes?: string[];
  common_mistakes?: string[];
  real_world_example?: string;
}

// ============================================================================
// AI-Powered Q&A Generation
// ============================================================================

/**
 * Generate high-quality Q&A pairs using Claude
 */
export async function generateExpertQA(
  content: string,
  category: Category,
  options: {
    num_pairs?: number;
    difficulty?: 'beginner' | 'intermediate' | 'advanced' | 'expert';
    include_safety?: boolean;
    include_code_refs?: boolean;
  } = {}
): Promise<EnhancedTrainingPair[]> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    logger.warn('No ANTHROPIC_API_KEY - using template generation');
    return generateTemplateQA(content, category, options.num_pairs || 5);
  }

  const categoryDef = CATEGORY_DEFINITIONS[category];
  const numPairs = options.num_pairs || 5;
  const difficulty = options.difficulty || 'intermediate';

  const prompt = `You are an expert in ${categoryDef.description}.

Based on the following content, generate ${numPairs} high-quality Q&A training pairs for fine-tuning an AI assistant that helps tradespeople and contractors.

CONTENT:
"""
${content.substring(0, 3000)}
"""

REQUIREMENTS:
1. Questions should be practical and job-relevant
2. Answers should be detailed (3-5 sentences minimum)
3. Include specific measurements, code references, or specifications where applicable
4. Target difficulty level: ${difficulty}
${options.include_safety ? '5. Include relevant safety considerations' : ''}
${options.include_code_refs ? '6. Reference applicable codes (NEC, NFPA, IPC, etc.) where relevant' : ''}

Return a JSON array with this exact structure:
[
  {
    "question": "practical question a tradesperson would ask",
    "answer": "detailed, accurate answer with specifics",
    "difficulty": "${difficulty}",
    "prerequisites": ["list of concepts needed to understand this"],
    "related_topics": ["related concepts to explore"],
    "code_references": ["NEC 210.8", "etc if applicable"],
    "safety_notes": ["safety considerations if applicable"],
    "common_mistakes": ["mistakes to avoid"],
    "real_world_example": "brief practical scenario"
  }
]

Generate ${numPairs} pairs. Be specific and practical.`;

  const script = `
import anthropic
import json

try:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": """${prompt.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"""}]
    )

    response_text = message.content[0].text

    # Extract JSON
    import re
    json_match = re.search(r'\\[.*\\]', response_text, re.DOTALL)
    if json_match:
        pairs = json.loads(json_match.group())
        print(json.dumps({"success": True, "pairs": pairs}))
    else:
        print(json.dumps({"success": False, "error": "No JSON found"}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 90000 });
    const result = JSON.parse(stdout.trim());

    if (!result.success) {
      logger.warn('AI generation failed', { error: result.error });
      return generateTemplateQA(content, category, numPairs);
    }

    return result.pairs.map(
      (
        pair: {
          question: string;
          answer: string;
          difficulty: string;
          prerequisites: string[];
          related_topics: string[];
          code_references: string[];
          safety_notes: string[];
          common_mistakes: string[];
          real_world_example: string;
        },
        idx: number
      ) => ({
        id: `ai_${Date.now()}_${idx}`,
        type: 'qa' as const,
        instruction: pair.question,
        input: '',
        output: pair.answer,
        system_prompt: getSystemPromptForCategory(category),
        quality_score: 85,
        difficulty: pair.difficulty || difficulty,
        prerequisites: pair.prerequisites || [],
        related_topics: pair.related_topics || [],
        code_references: pair.code_references || [],
        safety_notes: pair.safety_notes,
        common_mistakes: pair.common_mistakes,
        real_world_example: pair.real_world_example,
      })
    );
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logger.warn('AI generation error', { error: errorMessage });
    return generateTemplateQA(content, category, numPairs);
  }
}

/**
 * Evaluate quality of a training pair
 */
export async function evaluateQuality(
  pair: TrainingPair,
  category: Category
): Promise<QualityReport> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return generateBasicQualityReport(pair);
  }

  const categoryDef = CATEGORY_DEFINITIONS[category];

  const prompt = `You are a quality evaluator for training data in ${categoryDef.description}.

Evaluate this Q&A pair for training an AI assistant:

QUESTION: ${pair.instruction}
ANSWER: ${pair.output}

Score each dimension 0-100 and provide specific feedback:

1. ACCURACY: Is the information technically correct?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is it easy to understand?
4. PRACTICALITY: Is it useful for real work?
5. CODE_COMPLIANCE: Does it correctly reference codes/standards?

Return JSON:
{
  "accuracy": {"score": 0-100, "feedback": "specific feedback"},
  "completeness": {"score": 0-100, "feedback": "specific feedback"},
  "clarity": {"score": 0-100, "feedback": "specific feedback"},
  "practicality": {"score": 0-100, "feedback": "specific feedback"},
  "code_compliance": {"score": 0-100, "feedback": "specific feedback"},
  "suggestions": ["improvement 1", "improvement 2"],
  "ready_for_training": true/false
}`;

  const script = `
import anthropic
import json

try:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": """${prompt.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"""}]
    )

    response_text = message.content[0].text

    import re
    json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        print(json.dumps({"success": True, "result": result}))
    else:
        print(json.dumps({"success": False, "error": "No JSON found"}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 30000 });
    const result = JSON.parse(stdout.trim());

    if (!result.success) {
      return generateBasicQualityReport(pair);
    }

    const dims = result.result;
    const scores = [
      dims.accuracy.score,
      dims.completeness.score,
      dims.clarity.score,
      dims.practicality.score,
      dims.code_compliance.score,
    ];
    const overall = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);

    return {
      overall_score: overall,
      dimensions: dims,
      suggestions: dims.suggestions || [],
      ready_for_training: dims.ready_for_training ?? overall >= 70,
    };
  } catch {
    return generateBasicQualityReport(pair);
  }
}

/**
 * Improve a training pair based on quality feedback
 */
export async function improveTrainingPair(
  pair: TrainingPair,
  qualityReport: QualityReport,
  category: Category
): Promise<TrainingPair> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey || qualityReport.overall_score >= 85) {
    return pair;
  }

  const categoryDef = CATEGORY_DEFINITIONS[category];

  const prompt = `You are an expert in ${categoryDef.description}.

Improve this training pair based on the quality feedback:

ORIGINAL QUESTION: ${pair.instruction}
ORIGINAL ANSWER: ${pair.output}

QUALITY FEEDBACK:
${qualityReport.suggestions.map((s) => `- ${s}`).join('\n')}

Scores:
- Accuracy: ${qualityReport.dimensions.accuracy.score}/100 - ${qualityReport.dimensions.accuracy.feedback}
- Completeness: ${qualityReport.dimensions.completeness.score}/100 - ${qualityReport.dimensions.completeness.feedback}
- Clarity: ${qualityReport.dimensions.clarity.score}/100 - ${qualityReport.dimensions.clarity.feedback}
- Practicality: ${qualityReport.dimensions.practicality.score}/100 - ${qualityReport.dimensions.practicality.feedback}

Return improved Q&A as JSON:
{
  "question": "improved question if needed",
  "answer": "improved, more complete, accurate answer"
}`;

  const script = `
import anthropic
import json

try:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": """${prompt.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"""}]
    )

    response_text = message.content[0].text

    import re
    json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        print(json.dumps({"success": True, "result": result}))
    else:
        print(json.dumps({"success": False, "error": "No JSON found"}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 30000 });
    const result = JSON.parse(stdout.trim());

    if (!result.success) {
      return pair;
    }

    return {
      ...pair,
      id: `improved_${pair.id}`,
      instruction: result.result.question || pair.instruction,
      output: result.result.answer || pair.output,
      quality_score: Math.min(95, qualityReport.overall_score + 15),
    };
  } catch {
    return pair;
  }
}

// ============================================================================
// Difficulty Assessment
// ============================================================================

export interface DifficultyAssessment {
  level: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  score: number; // 1-10
  factors: string[];
  prerequisites: string[];
  estimated_experience: string; // e.g., "1-2 years in trade"
}

/**
 * Assess difficulty level of content
 */
export function assessDifficulty(content: string, category: Category): DifficultyAssessment {
  const textLower = content.toLowerCase();

  // Expert indicators
  const expertTerms = [
    'coordination study',
    'arc flash',
    'psychrometric',
    'enthalpy',
    'hydraulic calculation',
    'sprinkler density',
    'demand factor',
    'diversity factor',
    'sequence of operations',
    'commissioning',
  ];

  // Advanced indicators
  const advancedTerms = [
    'load calculation',
    'voltage drop',
    'derating',
    'sizing',
    'Manual J',
    'NFPA',
    'NEC Article',
    'pressure drop',
    'friction loss',
    'superheat',
    'subcooling',
  ];

  // Intermediate indicators
  const intermediateTerms = [
    'circuit',
    'breaker',
    'ductwork',
    'fitting',
    'valve',
    'thermostat',
    'grounding',
    'venting',
    'slope',
    'pitch',
  ];

  let score = 3; // Start at beginner-intermediate
  const factors: string[] = [];
  const prerequisites: string[] = [];

  // Check for expert terms
  for (const term of expertTerms) {
    if (textLower.includes(term)) {
      score += 2;
      factors.push(`Contains expert concept: ${term}`);
    }
  }

  // Check for advanced terms
  for (const term of advancedTerms) {
    if (textLower.includes(term)) {
      score += 1;
      factors.push(`Contains advanced concept: ${term}`);
    }
  }

  // Check for intermediate terms
  for (const term of intermediateTerms) {
    if (textLower.includes(term)) {
      score += 0.3;
    }
  }

  // Check for calculations/formulas
  if (/[=×÷+-]\s*\d/.test(content) || /formula|calculate|equation/i.test(content)) {
    score += 1;
    factors.push('Contains calculations or formulas');
    prerequisites.push('Basic math skills');
  }

  // Check for code references
  if (/NEC|NFPA|IBC|IRC|IPC|IMC/i.test(content)) {
    score += 1;
    factors.push('References building/trade codes');
    prerequisites.push('Familiarity with trade codes');
  }

  // Normalize score
  score = Math.min(10, Math.max(1, score));

  let level: DifficultyAssessment['level'];
  let experience: string;

  if (score <= 3) {
    level = 'beginner';
    experience = 'Apprentice or student';
  } else if (score <= 5) {
    level = 'intermediate';
    experience = '1-3 years in trade';
  } else if (score <= 7) {
    level = 'advanced';
    experience = '3-7 years, journeyman level';
  } else {
    level = 'expert';
    experience = '7+ years, master/engineer level';
  }

  return {
    level,
    score: Math.round(score),
    factors,
    prerequisites,
    estimated_experience: experience,
  };
}

// ============================================================================
// Spaced Repetition for Review
// ============================================================================

export interface ReviewSchedule {
  entry_id: string;
  next_review: Date;
  interval_days: number;
  ease_factor: number;
  repetitions: number;
  last_quality: number; // 0-5 rating from last review
}

/**
 * Calculate next review date using SM-2 algorithm
 */
export function calculateNextReview(
  schedule: ReviewSchedule,
  quality: number // 0-5 rating of recall
): ReviewSchedule {
  let { ease_factor, interval_days, repetitions } = schedule;

  if (quality < 3) {
    // Failed review - reset
    repetitions = 0;
    interval_days = 1;
  } else {
    // Successful review
    if (repetitions === 0) {
      interval_days = 1;
    } else if (repetitions === 1) {
      interval_days = 6;
    } else {
      interval_days = Math.round(interval_days * ease_factor);
    }
    repetitions += 1;
  }

  // Update ease factor
  ease_factor = Math.max(1.3, ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)));

  const next_review = new Date();
  next_review.setDate(next_review.getDate() + interval_days);

  return {
    entry_id: schedule.entry_id,
    next_review,
    interval_days,
    ease_factor,
    repetitions,
    last_quality: quality,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function getSystemPromptForCategory(category: Category): string {
  const mapping: Record<string, keyof typeof SYSTEM_PROMPTS> = {
    electrical_systems: 'electrical_expert',
    hvac_systems: 'hvac_technician',
    plumbing_systems: 'mep_engineer',
    solar_power: 'solar_specialist',
    estimating: 'estimator',
    safety_compliance: 'safety_officer',
    building_codes: 'code_inspector',
  };

  const key = mapping[category] || 'trading_expert';
  return SYSTEM_PROMPTS[key] || '';
}

function generateBasicQualityReport(pair: TrainingPair): QualityReport {
  const answerLength = pair.output?.length || 0;
  const hasCodeRef = /NEC|NFPA|IBC|IRC/i.test(pair.output || '');
  const hasSpecifics = /\d+\s*(ft|inch|AWG|amp|volt|BTU|CFM|GPM|PSI)/i.test(pair.output || '');

  let score = 50;
  if (answerLength > 200) score += 15;
  if (answerLength > 400) score += 10;
  if (hasCodeRef) score += 10;
  if (hasSpecifics) score += 10;

  return {
    overall_score: Math.min(90, score),
    dimensions: {
      accuracy: { score: 70, feedback: 'Unable to verify without AI - manual review recommended' },
      completeness: {
        score: answerLength > 200 ? 80 : 60,
        feedback: answerLength > 200 ? 'Good length' : 'Could be more detailed',
      },
      clarity: { score: 75, feedback: 'Appears readable' },
      practicality: {
        score: hasSpecifics ? 85 : 65,
        feedback: hasSpecifics ? 'Contains specific values' : 'Could use more specifics',
      },
      code_compliance: {
        score: hasCodeRef ? 85 : 50,
        feedback: hasCodeRef ? 'References codes' : 'Missing code references',
      },
    },
    suggestions: [
      ...(answerLength < 200 ? ['Add more detail to the answer'] : []),
      ...(!hasCodeRef ? ['Add relevant code references (NEC, NFPA, etc.)'] : []),
      ...(!hasSpecifics ? ['Include specific measurements or values'] : []),
    ],
    ready_for_training: score >= 70,
  };
}

function generateTemplateQA(
  content: string,
  category: Category,
  count: number
): EnhancedTrainingPair[] {
  const categoryDef = CATEGORY_DEFINITIONS[category];
  const pairs: EnhancedTrainingPair[] = [];

  // Extract potential topics from content
  const detectedTopics = categoryDef.keywords.filter((kw) =>
    content.toLowerCase().includes(kw.toLowerCase())
  );

  const topics = detectedTopics.length > 0 ? detectedTopics : ['this concept'];

  for (let i = 0; i < Math.min(count, topics.length); i++) {
    const topic = topics[i];

    pairs.push({
      id: `template_${Date.now()}_${i}`,
      type: 'qa',
      instruction: `Explain ${topic} and how it applies in practice.`,
      input: '',
      output: extractRelevantContent(content, topic),
      system_prompt: getSystemPromptForCategory(category),
      quality_score: 60,
      difficulty: 'intermediate',
      prerequisites: [],
      related_topics: topics.filter((t) => t !== topic).slice(0, 3),
      code_references: [],
    });
  }

  return pairs;
}

function extractRelevantContent(content: string, topic: string): string {
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
  const relevant = sentences.filter((s) => s.toLowerCase().includes(topic.toLowerCase()));

  if (relevant.length > 0) {
    return relevant.slice(0, 4).join('. ').trim() + '.';
  }

  return sentences.slice(0, 4).join('. ').trim() + '.';
}
