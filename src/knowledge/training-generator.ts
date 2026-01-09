/**
 * Training Data Generator
 *
 * Transforms knowledge entries into instruction-tuning training pairs
 * Supports multiple generation strategies for diverse training data
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import {
  KnowledgeEntry,
  TrainingPair,
  Category,
  CATEGORY_DEFINITIONS,
  AlpacaFormat,
} from './schema.js';
import { KnowledgeDatabase } from './database.js';
import logger from '../utils/logger.js';

const execPromise = promisify(exec);

// Helper to extract error message from unknown error type
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

// ============================================================================
// System Prompts for Different Personas
// ============================================================================

export const SYSTEM_PROMPTS = {
  // Trading & Finance
  trading_expert: `You are an expert trading educator with decades of experience in technical analysis, chart patterns, and market psychology. You explain concepts clearly with practical examples.`,

  risk_manager: `You are a professional risk manager specializing in position sizing, stop-loss strategies, and capital preservation. You prioritize protecting capital above all else.`,

  chart_analyst: `You are a technical chart analyst who specializes in reading candlestick patterns and price action. You can identify patterns and explain their significance.`,

  options_specialist: `You are an options trading specialist who understands Greeks, volatility, and complex options strategies. You explain options concepts in practical terms.`,

  trading_coach: `You are a trading coach who helps traders develop discipline, manage emotions, and build consistent trading habits. You focus on psychology and process.`,

  // Sales & Business
  sales_expert: `You are a world-class sales trainer with experience closing high-ticket deals. You teach proven techniques for prospecting, objection handling, and closing with integrity.`,

  negotiation_expert: `You are a master negotiator who has trained executives and dealmakers. You teach strategic negotiation frameworks and tactics for win-win outcomes.`,

  business_strategist: `You are a business strategist who has advised Fortune 500 companies and startups. You think in terms of competitive advantage, market positioning, and sustainable growth.`,

  marketing_expert: `You are a marketing expert who understands both traditional and digital marketing. You focus on customer acquisition, brand building, and growth strategies.`,

  leadership_coach: `You are an executive leadership coach who has developed leaders at top organizations. You teach authentic leadership, vision-setting, and team empowerment.`,

  // Personal Development
  success_coach: `You are a peak performance coach who has studied the habits of ultra-successful people. You teach practical principles for achievement, productivity, and personal mastery.`,

  mindset_coach: `You are a mindset and performance coach specializing in belief systems, mental frameworks, and psychological transformation. You help people overcome limiting beliefs.`,

  productivity_expert: `You are a productivity and time management expert who teaches systems for deep work, focus, and getting more done with less effort.`,

  habit_expert: `You are a behavioral psychologist specializing in habit formation and behavior change. You explain the science of habits and practical strategies for lasting change.`,

  // Wealth & Investing
  wealth_advisor: `You are a wealth strategist who teaches long-term wealth building principles. You focus on financial literacy, asset allocation, and creating multiple income streams.`,

  real_estate_expert: `You are a real estate investing expert with experience in residential, commercial, and multi-family properties. You teach practical strategies for building wealth through real estate.`,

  // Communication
  communication_coach: `You are a communication expert who teaches effective speaking, listening, and interpersonal skills. You help people become more persuasive and influential communicators.`,

  public_speaking_coach: `You are a professional speaking coach who has trained TEDx speakers and executives. You teach presentation skills, storytelling, and confident delivery.`,

  // Energy & MEP
  energy_engineer: `You are a licensed professional engineer specializing in energy systems, power distribution, and renewable energy. You explain complex electrical and energy concepts in practical terms for contractors and installers.`,

  solar_specialist: `You are a NABCEP-certified solar installer and designer with extensive experience in residential and commercial PV systems. You teach proper installation techniques, system sizing, and code compliance.`,

  electrical_expert: `You are a master electrician and electrical contractor with decades of experience. You explain NEC code requirements, proper wiring methods, and troubleshooting techniques.`,

  hvac_technician: `You are an HVAC master technician and instructor who teaches system design, installation, and service. You explain heating, cooling, and refrigeration concepts clearly.`,

  mep_engineer: `You are a licensed MEP (Mechanical, Electrical, Plumbing) engineer who designs building systems. You explain engineering calculations, code compliance, and best practices.`,

  // Trades & Construction
  estimator: `You are a senior construction estimator who has bid millions in projects. You teach accurate takeoffs, labor estimation, and competitive bidding strategies.`,

  project_manager: `You are a PMP-certified construction project manager with experience on large commercial projects. You teach scheduling, coordination, and project controls.`,

  code_inspector: `You are a certified building inspector and plans examiner. You explain code requirements, inspection processes, and common violations to avoid.`,

  safety_officer: `You are an OSHA-certified safety professional and construction safety trainer. You teach jobsite safety, hazard recognition, and compliance requirements.`,

  contractor_coach: `You are a successful contractor who has built and scaled multiple construction businesses. You teach business operations, client relations, and profitable bidding.`,
};

// ============================================================================
// Question Templates by Category
// ============================================================================

const QUESTION_TEMPLATES: Record<Category, string[]> = {
  candlestick_patterns: [
    'What is a {topic} candlestick pattern and how do I identify it?',
    'How should I trade when I see a {topic} pattern?',
    'What does a {topic} pattern tell us about market sentiment?',
    'Can you explain the psychology behind the {topic} pattern?',
    'What confirmation signals should I look for with a {topic} pattern?',
    'What is the success rate of the {topic} pattern?',
    'How does volume affect the reliability of a {topic} pattern?',
  ],
  chart_patterns: [
    'How do I identify a {topic} chart pattern?',
    'What is the typical price target for a {topic} pattern?',
    'How should I set my stop loss when trading a {topic}?',
    'What timeframes work best for trading {topic} patterns?',
    'What are the key characteristics of a valid {topic} pattern?',
    'How do I distinguish a true {topic} from a false one?',
  ],
  technical_indicators: [
    'How do I use {topic} in my trading?',
    'What settings should I use for {topic}?',
    'How do I interpret {topic} signals?',
    'What are the limitations of {topic}?',
    'How does {topic} compare to other indicators?',
    'Can you explain {topic} divergence?',
  ],
  risk_management: [
    'How should I calculate my position size?',
    'What is a good risk-reward ratio to aim for?',
    'How do I set an effective stop loss?',
    'What percentage of my capital should I risk per trade?',
    'How do I manage drawdowns effectively?',
    'When should I scale into or out of a position?',
  ],
  trading_psychology: [
    'How do I deal with {topic} in trading?',
    'What causes traders to make emotional decisions?',
    'How can I develop more trading discipline?',
    'What should I do after a losing streak?',
    'How do I avoid revenge trading?',
    'What mindset habits lead to consistent trading?',
  ],
  market_structure: [
    'How do I identify {topic} in the market?',
    'What defines a valid {topic} level?',
    'How do I know when a {topic} will hold?',
    'What causes {topic} to form?',
    'How should I trade around {topic}?',
  ],
  options_strategies: [
    'How do I set up a {topic} options trade?',
    'What are the risks of a {topic} strategy?',
    'When should I use a {topic} strategy?',
    'How does implied volatility affect {topic}?',
    'What is the max profit and loss for a {topic}?',
  ],
  fundamental_analysis: [
    'What metrics should I look at for {topic}?',
    'How do I evaluate a company using {topic}?',
    'What does {topic} tell us about a stock?',
    'How do I combine {topic} with technical analysis?',
  ],
  order_flow: [
    'How do I read {topic}?',
    'What does {topic} tell us about institutional activity?',
    'How can I use {topic} to improve my entries?',
    'What tools do I need for {topic} analysis?',
  ],
  volume_analysis: [
    'How do I interpret {topic}?',
    'What does high {topic} indicate?',
    'How do I use {topic} to confirm trades?',
    'What is the relationship between {topic} and price?',
  ],

  // Sales & Persuasion
  sales_techniques: [
    'What is the best way to handle {topic}?',
    'How do I improve my {topic} skills?',
    'What makes {topic} effective in sales?',
    'How do top salespeople approach {topic}?',
  ],
  negotiation: [
    'How do I use {topic} in negotiation?',
    'What is the {topic} technique?',
    'When should I apply {topic}?',
    'How do I counter {topic}?',
  ],
  persuasion: [
    'How does {topic} influence decisions?',
    'What is the psychology behind {topic}?',
    'How can I ethically use {topic}?',
    'What are examples of {topic} in action?',
  ],
  closing: [
    'How do I execute a {topic}?',
    'When is {topic} most effective?',
    'What signals indicate I should use {topic}?',
    'How do I practice {topic}?',
  ],

  // Business & Entrepreneurship
  business_strategy: [
    'What is {topic} and why does it matter?',
    'How do successful companies use {topic}?',
    'How do I develop a {topic} for my business?',
    'What are the key elements of {topic}?',
  ],
  marketing: [
    'How do I implement {topic} in marketing?',
    'What makes {topic} effective?',
    'How do I measure {topic} success?',
    'What are best practices for {topic}?',
  ],
  leadership: [
    'What makes {topic} important for leaders?',
    'How do great leaders practice {topic}?',
    'How can I develop {topic} skills?',
    'What are the benefits of {topic}?',
  ],
  management: [
    'How do I effectively use {topic}?',
    'What are best practices for {topic}?',
    'How does {topic} improve team performance?',
    'What mistakes should I avoid with {topic}?',
  ],
  startups: [
    'Why is {topic} important for startups?',
    'How do I achieve {topic}?',
    'What are signs of {topic}?',
    'How do successful founders approach {topic}?',
  ],

  // Self-Help & Personal Development
  mindset: [
    'What is {topic} and how does it work?',
    'How do I develop a {topic}?',
    'What are the benefits of {topic}?',
    'How do I shift from my current mindset to {topic}?',
  ],
  habits: [
    'How do I build {topic}?',
    'What is the science behind {topic}?',
    'How long does it take to form {topic}?',
    'What are strategies for maintaining {topic}?',
  ],
  productivity: [
    'How does {topic} improve productivity?',
    'What is the {topic} method?',
    'How do I implement {topic} in my routine?',
    'What are common mistakes with {topic}?',
  ],
  motivation: [
    'How do I find {topic}?',
    'What drives {topic}?',
    'How do I maintain {topic} over time?',
    'What is the difference between {topic} and discipline?',
  ],
  success_principles: [
    'What is {topic} and why does it matter?',
    'How do successful people apply {topic}?',
    'How do I implement {topic} in my life?',
    'What examples demonstrate {topic}?',
  ],

  // Wealth & Investing
  wealth_building: [
    'How does {topic} contribute to wealth?',
    'What is the role of {topic} in financial success?',
    'How do I start with {topic}?',
    'What mistakes should I avoid with {topic}?',
  ],
  real_estate: [
    'How do I evaluate {topic} in real estate?',
    'What should I know about {topic}?',
    'How do successful investors approach {topic}?',
    'What are the risks of {topic}?',
  ],
  passive_income: [
    'How do I create {topic}?',
    'What are the best sources of {topic}?',
    'How long does it take to build {topic}?',
    'What is needed to maintain {topic}?',
  ],

  // Communication & Influence
  communication: [
    'How do I improve my {topic} skills?',
    'Why is {topic} important?',
    'What are techniques for better {topic}?',
    'How do I practice {topic}?',
  ],
  public_speaking: [
    'How do I use {topic} in presentations?',
    'What makes {topic} effective?',
    'How do I overcome fear of {topic}?',
    'What are tips for better {topic}?',
  ],
  networking: [
    'How do I approach {topic}?',
    'What makes {topic} successful?',
    'How do I follow up after {topic}?',
    'What are best practices for {topic}?',
  ],

  // Energy & Utilities
  energy_systems: [
    'How does {topic} work in power systems?',
    'What should I know about {topic}?',
    'How do I calculate {topic}?',
    'What are the requirements for {topic}?',
  ],
  solar_power: [
    'How do I size {topic} for a solar system?',
    'What is the proper way to install {topic}?',
    'What code requirements apply to {topic}?',
    'How do I troubleshoot {topic} issues?',
  ],
  electrical_systems: [
    'What does the NEC say about {topic}?',
    'How do I properly install {topic}?',
    'What are the calculations for {topic}?',
    'What are common mistakes with {topic}?',
  ],
  hvac_systems: [
    'How do I size {topic}?',
    'What is the proper procedure for {topic}?',
    'How do I troubleshoot {topic}?',
    'What are the efficiency considerations for {topic}?',
  ],

  // MEP Engineering
  mechanical_engineering: [
    'How do I calculate {topic}?',
    'What are the design considerations for {topic}?',
    'How do I select the right {topic}?',
    'What codes apply to {topic}?',
  ],
  plumbing_systems: [
    'What are the code requirements for {topic}?',
    'How do I properly install {topic}?',
    'What sizing calculations apply to {topic}?',
    'How do I troubleshoot {topic}?',
  ],
  fire_protection: [
    'What does NFPA require for {topic}?',
    'How do I design {topic} systems?',
    'What are the inspection requirements for {topic}?',
    'How do I calculate {topic}?',
  ],
  building_automation: [
    'How do I program {topic}?',
    'What is the sequence of operations for {topic}?',
    'How do I integrate {topic}?',
    'What are best practices for {topic}?',
  ],

  // Trades & Construction
  estimating: [
    'How do I estimate {topic}?',
    'What is the standard unit cost for {topic}?',
    'How do I do a takeoff for {topic}?',
    'What labor factors apply to {topic}?',
  ],
  project_management_construction: [
    'How do I manage {topic} on a project?',
    'What is the process for {topic}?',
    'How do I handle {topic} issues?',
    'What documentation is needed for {topic}?',
  ],
  building_codes: [
    'What does the code require for {topic}?',
    'How do I get a permit for {topic}?',
    'What are common {topic} violations?',
    'How do I pass inspection for {topic}?',
  ],
  safety_compliance: [
    'What are OSHA requirements for {topic}?',
    'How do I train workers on {topic}?',
    'What PPE is required for {topic}?',
    'How do I document {topic}?',
  ],
  blueprints_drawings: [
    'How do I read {topic} on blueprints?',
    'What does this {topic} symbol mean?',
    'How do I interpret {topic} dimensions?',
    'What information is shown on {topic}?',
  ],

  // Contractor Business
  contractor_business: [
    'What are the requirements for {topic}?',
    'How do I set up {topic} for my business?',
    'What should I know about {topic}?',
    'How do successful contractors handle {topic}?',
  ],
  bidding_proposals: [
    'How do I write a {topic}?',
    'What should be included in {topic}?',
    'How do I price {topic} competitively?',
    'What are common mistakes with {topic}?',
  ],
  client_management: [
    'How do I handle {topic} with clients?',
    'What is the best approach to {topic}?',
    'How do I communicate about {topic}?',
    'What are best practices for {topic}?',
  ],

  general: [
    'Can you explain this concept?',
    'What should I know about this topic?',
    'How does this apply to real life?',
    'What are the key takeaways?',
  ],
};

// ============================================================================
// Training Pair Generation
// ============================================================================

export interface GeneratorOptions {
  min_quality_score?: number;
  pairs_per_entry?: number;
  include_system_prompt?: boolean;
  system_prompt_type?: keyof typeof SYSTEM_PROMPTS;
  generate_qa?: boolean;
  generate_instruction?: boolean;
  generate_conversation?: boolean;
}

/**
 * Generate training pairs from a knowledge entry
 */
export function generateTrainingPairs(
  entry: KnowledgeEntry,
  options: GeneratorOptions = {}
): TrainingPair[] {
  const pairs: TrainingPair[] = [];
  const {
    pairs_per_entry = 3,
    include_system_prompt = true,
    system_prompt_type = 'trading_expert',
    generate_qa = true,
    generate_instruction = true,
    generate_conversation = true,
  } = options;

  const systemPrompt = include_system_prompt ? SYSTEM_PROMPTS[system_prompt_type] : undefined;
  const content = entry.cleaned_text;
  const category = entry.category;
  const topics = entry.topics;

  // Generate Q&A pairs
  if (generate_qa && topics.length > 0) {
    const templates = QUESTION_TEMPLATES[category] || QUESTION_TEMPLATES.general;

    for (let i = 0; i < Math.min(pairs_per_entry, templates.length); i++) {
      const topic = topics[i % topics.length];
      const question = templates[i].replace('{topic}', topic);

      pairs.push({
        id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        type: 'qa',
        instruction: question,
        input: '',
        output: generateAnswer(content, question, topic),
        system_prompt: systemPrompt,
        quality_score: calculateQualityScore(content, question),
      });
    }
  }

  // Generate instruction-following pairs
  if (generate_instruction) {
    // Summarization task
    pairs.push({
      id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      type: 'instruction',
      instruction: 'Summarize the key trading concepts from the following text.',
      input: content,
      output: generateSummary(content),
      system_prompt: systemPrompt,
      quality_score: calculateQualityScore(content, 'summarize'),
    });

    // Key points extraction
    if (content.length > 200) {
      pairs.push({
        id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        type: 'instruction',
        instruction: 'Extract the main actionable trading insights from this content.',
        input: content,
        output: extractKeyPoints(content),
        system_prompt: systemPrompt,
        quality_score: calculateQualityScore(content, 'extract'),
      });
    }
  }

  // Generate conversation-style pairs
  if (generate_conversation && topics.length > 0) {
    const topic = topics[0];
    const categoryDef = CATEGORY_DEFINITIONS[category];

    pairs.push({
      id: `gen_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      type: 'conversation',
      instruction: `I'm trying to understand ${topic}. Can you explain it?`,
      input: '',
      output: generateExplanation(content, topic, categoryDef.description),
      system_prompt: systemPrompt,
      quality_score: calculateQualityScore(content, topic),
    });
  }

  return pairs;
}

/**
 * Generate an answer from content based on question
 */
function generateAnswer(content: string, question: string, topic: string): string {
  // Extract relevant sentences containing the topic
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
  const relevantSentences = sentences.filter((s) => s.toLowerCase().includes(topic.toLowerCase()));

  if (relevantSentences.length > 0) {
    // Construct answer from relevant content
    const answer = relevantSentences.slice(0, 3).join('. ').trim();
    return answer.endsWith('.') ? answer : answer + '.';
  }

  // Fallback: use first few sentences
  return sentences.slice(0, 3).join('. ').trim() + '.';
}

/**
 * Generate a summary of content
 */
function generateSummary(content: string): string {
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);

  if (sentences.length <= 3) {
    return content.trim();
  }

  // Take first, middle, and last meaningful sentences
  const summary = [
    sentences[0],
    sentences[Math.floor(sentences.length / 2)],
    sentences[sentences.length - 1],
  ]
    .join('. ')
    .trim();

  return summary.endsWith('.') ? summary : summary + '.';
}

/**
 * Extract key points from content
 */
function extractKeyPoints(content: string): string {
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);

  // Look for sentences with key indicators
  const keyIndicators = [
    'important',
    'key',
    'remember',
    'always',
    'never',
    'must',
    'should',
    'critical',
    'essential',
    'rule',
    'tip',
    'strategy',
    'pattern',
    'signal',
    'confirm',
  ];

  const keyPoints: string[] = [];

  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    if (keyIndicators.some((indicator) => lower.includes(indicator))) {
      keyPoints.push(`- ${sentence.trim()}`);
    }
  }

  if (keyPoints.length === 0) {
    // Fallback: take first few sentences as bullet points
    return sentences
      .slice(0, 4)
      .map((s) => `- ${s.trim()}`)
      .join('\n');
  }

  return keyPoints.slice(0, 5).join('\n');
}

/**
 * Generate an explanation for a topic
 */
function generateExplanation(content: string, topic: string, categoryDescription: string): string {
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
  const relevantSentences = sentences.filter((s) => s.toLowerCase().includes(topic.toLowerCase()));

  let explanation = '';

  if (relevantSentences.length > 0) {
    explanation = relevantSentences.slice(0, 4).join('. ').trim();
  } else {
    explanation = sentences.slice(0, 4).join('. ').trim();
  }

  // Add context about the category
  const prefix = `${topic} is a concept in ${categoryDescription.toLowerCase()}. `;

  return prefix + explanation + (explanation.endsWith('.') ? '' : '.');
}

/**
 * Calculate quality score for a training pair
 */
function calculateQualityScore(content: string, _query: string): number {
  let score = 50; // Base score

  // Length checks
  if (content.length > 100) score += 10;
  if (content.length > 300) score += 10;
  if (content.length > 500) score += 5;

  // Content quality indicators
  const qualityIndicators = [
    'example',
    'for instance',
    'such as',
    'specifically',
    'in practice',
    'step',
    'first',
    'then',
    'finally',
    'important',
  ];

  for (const indicator of qualityIndicators) {
    if (content.toLowerCase().includes(indicator)) {
      score += 3;
    }
  }

  // Cap at 95
  return Math.min(95, score);
}

// ============================================================================
// Batch Processing
// ============================================================================

/**
 * Generate training data from all knowledge entries in the database
 */
export async function generateFromDatabase(
  db: KnowledgeDatabase,
  options: GeneratorOptions = {}
): Promise<{
  total_entries_processed: number;
  total_pairs_generated: number;
  pairs_by_type: Record<string, number>;
}> {
  const stats = await db.getStats();
  let totalPairs = 0;
  const pairsByType: Record<string, number> = {
    qa: 0,
    instruction: 0,
    conversation: 0,
  };

  // Process each category
  for (const category of Object.keys(CATEGORY_DEFINITIONS) as Category[]) {
    const entries = await db.listByCategory(category);

    for (const entry of entries) {
      // Skip low quality entries
      if (entry.quality_score < (options.min_quality_score || 30)) {
        continue;
      }

      // Get full entry
      const fullEntry = await db.getEntry(entry.id);
      if (!fullEntry) continue;

      // Generate pairs
      const pairs = generateTrainingPairs(fullEntry, options);

      // Store pairs in database
      for (const pair of pairs) {
        await db.addTrainingPair(entry.id, pair);
        totalPairs++;
        pairsByType[pair.type]++;
      }
    }
  }

  logger.info('Training data generation complete', {
    entries: stats.total_entries,
    pairs: totalPairs,
    byType: pairsByType,
  });

  return {
    total_entries_processed: stats.total_entries,
    total_pairs_generated: totalPairs,
    pairs_by_type: pairsByType,
  };
}

/**
 * Generate synthetic Q&A pairs using an LLM
 */
export async function generateSyntheticPairs(
  content: string,
  category: Category,
  numPairs: number = 5
): Promise<TrainingPair[]> {
  // Check if we have an API key for generation
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    logger.warn('No ANTHROPIC_API_KEY found, using template-based generation');
    // Fallback to template-based generation
    return generateTemplatePairs(content, category, numPairs);
  }

  const script = `
import anthropic
import json

try:
    client = anthropic.Anthropic(api_key="${apiKey}")

    prompt = """Based on this trading/finance content, generate ${numPairs} high-quality Q&A pairs for training an AI assistant.

Content:
${content.replace(/"/g, '\\"').replace(/\n/g, '\\n')}

Category: ${category}

Generate diverse questions that cover:
1. Factual understanding
2. Practical application
3. Pattern recognition
4. Risk considerations
5. Real-world scenarios

Return as JSON array with format:
[
  {
    "instruction": "question here",
    "input": "",
    "output": "detailed answer here"
  }
]

Make answers comprehensive (2-4 sentences) and actionable."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text

    # Extract JSON from response
    import re
    json_match = re.search(r'\\[.*\\]', response_text, re.DOTALL)
    if json_match:
        pairs = json.loads(json_match.group())
        print(json.dumps({"success": True, "pairs": pairs}))
    else:
        print(json.dumps({"success": False, "error": "No JSON found in response"}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 60000 });
    const result = JSON.parse(stdout.trim());

    if (!result.success) {
      logger.warn('Synthetic generation failed, using templates', { error: result.error });
      return generateTemplatePairs(content, category, numPairs);
    }

    return result.pairs.map((pair: AlpacaFormat, idx: number) => ({
      id: `syn_${Date.now()}_${idx}`,
      type: 'qa' as const,
      instruction: pair.instruction,
      input: pair.input,
      output: pair.output,
      system_prompt: SYSTEM_PROMPTS.trading_expert,
      quality_score: 80, // Synthetic pairs get high base score
    }));
  } catch (error: unknown) {
    logger.warn('Synthetic generation error, using templates', { error: getErrorMessage(error) });
    return generateTemplatePairs(content, category, numPairs);
  }
}

/**
 * Fallback template-based pair generation
 */
function generateTemplatePairs(
  content: string,
  category: Category,
  numPairs: number
): TrainingPair[] {
  const templates = QUESTION_TEMPLATES[category] || QUESTION_TEMPLATES.general;
  const pairs: TrainingPair[] = [];

  // Extract potential topics from content
  const categoryDef = CATEGORY_DEFINITIONS[category];
  const detectedTopics = categoryDef.keywords.filter((keyword) =>
    content.toLowerCase().includes(keyword.toLowerCase())
  );

  const topics = detectedTopics.length > 0 ? detectedTopics : ['this concept'];

  for (let i = 0; i < Math.min(numPairs, templates.length); i++) {
    const topic = topics[i % topics.length];
    const question = templates[i].replace('{topic}', topic);

    pairs.push({
      id: `tpl_${Date.now()}_${i}`,
      type: 'qa',
      instruction: question,
      input: '',
      output: generateAnswer(content, question, topic),
      system_prompt: SYSTEM_PROMPTS.trading_expert,
      quality_score: 60,
    });
  }

  return pairs;
}
