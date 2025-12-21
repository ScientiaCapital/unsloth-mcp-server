/**
 * Knowledge Base Module
 *
 * Complete pipeline for:
 * 1. OCR processing of book photos
 * 2. Knowledge cataloguing and organization
 * 3. Training data generation for fine-tuning
 */

// Schema - types
export type {
  KnowledgeEntry,
  SourceInfo,
  Category,
  TrainingPair,
  AlpacaFormat,
  ShareGPTFormat,
  ChatMLFormat,
} from './schema.js';

// Schema - values
export {
  CATEGORY_DEFINITIONS,
  DB_SCHEMA,
  toAlpacaFormat,
  toShareGPTFormat,
  toChatMLFormat,
} from './schema.js';

// Database operations
export { KnowledgeDatabase, knowledgeDb } from './database.js';

// OCR processing - types
export type { OCRResult, OCROptions } from './ocr.js';

// OCR processing - values
export {
  checkOCRBackends,
  processImage,
  processImageBatch,
  classifyContent,
  cleanText,
} from './ocr.js';

// Training data generation - types
export type { GeneratorOptions } from './training-generator.js';

// Training data generation - values
export {
  SYSTEM_PROMPTS,
  generateTrainingPairs,
  generateFromDatabase,
  generateSyntheticPairs,
} from './training-generator.js';

// Dataset builder - types
export type {
  DatasetEntry,
  DataSource,
  ContentBlock,
  EntryMetadata,
  QualityMetrics,
} from './dataset-builder.js';

// Dataset builder - values
export {
  DatasetBuilder,
  PUBLIC_DATA_SOURCES,
  TRADE_TEMPLATES,
  CONTRIBUTION_GUIDE,
} from './dataset-builder.js';

// AI Enhancement - types
export type {
  QualityReport,
  EnhancedTrainingPair,
  DifficultyAssessment,
  ReviewSchedule,
} from './ai-enhancer.js';

// AI Enhancement - values
export {
  generateExpertQA,
  evaluateQuality,
  improveTrainingPair,
  assessDifficulty,
  calculateNextReview,
} from './ai-enhancer.js';

// Hugging Face Export - types
export type { ExportOptions } from './huggingface-export.js';

// Hugging Face Export - values
export {
  generateDatasetCard,
  exportForHuggingFace,
  uploadToHuggingFace,
} from './huggingface-export.js';
