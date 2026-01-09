/**
 * knowledge.ts - Knowledge Base Tools (10 tools)
 *
 * Tools for managing the knowledge catalogue and training data generation.
 * Supports OCR processing, cataloguing, search, and training pair export.
 *
 * TOOLS:
 *   1. process_book_image - OCR a book/document image
 *   2. batch_process_images - Process multiple images
 *   3. search_knowledge - Full-text search
 *   4. list_knowledge_by_category - Browse by category
 *   5. get_knowledge_entry - Get specific entry
 *   6. generate_training_pairs - Create training pairs
 *   7. export_training_data - Export to Alpaca/ShareGPT/ChatML
 *   8. knowledge_stats - Database statistics
 *   9. check_ocr_backends - Check available OCR backends
 *   10. list_categories - List all categories
 */

import { ToolDefinition, ToolHandler, ToolModule } from './types.js';

// Category enum for type safety
const CATEGORIES = [
  'candlestick_patterns',
  'chart_patterns',
  'technical_indicators',
  'risk_management',
  'trading_psychology',
  'market_structure',
  'options_strategies',
  'fundamental_analysis',
  'order_flow',
  'volume_analysis',
  'general',
] as const;

/**
 * Tool definitions
 */
export const KNOWLEDGE_TOOLS: ToolDefinition[] = [
  {
    name: 'process_book_image',
    description:
      'OCR a book/document image and catalogue the extracted text into the knowledge base',
    inputSchema: {
      type: 'object',
      properties: {
        image_path: {
          type: 'string',
          description: 'Path to the image file (jpg, png, etc.)',
        },
        book_title: {
          type: 'string',
          description: 'Title of the book (optional)',
        },
        author: {
          type: 'string',
          description: 'Author of the book (optional)',
        },
        chapter: {
          type: 'string',
          description: 'Chapter name or number (optional)',
        },
        page_numbers: {
          type: 'string',
          description: 'Page number(s) (optional)',
        },
        category: {
          type: 'string',
          description: 'Content category for classification',
          enum: [...CATEGORIES],
        },
        tags: {
          type: 'array',
          description: 'Tags for this content (optional)',
        },
        ocr_backend: {
          type: 'string',
          description: 'OCR backend to use (auto, tesseract, easyocr, claude)',
          enum: ['auto', 'tesseract', 'easyocr', 'claude'],
        },
      },
      required: ['image_path'],
    },
  },
  {
    name: 'batch_process_images',
    description: 'Process multiple book images at once and catalogue them',
    inputSchema: {
      type: 'object',
      properties: {
        image_paths: {
          type: 'array',
          description: 'Array of image file paths',
        },
        book_title: {
          type: 'string',
          description: 'Title of the book (applies to all)',
        },
        author: {
          type: 'string',
          description: 'Author of the book (applies to all)',
        },
        category: {
          type: 'string',
          description: 'Content category',
          enum: [...CATEGORIES],
        },
        ocr_backend: {
          type: 'string',
          description: 'OCR backend to use',
          enum: ['auto', 'tesseract', 'easyocr', 'claude'],
        },
      },
      required: ['image_paths'],
    },
  },
  {
    name: 'search_knowledge',
    description: 'Search the knowledge base using full-text search',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query',
        },
        limit: {
          type: 'number',
          description: 'Maximum results to return (default: 20)',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'list_knowledge_by_category',
    description: 'List knowledge entries by category',
    inputSchema: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          description: 'Category to filter by',
          enum: [...CATEGORIES],
        },
        limit: {
          type: 'number',
          description: 'Maximum results to return (default: 50)',
        },
      },
      required: ['category'],
    },
  },
  {
    name: 'get_knowledge_entry',
    description: 'Get a specific knowledge entry by ID',
    inputSchema: {
      type: 'object',
      properties: {
        entry_id: {
          type: 'string',
          description: 'The knowledge entry ID',
        },
      },
      required: ['entry_id'],
    },
  },
  {
    name: 'generate_training_pairs',
    description: 'Generate training data pairs from knowledge base entries',
    inputSchema: {
      type: 'object',
      properties: {
        entry_id: {
          type: 'string',
          description:
            'Generate pairs from specific entry (optional - if not provided, generates from all)',
        },
        min_quality_score: {
          type: 'number',
          description: 'Minimum quality score for entries (0-100, default: 30)',
        },
        pairs_per_entry: {
          type: 'number',
          description: 'Number of pairs to generate per entry (default: 3)',
        },
        include_system_prompt: {
          type: 'boolean',
          description: 'Include system prompts in pairs (default: true)',
        },
        generate_synthetic: {
          type: 'boolean',
          description: 'Use AI to generate additional synthetic pairs (requires API key)',
        },
      },
      required: [],
    },
  },
  {
    name: 'export_training_data',
    description: 'Export all training pairs to a file for fine-tuning',
    inputSchema: {
      type: 'object',
      properties: {
        output_path: {
          type: 'string',
          description: 'Path to save the training data file',
        },
        format: {
          type: 'string',
          description: 'Output format (alpaca, sharegpt, chatml)',
          enum: ['alpaca', 'sharegpt', 'chatml'],
        },
        min_quality_score: {
          type: 'number',
          description: 'Minimum quality score to include (default: 0)',
        },
      },
      required: ['output_path', 'format'],
    },
  },
  {
    name: 'knowledge_stats',
    description: 'Get statistics about the knowledge base',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'check_ocr_backends',
    description: 'Check which OCR backends are available on the system',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'list_categories',
    description: 'List all available knowledge categories with descriptions',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
];

/**
 * Helper to get error message from unknown error
 */
function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Handler implementations
 */
export const KNOWLEDGE_HANDLERS: Record<string, ToolHandler> = {
  process_book_image: async (args, ctx) => {
    const {
      image_path,
      book_title,
      author,
      chapter,
      page_numbers,
      category,
      tags = [],
      ocr_backend = 'auto',
    } = args as {
      image_path: string;
      book_title?: string;
      author?: string;
      chapter?: string;
      page_numbers?: string;
      category?: string;
      tags?: string[];
      ocr_backend?: 'auto' | 'tesseract' | 'easyocr' | 'claude';
    };

    ctx.logger.info(`Processing image: ${image_path}`);

    try {
      // Process image with OCR
      const ocrResult = (await ctx.processImage(image_path, {
        backend: ocr_backend,
        enhance_image: true,
      })) as {
        raw_text: string;
        cleaned_text: string;
        confidence: number;
        backend_used: string;
        processing_time_ms: number;
      };

      const finalCategory = category || 'general';

      // Add to knowledge base
      const entryId = await ctx.knowledgeDb.addEntry({
        source: {
          type: 'book',
          book_title,
          author,
          chapter,
          page_numbers,
          image_path,
          capture_date: new Date().toISOString(),
        },
        raw_text: ocrResult.raw_text,
        cleaned_text: ocrResult.cleaned_text,
        category: finalCategory,
        topics: [],
        tags,
        quality_score: Math.round(ocrResult.confidence),
        ocr_confidence: ocrResult.confidence,
        manually_reviewed: false,
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                entry_id: entryId,
                ocr_backend: ocrResult.backend_used,
                ocr_confidence: ocrResult.confidence,
                processing_time_ms: ocrResult.processing_time_ms,
                category: finalCategory,
                text_preview:
                  ocrResult.cleaned_text.substring(0, 300) +
                  (ocrResult.cleaned_text.length > 300 ? '...' : ''),
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error processing book image: ${getErrorMessage(error)}`);
    }
  },

  batch_process_images: async (args, ctx) => {
    const {
      image_paths,
      book_title,
      author,
      category,
      ocr_backend = 'auto',
    } = args as {
      image_paths: string[];
      book_title?: string;
      author?: string;
      category?: string;
      ocr_backend?: 'auto' | 'tesseract' | 'easyocr' | 'claude';
    };

    ctx.logger.info(`Batch processing ${image_paths.length} images`);

    const results: Array<{ path: string; entry_id?: string; error?: string }> = [];

    for (const imagePath of image_paths) {
      try {
        const ocrResult = (await ctx.processImage(imagePath, {
          backend: ocr_backend,
          enhance_image: true,
        })) as {
          raw_text: string;
          cleaned_text: string;
          confidence: number;
        };

        const finalCategory = category || 'general';

        const entryId = await ctx.knowledgeDb.addEntry({
          source: {
            type: 'book',
            book_title,
            author,
            image_path: imagePath,
            capture_date: new Date().toISOString(),
          },
          raw_text: ocrResult.raw_text,
          cleaned_text: ocrResult.cleaned_text,
          category: finalCategory,
          topics: [],
          tags: [],
          quality_score: Math.round(ocrResult.confidence),
          ocr_confidence: ocrResult.confidence,
          manually_reviewed: false,
        });

        results.push({ path: imagePath, entry_id: entryId as string });
      } catch (error: unknown) {
        results.push({ path: imagePath, error: getErrorMessage(error) });
      }
    }

    const successful = results.filter((r) => r.entry_id).length;
    const failed = results.filter((r) => r.error).length;

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              success: true,
              total: image_paths.length,
              successful,
              failed,
              results,
            },
            null,
            2
          ),
        },
      ],
    };
  },

  search_knowledge: async (args, ctx) => {
    const { query, limit = 20 } = args as { query: string; limit?: number };
    ctx.logger.info(`Searching knowledge: ${query}`);

    try {
      const entries = await ctx.knowledgeDb.searchEntries(query, limit);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                query,
                count: (entries as unknown[]).length,
                entries,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error searching knowledge base: ${getErrorMessage(error)}`);
    }
  },

  list_knowledge_by_category: async (args, ctx) => {
    const { category, limit = 50 } = args as { category: string; limit?: number };
    ctx.logger.info(`Listing knowledge by category: ${category}`);

    try {
      const entries = await ctx.knowledgeDb.listByCategory(category, limit);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                category,
                count: (entries as unknown[]).length,
                entries,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error listing knowledge entries: ${getErrorMessage(error)}`);
    }
  },

  get_knowledge_entry: async (args, ctx) => {
    const { entry_id } = args as { entry_id: string };
    ctx.logger.info(`Getting knowledge entry: ${entry_id}`);

    try {
      const entry = await ctx.knowledgeDb.getEntry(entry_id);
      if (!entry) {
        return {
          content: [{ type: 'text', text: `Knowledge entry not found: ${entry_id}` }],
          isError: true,
        };
      }
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                entry,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error getting knowledge entry: ${getErrorMessage(error)}`);
    }
  },

  generate_training_pairs: async (args, ctx) => {
    const {
      entry_id,
      min_quality_score = 30,
      pairs_per_entry = 3,
      include_system_prompt = true,
    } = args as {
      entry_id?: string;
      min_quality_score?: number;
      pairs_per_entry?: number;
      include_system_prompt?: boolean;
    };

    ctx.logger.info('Generating training pairs');

    try {
      if (entry_id) {
        // Generate for specific entry
        const entry = (await ctx.knowledgeDb.getEntry(entry_id)) as {
          cleaned_text: string;
          category: string;
        } | null;
        if (!entry) {
          throw new Error(`Entry not found: ${entry_id}`);
        }

        const pairs = await ctx.generateTrainingPairs(entry.cleaned_text, entry.category, {
          min_quality_score,
          pairs_per_entry,
          include_system_prompt,
        });

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(
                {
                  success: true,
                  entry_id,
                  pairs_generated: (pairs as unknown[]).length,
                  pairs,
                },
                null,
                2
              ),
            },
          ],
        };
      } else {
        // Generate for all entries
        const allPairs = await ctx.knowledgeDb.getAllTrainingPairs(min_quality_score);

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(
                {
                  success: true,
                  total_pairs: (allPairs as unknown[]).length,
                  message: 'Training pairs retrieved from all entries',
                },
                null,
                2
              ),
            },
          ],
        };
      }
    } catch (error: unknown) {
      throw new Error(`Error generating training pairs: ${getErrorMessage(error)}`);
    }
  },

  export_training_data: async (args, ctx) => {
    const {
      output_path,
      format,
      min_quality_score = 0,
    } = args as {
      output_path: string;
      format: 'alpaca' | 'sharegpt' | 'chatml';
      min_quality_score?: number;
    };

    ctx.logger.info(`Exporting training data to ${output_path} as ${format}`);

    try {
      const count = await ctx.knowledgeDb.exportTrainingData(
        output_path,
        format,
        min_quality_score
      );
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                format,
                output_path,
                pairs_exported: count,
                message: `Training data exported to ${output_path}. Ready for fine-tuning!`,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error exporting training data: ${getErrorMessage(error)}`);
    }
  },

  knowledge_stats: async (_args, ctx) => {
    ctx.logger.info('Getting knowledge stats');

    try {
      const stats = await ctx.knowledgeDb.getStats();
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                stats,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error getting knowledge stats: ${getErrorMessage(error)}`);
    }
  },

  check_ocr_backends: async (_args, ctx) => {
    ctx.logger.info('Checking OCR backends');

    try {
      const backends = (await ctx.checkOCRBackends()) as Record<string, boolean>;
      const available = Object.entries(backends)
        .filter(([, v]) => v)
        .map(([k]) => k);

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                backends,
                available,
                recommendation: backends.tesseract
                  ? 'tesseract (fast, good for clear text)'
                  : backends.easyocr
                    ? 'easyocr (slower, better accuracy)'
                    : backends.claude
                      ? 'claude (best for charts/diagrams, requires API key)'
                      : 'No OCR backend available. Install pytesseract or easyocr.',
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error checking OCR backends: ${getErrorMessage(error)}`);
    }
  },

  list_categories: async (_args, ctx) => {
    ctx.logger.info('Listing categories');

    const categories = Object.entries(ctx.categoryDefinitions).map(([key, value]) => ({
      id: key,
      name: value.name,
      description: value.description,
    }));

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
              success: true,
              count: categories.length,
              categories,
            },
            null,
            2
          ),
        },
      ],
    };
  },
};

/**
 * Module export
 */
export const knowledgeModule: ToolModule = {
  tools: KNOWLEDGE_TOOLS,
  handlers: KNOWLEDGE_HANDLERS,
};

export default knowledgeModule;
