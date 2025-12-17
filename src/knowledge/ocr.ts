/**
 * OCR Processing for Book Photos
 *
 * Supports multiple OCR backends:
 * 1. pytesseract (local, free) - good for clear text
 * 2. easyocr (local, free) - better for mixed content
 * 3. Claude Vision API (cloud, paid) - best for charts/diagrams
 *
 * The processor will try available backends in order of preference
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import logger from '../utils/logger.js';
import { Category, CATEGORY_DEFINITIONS } from './schema.js';

const execPromise = promisify(exec);

export interface OCRResult {
  raw_text: string;
  cleaned_text: string;
  confidence: number;
  backend_used: string;
  processing_time_ms: number;
  detected_language?: string;
  bounding_boxes?: Array<{
    text: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  }>;
}

export interface OCROptions {
  backend?: 'auto' | 'tesseract' | 'easyocr' | 'claude';
  language?: string;
  enhance_image?: boolean;
  preserve_layout?: boolean;
  claude_api_key?: string;
}

/**
 * Check available OCR backends
 */
export async function checkOCRBackends(): Promise<{
  tesseract: boolean;
  easyocr: boolean;
  claude: boolean;
}> {
  const results = {
    tesseract: false,
    easyocr: false,
    claude: false,
  };

  // Check tesseract
  try {
    await execPromise('python3 -c "import pytesseract; pytesseract.get_tesseract_version()"');
    results.tesseract = true;
  } catch {
    // Not available
  }

  // Check easyocr
  try {
    await execPromise('python3 -c "import easyocr"');
    results.easyocr = true;
  } catch {
    // Not available
  }

  // Check Claude API key
  if (process.env.ANTHROPIC_API_KEY) {
    results.claude = true;
  }

  return results;
}

/**
 * Process image with OCR
 */
export async function processImage(
  imagePath: string,
  options: OCROptions = {}
): Promise<OCRResult> {
  const startTime = Date.now();

  // Validate image exists
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Image not found: ${imagePath}`);
  }

  // Check available backends
  const backends = await checkOCRBackends();

  // Determine which backend to use
  let backend = options.backend || 'auto';
  if (backend === 'auto') {
    if (backends.tesseract) {
      backend = 'tesseract';
    } else if (backends.easyocr) {
      backend = 'easyocr';
    } else {
      throw new Error('No OCR backend available. Install pytesseract or easyocr.');
    }
  }

  // Validate chosen backend is available
  if (backend === 'tesseract' && !backends.tesseract) {
    throw new Error('Tesseract not available. Install with: pip install pytesseract');
  }
  if (backend === 'easyocr' && !backends.easyocr) {
    throw new Error('EasyOCR not available. Install with: pip install easyocr');
  }
  if (backend === 'claude' && !backends.claude) {
    throw new Error('Claude Vision requires ANTHROPIC_API_KEY environment variable');
  }

  let result: OCRResult;

  switch (backend) {
    case 'tesseract':
      result = await processWithTesseract(imagePath, options);
      break;
    case 'easyocr':
      result = await processWithEasyOCR(imagePath, options);
      break;
    case 'claude':
      result = await processWithClaude(imagePath, options);
      break;
    default:
      throw new Error(`Unknown OCR backend: ${backend}`);
  }

  result.processing_time_ms = Date.now() - startTime;
  result.backend_used = backend;

  logger.info('OCR processing complete', {
    backend,
    confidence: result.confidence,
    textLength: result.raw_text.length,
    processingTime: result.processing_time_ms,
  });

  return result;
}

/**
 * Process with Tesseract
 */
async function processWithTesseract(imagePath: string, options: OCROptions): Promise<OCRResult> {
  const lang = options.language || 'eng';
  const preserveLayout = options.preserve_layout ?? true;

  const script = `
import pytesseract
from PIL import Image
import json
import re

try:
    # Load image
    image = Image.open("${imagePath.replace(/\\/g, '\\\\')}")

    # Optional: Enhance image
    ${
      options.enhance_image
        ? `
    from PIL import ImageEnhance, ImageFilter
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    `
        : ''
    }

    # OCR configuration
    config = "--oem 3 --psm ${preserveLayout ? '6' : '3'}"

    # Get text with confidence data
    data = pytesseract.image_to_data(image, lang="${lang}", config=config, output_type=pytesseract.Output.DICT)

    # Extract text and calculate average confidence
    texts = []
    confidences = []
    boxes = []

    for i, text in enumerate(data['text']):
        if text.strip():
            conf = data['conf'][i]
            if conf > 0:  # Valid confidence
                texts.append(text)
                confidences.append(conf)
                boxes.append({
                    "text": text,
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "width": data['width'][i],
                    "height": data['height'][i],
                    "confidence": conf
                })

    raw_text = pytesseract.image_to_string(image, lang="${lang}", config=config)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Clean text
    cleaned = raw_text
    # Remove excessive whitespace
    cleaned = re.sub(r'\\n{3,}', '\\n\\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    # Fix common OCR errors
    cleaned = re.sub(r'[|]', 'I', cleaned)  # Pipe to I
    cleaned = cleaned.strip()

    result = {
        "success": True,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "confidence": round(avg_confidence, 2),
        "bounding_boxes": boxes[:50]  # Limit boxes returned
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`);
    const result = JSON.parse(stdout.trim());
    if (!result.success) {
      throw new Error(result.error);
    }
    return {
      raw_text: result.raw_text,
      cleaned_text: result.cleaned_text,
      confidence: result.confidence,
      backend_used: 'tesseract',
      processing_time_ms: 0,
      bounding_boxes: result.bounding_boxes,
    };
  } catch (error: any) {
    throw new Error(`Tesseract OCR failed: ${error.message}`);
  }
}

/**
 * Process with EasyOCR
 */
async function processWithEasyOCR(imagePath: string, options: OCROptions): Promise<OCRResult> {
  const lang = options.language || 'en';

  const script = `
import easyocr
import json
import re

try:
    # Initialize reader (will download models if needed)
    reader = easyocr.Reader(['${lang}'], gpu=False)

    # Read image
    results = reader.readtext("${imagePath.replace(/\\/g, '\\\\')}")

    # Process results
    texts = []
    confidences = []
    boxes = []

    for (bbox, text, conf) in results:
        texts.append(text)
        confidences.append(conf)
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        boxes.append({
            "text": text,
            "x": int(min(x_coords)),
            "y": int(min(y_coords)),
            "width": int(max(x_coords) - min(x_coords)),
            "height": int(max(y_coords) - min(y_coords)),
            "confidence": round(conf * 100, 2)
        })

    raw_text = ' '.join(texts)
    avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0

    # Clean text - try to preserve some structure
    # Sort boxes by y coordinate first, then x
    sorted_boxes = sorted(boxes, key=lambda b: (b['y'] // 20, b['x']))
    lines = []
    current_line = []
    current_y = -100

    for box in sorted_boxes:
        if abs(box['y'] - current_y) > 15:  # New line
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [box['text']]
            current_y = box['y']
        else:
            current_line.append(box['text'])

    if current_line:
        lines.append(' '.join(current_line))

    cleaned = '\\n'.join(lines)
    # Fix common issues
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = cleaned.strip()

    result = {
        "success": True,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "confidence": round(avg_confidence, 2),
        "bounding_boxes": boxes[:50]
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 120000 });
    const result = JSON.parse(stdout.trim());
    if (!result.success) {
      throw new Error(result.error);
    }
    return {
      raw_text: result.raw_text,
      cleaned_text: result.cleaned_text,
      confidence: result.confidence,
      backend_used: 'easyocr',
      processing_time_ms: 0,
      bounding_boxes: result.bounding_boxes,
    };
  } catch (error: any) {
    throw new Error(`EasyOCR failed: ${error.message}`);
  }
}

/**
 * Process with Claude Vision API
 */
async function processWithClaude(imagePath: string, options: OCROptions): Promise<OCRResult> {
  const apiKey = options.claude_api_key || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error('Claude API key required');
  }

  // Read image and convert to base64
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString('base64');

  // Determine media type
  const ext = path.extname(imagePath).toLowerCase();
  const mediaType =
    ext === '.png'
      ? 'image/png'
      : ext === '.jpg' || ext === '.jpeg'
        ? 'image/jpeg'
        : ext === '.gif'
          ? 'image/gif'
          : ext === '.webp'
            ? 'image/webp'
            : 'image/jpeg';

  const script = `
import anthropic
import json
import base64

try:
    client = anthropic.Anthropic(api_key="${apiKey}")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "${mediaType}",
                            "data": "${base64Image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract ALL text from this image. This appears to be from a trading/finance book.

Instructions:
1. Extract every word, number, and symbol visible
2. Preserve the structure and formatting as much as possible
3. If there are charts or diagrams, describe what they show
4. For candlestick patterns or charts, describe the pattern name and key features
5. Include any captions, labels, or annotations

Return the extracted text in a clean, readable format."""
                    }
                ]
            }
        ]
    )

    extracted_text = message.content[0].text

    result = {
        "success": True,
        "raw_text": extracted_text,
        "cleaned_text": extracted_text,
        "confidence": 95.0  # Claude Vision is highly accurate
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
`;

  try {
    const { stdout } = await execPromise(`python3 -c '${script}'`, { timeout: 60000 });
    const result = JSON.parse(stdout.trim());
    if (!result.success) {
      throw new Error(result.error);
    }
    return {
      raw_text: result.raw_text,
      cleaned_text: result.cleaned_text,
      confidence: result.confidence,
      backend_used: 'claude',
      processing_time_ms: 0,
    };
  } catch (error: any) {
    throw new Error(`Claude Vision API failed: ${error.message}`);
  }
}

/**
 * Auto-classify content based on keywords
 */
export function classifyContent(text: string): {
  category: Category;
  confidence: number;
  detected_topics: string[];
} {
  const textLower = text.toLowerCase();
  const scores: Record<Category, number> = {} as Record<Category, number>;
  const detectedTopics: string[] = [];

  // Score each category
  for (const [category, definition] of Object.entries(CATEGORY_DEFINITIONS)) {
    let score = 0;

    for (const keyword of definition.keywords) {
      const regex = new RegExp(`\\b${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      const matches = textLower.match(regex);
      if (matches) {
        score += matches.length;
        if (!detectedTopics.includes(keyword)) {
          detectedTopics.push(keyword);
        }
      }
    }

    scores[category as Category] = score;
  }

  // Find best category
  let bestCategory: Category = 'general';
  let bestScore = 0;

  for (const [category, score] of Object.entries(scores)) {
    if (score > bestScore) {
      bestScore = score;
      bestCategory = category as Category;
    }
  }

  // Calculate confidence (rough estimate)
  const totalKeywords = detectedTopics.length;
  const confidence = Math.min(95, Math.max(30, totalKeywords * 15));

  return {
    category: bestCategory,
    confidence,
    detected_topics: detectedTopics.slice(0, 10),
  };
}

/**
 * Clean and normalize extracted text
 */
export function cleanText(rawText: string): string {
  let text = rawText;

  // Normalize line endings
  text = text.replace(/\r\n/g, '\n');

  // Remove excessive blank lines
  text = text.replace(/\n{3,}/g, '\n\n');

  // Remove excessive spaces
  text = text.replace(/ {2,}/g, ' ');

  // Fix common OCR errors
  text = text.replace(/[|]/g, 'I');
  text = text.replace(/[0O](?=[a-z])/g, 'O'); // 0 before lowercase is likely O
  text = text.replace(/(?<=[a-z])[0](?=[a-z])/g, 'o'); // 0 between lowercase is likely o
  text = text.replace(/[1l](?=[A-Z])/g, 'I'); // 1 or l before uppercase is likely I

  // Trim whitespace from each line
  text = text
    .split('\n')
    .map((line) => line.trim())
    .join('\n');

  // Final trim
  text = text.trim();

  return text;
}

/**
 * Batch process multiple images
 */
export async function processImageBatch(
  imagePaths: string[],
  options: OCROptions = {},
  onProgress?: (current: number, total: number, result: OCRResult | null) => void
): Promise<OCRResult[]> {
  const results: OCRResult[] = [];

  for (let i = 0; i < imagePaths.length; i++) {
    try {
      const result = await processImage(imagePaths[i], options);
      results.push(result);
      onProgress?.(i + 1, imagePaths.length, result);
    } catch (error: any) {
      logger.error(`Failed to process image ${imagePaths[i]}`, { error: error.message });
      results.push({
        raw_text: '',
        cleaned_text: '',
        confidence: 0,
        backend_used: 'error',
        processing_time_ms: 0,
      });
      onProgress?.(i + 1, imagePaths.length, null);
    }
  }

  return results;
}
