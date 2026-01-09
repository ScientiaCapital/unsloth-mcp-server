/**
 * Knowledge Database Operations
 *
 * SQLite-based storage for the knowledge catalogue
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import {
  KnowledgeEntry,
  TrainingPair,
  Category,
  DB_SCHEMA,
  AlpacaFormat,
  ShareGPTFormat,
  ChatMLFormat,
  toAlpacaFormat,
  toShareGPTFormat,
  toChatMLFormat,
} from './schema.js';
import logger from '../utils/logger.js';

const execPromise = promisify(exec);

// Helper to extract error message from unknown error type
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

// Default database path
const DEFAULT_DB_PATH = path.join(process.cwd(), 'data', 'knowledge.db');

export class KnowledgeDatabase {
  private dbPath: string;
  private initialized: boolean = false;

  constructor(dbPath?: string) {
    this.dbPath = dbPath || DEFAULT_DB_PATH;
  }

  /**
   * Initialize the database (create tables if needed)
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Ensure data directory exists
    const dbDir = path.dirname(this.dbPath);
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }

    // Execute schema creation via Python (sqlite3)
    const initScript = `
import sqlite3
import os

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Execute schema
schema = '''${DB_SCHEMA.replace(/'/g, "\\'")}'''

for statement in schema.split(';'):
    statement = statement.strip()
    if statement:
        try:
            cursor.execute(statement)
        except Exception as e:
            # Ignore errors for IF NOT EXISTS statements
            if 'already exists' not in str(e).lower():
                print(f"Warning: {e}")

conn.commit()
conn.close()
print('{"success": true}')
`;

    try {
      await execPromise(`python3 -c '${initScript}'`);
      this.initialized = true;
      logger.info('Knowledge database initialized', { path: this.dbPath });
    } catch (error: unknown) {
      const msg = getErrorMessage(error);
      logger.error('Failed to initialize knowledge database', { error: msg });
      throw new Error(`Database initialization failed: ${msg}`);
    }
  }

  /**
   * Generate a unique ID
   */
  private generateId(): string {
    return `ke_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Add a new knowledge entry
   */
  async addEntry(
    entry: Omit<
      KnowledgeEntry,
      'id' | 'created_at' | 'updated_at' | 'training_pairs' | 'related_entries'
    >
  ): Promise<string> {
    await this.initialize();

    const id = this.generateId();
    const now = new Date().toISOString();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Insert main entry
    cursor.execute('''
        INSERT INTO knowledge_entries (
            id, created_at, updated_at,
            source_type, source_book_title, source_author, source_chapter,
            source_page_numbers, source_image_path, source_capture_date,
            raw_text, cleaned_text, category, quality_score, ocr_confidence, manually_reviewed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        "${id}",
        "${now}",
        "${now}",
        "${entry.source.type}",
        ${entry.source.book_title ? `"${entry.source.book_title.replace(/"/g, '\\"')}"` : 'None'},
        ${entry.source.author ? `"${entry.source.author.replace(/"/g, '\\"')}"` : 'None'},
        ${entry.source.chapter ? `"${entry.source.chapter.replace(/"/g, '\\"')}"` : 'None'},
        ${entry.source.page_numbers ? `"${entry.source.page_numbers}"` : 'None'},
        "${entry.source.image_path.replace(/"/g, '\\"')}",
        "${entry.source.capture_date}",
        """${entry.raw_text.replace(/"""/g, '\\"\\"\\"').replace(/\\/g, '\\\\')}""",
        """${entry.cleaned_text.replace(/"""/g, '\\"\\"\\"').replace(/\\/g, '\\\\')}""",
        "${entry.category}",
        ${entry.quality_score},
        ${entry.ocr_confidence},
        ${entry.manually_reviewed ? 1 : 0}
    ))

    # Insert topics
    for topic in ${JSON.stringify(entry.topics)}:
        cursor.execute('INSERT OR IGNORE INTO topics (name) VALUES (?)', (topic,))
        cursor.execute('SELECT id FROM topics WHERE name = ?', (topic,))
        topic_id = cursor.fetchone()[0]
        cursor.execute('INSERT OR IGNORE INTO entry_topics (entry_id, topic_id) VALUES (?, ?)', ("${id}", topic_id))

    # Insert tags
    for tag in ${JSON.stringify(entry.tags)}:
        cursor.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)', (tag,))
        cursor.execute('SELECT id FROM tags WHERE name = ?', (tag,))
        tag_id = cursor.fetchone()[0]
        cursor.execute('INSERT OR IGNORE INTO entry_tags (entry_id, tag_id) VALUES (?, ?)', ("${id}", tag_id))

    conn.commit()
    print(json.dumps({"success": True, "id": "${id}"}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      logger.info('Knowledge entry added', { id });
      return id;
    } catch (error: unknown) {
      logger.error('Failed to add knowledge entry', { error: getErrorMessage(error) });
      throw error;
    }
  }

  /**
   * Get an entry by ID
   */
  async getEntry(id: string): Promise<KnowledgeEntry | null> {
    await this.initialize();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute('SELECT * FROM knowledge_entries WHERE id = ?', ("${id}",))
    row = cursor.fetchone()

    if not row:
        print(json.dumps({"success": True, "entry": None}))
    else:
        columns = [desc[0] for desc in cursor.description]
        entry_dict = dict(zip(columns, row))

        # Get topics
        cursor.execute('''
            SELECT t.name FROM topics t
            JOIN entry_topics et ON t.id = et.topic_id
            WHERE et.entry_id = ?
        ''', ("${id}",))
        topics = [r[0] for r in cursor.fetchall()]

        # Get tags
        cursor.execute('''
            SELECT t.name FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            WHERE et.entry_id = ?
        ''', ("${id}",))
        tags = [r[0] for r in cursor.fetchall()]

        # Get training pairs
        cursor.execute('SELECT * FROM training_pairs WHERE entry_id = ?', ("${id}",))
        pairs = []
        for pair_row in cursor.fetchall():
            pair_columns = [desc[0] for desc in cursor.description]
            pairs.append(dict(zip(pair_columns, pair_row)))

        # Get related entries
        cursor.execute('SELECT related_entry_id FROM related_entries WHERE entry_id = ?', ("${id}",))
        related = [r[0] for r in cursor.fetchall()]

        result = {
            "id": entry_dict["id"],
            "created_at": entry_dict["created_at"],
            "updated_at": entry_dict["updated_at"],
            "source": {
                "type": entry_dict["source_type"],
                "book_title": entry_dict["source_book_title"],
                "author": entry_dict["source_author"],
                "chapter": entry_dict["source_chapter"],
                "page_numbers": entry_dict["source_page_numbers"],
                "image_path": entry_dict["source_image_path"],
                "capture_date": entry_dict["source_capture_date"]
            },
            "raw_text": entry_dict["raw_text"],
            "cleaned_text": entry_dict["cleaned_text"],
            "category": entry_dict["category"],
            "topics": topics,
            "tags": tags,
            "quality_score": entry_dict["quality_score"],
            "ocr_confidence": entry_dict["ocr_confidence"],
            "manually_reviewed": bool(entry_dict["manually_reviewed"]),
            "training_pairs": pairs,
            "related_entries": related
        }

        print(json.dumps({"success": True, "entry": result}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.entry;
    } catch (error: unknown) {
      logger.error('Failed to get knowledge entry', { error: getErrorMessage(error), id });
      throw error;
    }
  }

  /**
   * Search entries by text (full-text search)
   */
  async searchEntries(query: string, limit: number = 20): Promise<KnowledgeEntry[]> {
    await this.initialize();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Full-text search
    cursor.execute('''
        SELECT ke.* FROM knowledge_entries ke
        JOIN knowledge_fts fts ON ke.id = fts.id
        WHERE knowledge_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    ''', ("${query.replace(/"/g, '\\"')}", ${limit}))

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    entries = []
    for row in rows:
        entry_dict = dict(zip(columns, row))
        entry_id = entry_dict["id"]

        # Get topics
        cursor.execute('''
            SELECT t.name FROM topics t
            JOIN entry_topics et ON t.id = et.topic_id
            WHERE et.entry_id = ?
        ''', (entry_id,))
        topics = [r[0] for r in cursor.fetchall()]

        # Get tags
        cursor.execute('''
            SELECT t.name FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            WHERE et.entry_id = ?
        ''', (entry_id,))
        tags = [r[0] for r in cursor.fetchall()]

        entries.append({
            "id": entry_dict["id"],
            "created_at": entry_dict["created_at"],
            "updated_at": entry_dict["updated_at"],
            "source": {
                "type": entry_dict["source_type"],
                "book_title": entry_dict["source_book_title"],
                "image_path": entry_dict["source_image_path"],
                "capture_date": entry_dict["source_capture_date"]
            },
            "raw_text": entry_dict["raw_text"][:200] + "..." if len(entry_dict["raw_text"]) > 200 else entry_dict["raw_text"],
            "cleaned_text": entry_dict["cleaned_text"][:200] + "..." if len(entry_dict["cleaned_text"]) > 200 else entry_dict["cleaned_text"],
            "category": entry_dict["category"],
            "topics": topics,
            "tags": tags,
            "quality_score": entry_dict["quality_score"],
            "ocr_confidence": entry_dict["ocr_confidence"],
            "manually_reviewed": bool(entry_dict["manually_reviewed"]),
            "training_pairs": [],
            "related_entries": []
        })

    print(json.dumps({"success": True, "entries": entries}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.entries;
    } catch (error: unknown) {
      logger.error('Failed to search knowledge entries', { error: getErrorMessage(error), query });
      throw error;
    }
  }

  /**
   * List entries by category
   */
  async listByCategory(category: Category, limit: number = 50): Promise<KnowledgeEntry[]> {
    await this.initialize();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute('''
        SELECT * FROM knowledge_entries
        WHERE category = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', ("${category}", ${limit}))

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    entries = []
    for row in rows:
        entry_dict = dict(zip(columns, row))
        entries.append({
            "id": entry_dict["id"],
            "created_at": entry_dict["created_at"],
            "category": entry_dict["category"],
            "source": {
                "type": entry_dict["source_type"],
                "book_title": entry_dict["source_book_title"],
                "image_path": entry_dict["source_image_path"]
            },
            "cleaned_text": entry_dict["cleaned_text"][:100] + "..." if len(entry_dict["cleaned_text"]) > 100 else entry_dict["cleaned_text"],
            "quality_score": entry_dict["quality_score"]
        })

    print(json.dumps({"success": True, "entries": entries, "count": len(entries)}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.entries;
    } catch (error: unknown) {
      logger.error('Failed to list entries by category', {
        error: getErrorMessage(error),
        category,
      });
      throw error;
    }
  }

  /**
   * Add a training pair to an entry
   */
  async addTrainingPair(entryId: string, pair: Omit<TrainingPair, 'id'>): Promise<string> {
    await this.initialize();

    const pairId = `tp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute('''
        INSERT INTO training_pairs (id, entry_id, type, instruction, input, output, system_prompt, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        "${pairId}",
        "${entryId}",
        "${pair.type}",
        ${pair.instruction ? `"""${pair.instruction.replace(/"""/g, '\\"\\"\\"')}"""` : 'None'},
        ${pair.input ? `"""${pair.input.replace(/"""/g, '\\"\\"\\"')}"""` : 'None'},
        """${pair.output.replace(/"""/g, '\\"\\"\\"')}""",
        ${pair.system_prompt ? `"""${pair.system_prompt.replace(/"""/g, '\\"\\"\\"')}"""` : 'None'},
        ${pair.quality_score}
    ))
    conn.commit()
    print(json.dumps({"success": True, "id": "${pairId}"}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      logger.info('Training pair added', { pairId, entryId });
      return pairId;
    } catch (error: unknown) {
      logger.error('Failed to add training pair', { error: getErrorMessage(error) });
      throw error;
    }
  }

  /**
   * Get all training pairs for export
   */
  async getAllTrainingPairs(minQuality: number = 0): Promise<TrainingPair[]> {
    await this.initialize();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute('''
        SELECT * FROM training_pairs
        WHERE quality_score >= ?
        ORDER BY quality_score DESC
    ''', (${minQuality},))

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    pairs = []
    for row in rows:
        pair_dict = dict(zip(columns, row))
        pairs.append({
            "id": pair_dict["id"],
            "type": pair_dict["type"],
            "instruction": pair_dict["instruction"],
            "input": pair_dict["input"],
            "output": pair_dict["output"],
            "system_prompt": pair_dict["system_prompt"],
            "quality_score": pair_dict["quality_score"]
        })

    print(json.dumps({"success": True, "pairs": pairs, "count": len(pairs)}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.pairs;
    } catch (error: unknown) {
      logger.error('Failed to get training pairs', { error: getErrorMessage(error) });
      throw error;
    }
  }

  /**
   * Export training data to a file
   */
  async exportTrainingData(
    outputPath: string,
    format: 'alpaca' | 'sharegpt' | 'chatml',
    minQuality: number = 0
  ): Promise<{ count: number; path: string }> {
    const pairs = await this.getAllTrainingPairs(minQuality);

    let exportData: (AlpacaFormat | ShareGPTFormat | ChatMLFormat)[];

    switch (format) {
      case 'alpaca':
        exportData = pairs.map(toAlpacaFormat);
        break;
      case 'sharegpt':
        exportData = pairs.map(toShareGPTFormat);
        break;
      case 'chatml':
        exportData = pairs.map(toChatMLFormat);
        break;
    }

    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Write to file
    fs.writeFileSync(outputPath, JSON.stringify(exportData, null, 2));

    logger.info('Training data exported', { path: outputPath, count: exportData.length, format });

    return { count: exportData.length, path: outputPath };
  }

  /**
   * Get database statistics
   */
  async getStats(): Promise<{
    total_entries: number;
    entries_by_category: Record<string, number>;
    total_training_pairs: number;
    avg_quality_score: number;
  }> {
    await this.initialize();

    const script = `
import sqlite3
import json

db_path = "${this.dbPath.replace(/\\/g, '\\\\')}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Total entries
    cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
    total_entries = cursor.fetchone()[0]

    # Entries by category
    cursor.execute('SELECT category, COUNT(*) FROM knowledge_entries GROUP BY category')
    by_category = {row[0]: row[1] for row in cursor.fetchall()}

    # Total training pairs
    cursor.execute('SELECT COUNT(*) FROM training_pairs')
    total_pairs = cursor.fetchone()[0]

    # Average quality score
    cursor.execute('SELECT AVG(quality_score) FROM knowledge_entries')
    avg_quality = cursor.fetchone()[0] or 0

    print(json.dumps({
        "success": True,
        "stats": {
            "total_entries": total_entries,
            "entries_by_category": by_category,
            "total_training_pairs": total_pairs,
            "avg_quality_score": round(avg_quality, 2)
        }
    }))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
`;

    try {
      const { stdout } = await execPromise(`python3 -c '${script}'`);
      const result = JSON.parse(stdout.trim());
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.stats;
    } catch (error: unknown) {
      logger.error('Failed to get database stats', { error: getErrorMessage(error) });
      throw error;
    }
  }
}

// Default instance
export const knowledgeDb = new KnowledgeDatabase();
