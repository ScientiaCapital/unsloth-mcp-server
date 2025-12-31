/**
 * Web Scraper Utility for Knowledge Gathering
 *
 * Supports multiple providers for gathering training data:
 * - Exa AI: Semantic search, similar content discovery, deep research
 * - Firecrawl: Full site crawling, LLM-ready markdown extraction
 *
 * @module utils/web-scraper
 */

import axios, { AxiosInstance } from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import logger from './logger.js';

// ============================================================================
// Types & Interfaces
// ============================================================================

export type ScraperProvider = 'exa' | 'firecrawl';

export interface WebScraperConfig {
  provider: ScraperProvider;
  exa?: {
    apiKey: string;
    baseUrl?: string;
  };
  firecrawl?: {
    apiKey: string;
    baseUrl?: string;
  };
  outputDir?: string;
  maxConcurrent?: number;
  retryAttempts?: number;
}

// Exa Types
export interface ExaSearchOptions {
  query: string;
  numResults?: number;
  includeDomains?: string[];
  excludeDomains?: string[];
  startPublishedDate?: string;
  endPublishedDate?: string;
  useAutoprompt?: boolean;
  type?: 'keyword' | 'neural' | 'auto';
  category?: string;
  includeText?: boolean;
  includeHighlights?: boolean;
}

export interface ExaSearchResult {
  id: string;
  url: string;
  title: string;
  score: number;
  publishedDate?: string;
  author?: string;
  text?: string;
  highlights?: string[];
}

export interface ExaResearchOptions {
  query: string;
  maxResults?: number;
  outputFormat?: 'markdown' | 'json';
}

export interface ExaResearchResult {
  summary: string;
  sources: ExaSearchResult[];
  citations: { text: string; url: string }[];
}

// Firecrawl Types
export interface FirecrawlScrapeOptions {
  url: string;
  formats?: ('markdown' | 'html' | 'rawHtml' | 'links' | 'screenshot')[];
  onlyMainContent?: boolean;
  includeTags?: string[];
  excludeTags?: string[];
  waitFor?: number;
  timeout?: number;
}

export interface FirecrawlCrawlOptions {
  url: string;
  maxDepth?: number;
  maxPages?: number;
  includePaths?: string[];
  excludePaths?: string[];
  allowBackwardLinks?: boolean;
  allowExternalLinks?: boolean;
  ignoreSitemap?: boolean;
}

export interface FirecrawlMapOptions {
  url: string;
  search?: string;
  ignoreSitemap?: boolean;
  includeSubdomains?: boolean;
  limit?: number;
}

export interface FirecrawlExtractOptions {
  urls: string[];
  prompt?: string;
  schema?: Record<string, unknown>;
  systemPrompt?: string;
}

export interface ScrapedPage {
  url: string;
  title?: string;
  markdown?: string;
  html?: string;
  text?: string;
  links?: string[];
  metadata?: Record<string, unknown>;
  scrapedAt: Date;
  provider: ScraperProvider;
}

export interface CrawlJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  url: string;
  pagesScraped: number;
  totalPages?: number;
  startedAt: Date;
  completedAt?: Date;
  results?: ScrapedPage[];
  error?: string;
}

export interface TrainingDataResult {
  source: string;
  content: string;
  metadata: {
    url: string;
    title?: string;
    scrapedAt: Date;
    provider: ScraperProvider;
    wordCount: number;
    format: 'markdown' | 'text';
  };
}

// ============================================================================
// Web Scraper Class
// ============================================================================

export class WebScraper {
  private config: WebScraperConfig;
  private exaClient?: AxiosInstance;
  private firecrawlClient?: AxiosInstance;
  private crawlJobs: Map<string, CrawlJob> = new Map();

  constructor(config: WebScraperConfig) {
    this.config = {
      outputDir: './scraped_data',
      maxConcurrent: 3,
      retryAttempts: 3,
      ...config,
    };

    this.initializeClients();
  }

  private initializeClients(): void {
    // Initialize Exa client
    if (this.config.exa?.apiKey) {
      this.exaClient = axios.create({
        baseURL: this.config.exa.baseUrl || 'https://api.exa.ai',
        headers: {
          'x-api-key': this.config.exa.apiKey,
          'Content-Type': 'application/json',
        },
        timeout: 60000,
      });
      logger.info('Exa client initialized');
    }

    // Initialize Firecrawl client
    if (this.config.firecrawl?.apiKey) {
      this.firecrawlClient = axios.create({
        baseURL: this.config.firecrawl.baseUrl || 'https://api.firecrawl.dev/v1',
        headers: {
          Authorization: `Bearer ${this.config.firecrawl.apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 120000,
      });
      logger.info('Firecrawl client initialized');
    }

    // Ensure output directory exists
    if (this.config.outputDir && !fs.existsSync(this.config.outputDir)) {
      fs.mkdirSync(this.config.outputDir, { recursive: true });
    }
  }

  // ==========================================================================
  // Exa Methods
  // ==========================================================================

  /**
   * Search the web using Exa's semantic search
   */
  async exaSearch(options: ExaSearchOptions): Promise<ExaSearchResult[]> {
    if (!this.exaClient) {
      throw new Error('Exa client not configured. Provide exa.apiKey in config.');
    }

    logger.info('Performing Exa search', { query: options.query });

    try {
      const response = await this.exaClient.post('/search', {
        query: options.query,
        numResults: options.numResults || 10,
        includeDomains: options.includeDomains,
        excludeDomains: options.excludeDomains,
        startPublishedDate: options.startPublishedDate,
        endPublishedDate: options.endPublishedDate,
        useAutoprompt: options.useAutoprompt ?? true,
        type: options.type || 'auto',
        category: options.category,
        contents: {
          text: options.includeText ?? true,
          highlights: options.includeHighlights ?? true,
        },
      });

      const results: ExaSearchResult[] = response.data.results.map(
        (r: Record<string, unknown>) => ({
          id: r.id as string,
          url: r.url as string,
          title: r.title as string,
          score: r.score as number,
          publishedDate: r.publishedDate as string | undefined,
          author: r.author as string | undefined,
          text: r.text as string | undefined,
          highlights: r.highlights as string[] | undefined,
        })
      );

      logger.info('Exa search completed', { resultCount: results.length });
      return results;
    } catch (error) {
      logger.error('Exa search failed', { error });
      throw this.handleError(error, 'exa');
    }
  }

  /**
   * Find similar pages to a given URL using Exa
   */
  async exaFindSimilar(
    url: string,
    options?: { numResults?: number; includeDomains?: string[]; excludeDomains?: string[] }
  ): Promise<ExaSearchResult[]> {
    if (!this.exaClient) {
      throw new Error('Exa client not configured. Provide exa.apiKey in config.');
    }

    logger.info('Finding similar pages', { url });

    try {
      const response = await this.exaClient.post('/findSimilar', {
        url,
        numResults: options?.numResults || 10,
        includeDomains: options?.includeDomains,
        excludeDomains: options?.excludeDomains,
        contents: { text: true },
      });

      const results: ExaSearchResult[] = response.data.results.map(
        (r: Record<string, unknown>) => ({
          id: r.id as string,
          url: r.url as string,
          title: r.title as string,
          score: r.score as number,
          text: r.text as string | undefined,
        })
      );

      logger.info('Find similar completed', { resultCount: results.length });
      return results;
    } catch (error) {
      logger.error('Find similar failed', { error });
      throw this.handleError(error, 'exa');
    }
  }

  /**
   * Get contents from URLs using Exa
   */
  async exaGetContents(
    urls: string[],
    options?: { text?: boolean; highlights?: boolean }
  ): Promise<ExaSearchResult[]> {
    if (!this.exaClient) {
      throw new Error('Exa client not configured. Provide exa.apiKey in config.');
    }

    logger.info('Getting contents from URLs', { urlCount: urls.length });

    try {
      const response = await this.exaClient.post('/contents', {
        ids: urls,
        contents: {
          text: options?.text ?? true,
          highlights: options?.highlights ?? false,
        },
      });

      return response.data.results;
    } catch (error) {
      logger.error('Get contents failed', { error });
      throw this.handleError(error, 'exa');
    }
  }

  /**
   * Perform deep research using Exa's research API
   */
  async exaResearch(options: ExaResearchOptions): Promise<ExaResearchResult> {
    if (!this.exaClient) {
      throw new Error('Exa client not configured. Provide exa.apiKey in config.');
    }

    logger.info('Performing deep research', { query: options.query });

    try {
      const response = await this.exaClient.post('/research', {
        query: options.query,
        maxResults: options.maxResults || 20,
        outputFormat: options.outputFormat || 'markdown',
      });

      logger.info('Research completed', { sourceCount: response.data.sources?.length });
      return response.data;
    } catch (error) {
      logger.error('Research failed', { error });
      throw this.handleError(error, 'exa');
    }
  }

  // ==========================================================================
  // Firecrawl Methods
  // ==========================================================================

  /**
   * Scrape a single URL using Firecrawl
   */
  async firecrawlScrape(options: FirecrawlScrapeOptions): Promise<ScrapedPage> {
    if (!this.firecrawlClient) {
      throw new Error('Firecrawl client not configured. Provide firecrawl.apiKey in config.');
    }

    logger.info('Scraping URL with Firecrawl', { url: options.url });

    try {
      const response = await this.firecrawlClient.post('/scrape', {
        url: options.url,
        formats: options.formats || ['markdown'],
        onlyMainContent: options.onlyMainContent ?? true,
        includeTags: options.includeTags,
        excludeTags: options.excludeTags,
        waitFor: options.waitFor,
        timeout: options.timeout || 30000,
      });

      const data = response.data.data;
      const result: ScrapedPage = {
        url: options.url,
        title: data.metadata?.title,
        markdown: data.markdown,
        html: data.html,
        links: data.links,
        metadata: data.metadata,
        scrapedAt: new Date(),
        provider: 'firecrawl',
      };

      logger.info('Scrape completed', { url: options.url, hasMarkdown: !!result.markdown });
      return result;
    } catch (error) {
      logger.error('Scrape failed', { url: options.url, error });
      throw this.handleError(error, 'firecrawl');
    }
  }

  /**
   * Crawl an entire website using Firecrawl
   */
  async firecrawlCrawl(options: FirecrawlCrawlOptions): Promise<CrawlJob> {
    if (!this.firecrawlClient) {
      throw new Error('Firecrawl client not configured. Provide firecrawl.apiKey in config.');
    }

    logger.info('Starting crawl job', { url: options.url });

    try {
      const response = await this.firecrawlClient.post('/crawl', {
        url: options.url,
        maxDepth: options.maxDepth || 2,
        limit: options.maxPages || 100,
        includePaths: options.includePaths,
        excludePaths: options.excludePaths,
        allowBackwardLinks: options.allowBackwardLinks ?? false,
        allowExternalLinks: options.allowExternalLinks ?? false,
        ignoreSitemap: options.ignoreSitemap ?? false,
        scrapeOptions: {
          formats: ['markdown'],
          onlyMainContent: true,
        },
      });

      const jobId = response.data.id;
      const job: CrawlJob = {
        id: jobId,
        status: 'running',
        url: options.url,
        pagesScraped: 0,
        startedAt: new Date(),
      };

      this.crawlJobs.set(jobId, job);
      logger.info('Crawl job started', { jobId, url: options.url });

      return job;
    } catch (error) {
      logger.error('Crawl failed to start', { url: options.url, error });
      throw this.handleError(error, 'firecrawl');
    }
  }

  /**
   * Check the status of a crawl job
   */
  async firecrawlCrawlStatus(jobId: string): Promise<CrawlJob> {
    if (!this.firecrawlClient) {
      throw new Error('Firecrawl client not configured.');
    }

    try {
      const response = await this.firecrawlClient.get(`/crawl/${jobId}`);
      const data = response.data;

      const job: CrawlJob = {
        id: jobId,
        status:
          data.status === 'completed'
            ? 'completed'
            : data.status === 'failed'
              ? 'failed'
              : 'running',
        url: this.crawlJobs.get(jobId)?.url || '',
        pagesScraped: data.completed || 0,
        totalPages: data.total,
        startedAt: this.crawlJobs.get(jobId)?.startedAt || new Date(),
        completedAt: data.status === 'completed' ? new Date() : undefined,
        results: data.data?.map((d: Record<string, unknown>) => {
          const metadata = d.metadata as Record<string, unknown> | undefined;
          return {
            url: (metadata?.sourceURL as string) || (d.url as string),
            title: metadata?.title as string | undefined,
            markdown: d.markdown as string | undefined,
            metadata: metadata,
            scrapedAt: new Date(),
            provider: 'firecrawl' as ScraperProvider,
          };
        }),
      };

      this.crawlJobs.set(jobId, job);
      return job;
    } catch (error) {
      logger.error('Failed to get crawl status', { jobId, error });
      throw this.handleError(error, 'firecrawl');
    }
  }

  /**
   * Wait for a crawl job to complete
   */
  async firecrawlCrawlWait(
    jobId: string,
    options?: { pollInterval?: number; timeout?: number }
  ): Promise<CrawlJob> {
    const pollInterval = options?.pollInterval || 5000;
    const timeout = options?.timeout || 600000; // 10 minutes default
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const job = await this.firecrawlCrawlStatus(jobId);

      if (job.status === 'completed' || job.status === 'failed') {
        return job;
      }

      logger.debug('Crawl in progress', { jobId, pagesScraped: job.pagesScraped });
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Crawl job ${jobId} timed out after ${timeout}ms`);
  }

  /**
   * Map/discover URLs on a website using Firecrawl
   */
  async firecrawlMap(options: FirecrawlMapOptions): Promise<string[]> {
    if (!this.firecrawlClient) {
      throw new Error('Firecrawl client not configured.');
    }

    logger.info('Mapping URLs', { url: options.url });

    try {
      const response = await this.firecrawlClient.post('/map', {
        url: options.url,
        search: options.search,
        ignoreSitemap: options.ignoreSitemap ?? false,
        includeSubdomains: options.includeSubdomains ?? false,
        limit: options.limit || 5000,
      });

      const urls: string[] = response.data.links || [];
      logger.info('URL mapping completed', { urlCount: urls.length });
      return urls;
    } catch (error) {
      logger.error('URL mapping failed', { error });
      throw this.handleError(error, 'firecrawl');
    }
  }

  /**
   * Extract structured data from URLs using Firecrawl's AI extraction
   */
  async firecrawlExtract(options: FirecrawlExtractOptions): Promise<Record<string, unknown>[]> {
    if (!this.firecrawlClient) {
      throw new Error('Firecrawl client not configured.');
    }

    logger.info('Extracting structured data', { urlCount: options.urls.length });

    try {
      const response = await this.firecrawlClient.post('/extract', {
        urls: options.urls,
        prompt: options.prompt,
        schema: options.schema,
        systemPrompt: options.systemPrompt,
      });

      return response.data.data || [];
    } catch (error) {
      logger.error('Extraction failed', { error });
      throw this.handleError(error, 'firecrawl');
    }
  }

  // ==========================================================================
  // Training Data Methods
  // ==========================================================================

  /**
   * Gather training data from search results (Exa)
   */
  async gatherFromSearch(
    query: string,
    options?: {
      numResults?: number;
      domains?: string[];
      minWordCount?: number;
    }
  ): Promise<TrainingDataResult[]> {
    const searchResults = await this.exaSearch({
      query,
      numResults: options?.numResults || 20,
      includeDomains: options?.domains,
      includeText: true,
    });

    const minWords = options?.minWordCount || 100;
    const results: TrainingDataResult[] = [];

    for (const result of searchResults) {
      if (!result.text) continue;

      const wordCount = result.text.split(/\s+/).length;
      if (wordCount < minWords) continue;

      results.push({
        source: result.url,
        content: result.text,
        metadata: {
          url: result.url,
          title: result.title,
          scrapedAt: new Date(),
          provider: 'exa',
          wordCount,
          format: 'text',
        },
      });
    }

    logger.info('Gathered training data from search', {
      query,
      totalResults: searchResults.length,
      validResults: results.length,
    });

    return results;
  }

  /**
   * Gather training data from website crawl (Firecrawl)
   */
  async gatherFromSite(
    url: string,
    options?: {
      maxPages?: number;
      maxDepth?: number;
      includePaths?: string[];
      minWordCount?: number;
    }
  ): Promise<TrainingDataResult[]> {
    // Start crawl
    const job = await this.firecrawlCrawl({
      url,
      maxPages: options?.maxPages || 50,
      maxDepth: options?.maxDepth || 2,
      includePaths: options?.includePaths,
    });

    // Wait for completion
    const completedJob = await this.firecrawlCrawlWait(job.id);

    if (completedJob.status === 'failed') {
      throw new Error(`Crawl failed: ${completedJob.error}`);
    }

    const minWords = options?.minWordCount || 100;
    const results: TrainingDataResult[] = [];

    for (const page of completedJob.results || []) {
      const content = page.markdown || page.text || '';
      const wordCount = content.split(/\s+/).length;

      if (wordCount < minWords) continue;

      results.push({
        source: page.url,
        content,
        metadata: {
          url: page.url,
          title: page.title,
          scrapedAt: page.scrapedAt,
          provider: 'firecrawl',
          wordCount,
          format: page.markdown ? 'markdown' : 'text',
        },
      });
    }

    logger.info('Gathered training data from site', {
      url,
      pagesScraped: completedJob.pagesScraped,
      validResults: results.length,
    });

    return results;
  }

  /**
   * Gather training data using deep research (Exa Research API)
   */
  async gatherFromResearch(
    topic: string,
    options?: {
      maxResults?: number;
      outputFormat?: 'markdown' | 'json';
    }
  ): Promise<{
    summary: TrainingDataResult;
    sources: TrainingDataResult[];
  }> {
    const research = await this.exaResearch({
      query: topic,
      maxResults: options?.maxResults || 20,
      outputFormat: options?.outputFormat || 'markdown',
    });

    const summary: TrainingDataResult = {
      source: `research:${topic}`,
      content: research.summary,
      metadata: {
        url: `research:${topic}`,
        title: `Research: ${topic}`,
        scrapedAt: new Date(),
        provider: 'exa',
        wordCount: research.summary.split(/\s+/).length,
        format: 'markdown',
      },
    };

    const sources: TrainingDataResult[] = research.sources
      .filter((s) => s.text)
      .map((s) => ({
        source: s.url,
        content: s.text!,
        metadata: {
          url: s.url,
          title: s.title,
          scrapedAt: new Date(),
          provider: 'exa' as ScraperProvider,
          wordCount: s.text!.split(/\s+/).length,
          format: 'text' as const,
        },
      }));

    logger.info('Gathered research data', {
      topic,
      summaryWords: summary.metadata.wordCount,
      sourceCount: sources.length,
    });

    return { summary, sources };
  }

  /**
   * Save gathered data to disk
   */
  async saveTrainingData(
    data: TrainingDataResult[],
    options?: {
      filename?: string;
      format?: 'jsonl' | 'json' | 'markdown';
    }
  ): Promise<string> {
    const outputDir = this.config.outputDir || './scraped_data';
    const format = options?.format || 'jsonl';
    const filename =
      options?.filename || `training_data_${Date.now()}.${format === 'markdown' ? 'md' : format}`;
    const filepath = path.join(outputDir, filename);

    let content: string;

    switch (format) {
      case 'jsonl':
        content = data.map((d) => JSON.stringify(d)).join('\n');
        break;
      case 'json':
        content = JSON.stringify(data, null, 2);
        break;
      case 'markdown':
        content = data
          .map(
            (d) =>
              `# ${d.metadata.title || d.source}\n\nSource: ${d.metadata.url}\nScraped: ${d.metadata.scrapedAt}\n\n${d.content}\n\n---\n`
          )
          .join('\n');
        break;
      default:
        content = data.map((d) => JSON.stringify(d)).join('\n');
    }

    fs.writeFileSync(filepath, content, 'utf-8');
    logger.info('Saved training data', { filepath, recordCount: data.length });

    return filepath;
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Get statistics about gathered data
   */
  getStats(data: TrainingDataResult[]): {
    totalDocuments: number;
    totalWords: number;
    avgWordsPerDoc: number;
    byProvider: Record<string, number>;
    byFormat: Record<string, number>;
  } {
    const totalWords = data.reduce((sum, d) => sum + d.metadata.wordCount, 0);
    const byProvider: Record<string, number> = {};
    const byFormat: Record<string, number> = {};

    for (const d of data) {
      byProvider[d.metadata.provider] = (byProvider[d.metadata.provider] || 0) + 1;
      byFormat[d.metadata.format] = (byFormat[d.metadata.format] || 0) + 1;
    }

    return {
      totalDocuments: data.length,
      totalWords,
      avgWordsPerDoc: Math.round(totalWords / data.length),
      byProvider,
      byFormat,
    };
  }

  private handleError(error: unknown, provider: ScraperProvider): Error {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const message = error.response?.data?.error || error.message;

      if (status === 401) {
        return new Error(`${provider} API key is invalid or missing`);
      }
      if (status === 429) {
        return new Error(`${provider} rate limit exceeded. Please wait and try again.`);
      }
      if (status === 402) {
        return new Error(`${provider} payment required. Check your subscription.`);
      }

      return new Error(`${provider} API error (${status}): ${message}`);
    }

    return error instanceof Error ? error : new Error(String(error));
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a WebScraper instance from environment variables
 */
export function createWebScraper(overrides?: Partial<WebScraperConfig>): WebScraper {
  const config: WebScraperConfig = {
    provider: (process.env.WEB_SCRAPER_PROVIDER as ScraperProvider) || 'firecrawl',
    exa: process.env.EXA_API_KEY
      ? {
          apiKey: process.env.EXA_API_KEY,
          baseUrl: process.env.EXA_API_URL,
        }
      : undefined,
    firecrawl: process.env.FIRECRAWL_API_KEY
      ? {
          apiKey: process.env.FIRECRAWL_API_KEY,
          baseUrl: process.env.FIRECRAWL_API_URL,
        }
      : undefined,
    outputDir: process.env.SCRAPED_DATA_DIR || './scraped_data',
    ...overrides,
  };

  return new WebScraper(config);
}

export default WebScraper;
