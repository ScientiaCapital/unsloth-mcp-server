# Changelog

All notable changes to the Unsloth MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-11-07

### Added
- **Configuration System**: Flexible config via JSON files or environment variables
  - Support for `config.json`, `~/.unsloth-mcp-config.json`, or `CONFIG_FILE` env var
  - Hot reload capability
  - Validation and defaults for all settings
- **Caching Layer**: Smart caching for expensive operations
  - In-memory and disk-based cache with TTL
  - Configurable cache size and expiration
  - Automatic cleanup of expired entries
  - Cache statistics and management
- **Progress Reporting**: Real-time progress updates
  - Progress callback system
  - Stage-based progress tracking
  - ETA calculations
  - Integration with Python script execution
- **CLI Tool**: Enhanced command-line interface
  - Installation checking
  - Model listing
  - Configuration viewer
  - Metrics dashboard
  - Cache management
  - Colorized output

### Changed
- Resource limits now configurable via config system
- Improved modularity with separate util modules

## [2.0.1] - 2025-11-07

### Added
- **Testing Infrastructure**
  - Jest testing framework with 43 comprehensive tests
  - 100% coverage on validation and metrics utilities
  - Test scripts: test, test:watch, test:coverage
- **Logging & Observability**
  - Winston structured logging
  - File and console transports
  - Request/response logging
  - Performance metrics tracking
- **Input Validation**
  - Comprehensive validation for all 12 tools
  - Model name, file path, numeric range validation
  - Path traversal prevention
  - Helpful error messages with suggestions
- **Security**
  - Input sanitization for Python scripts
  - Dangerous pattern detection
  - Resource limits enforcement
  - Rate limiting (max 3 concurrent operations)
  - Fixed all npm security vulnerabilities (0 vulnerabilities)
- **Error Handling**
  - Smart error detection with context
  - Specific handling for timeouts, ENOENT, EACCES, OOM
  - Retry logic with exponential backoff
- **Performance Metrics**
  - Tool execution tracking
  - Success/failure rate monitoring
  - Duration statistics

### Changed
- TypeScript strict mode enabled
- Modular utility structure
- Production-ready logging

## [2.0.0] - 2025-11-06

### Added
- **SuperBPE Tokenizer Training**: Train tokenizers with up to 33% fewer tokens
- **Model Information Tool**: Get detailed model architecture info
- **Tokenizer Comparison**: Compare efficiency between tokenizers
- **Performance Benchmarking**: Benchmark inference speed and memory
- **Dataset Discovery**: Search Hugging Face datasets
- **Dataset Preparation**: Format datasets for fine-tuning
- 6 new tools total
- Comprehensive documentation

### Changed
- Version bump to 2.0.0
- Enhanced README with all new tools
- Added workflow examples

## [1.0.0] - 2025-11-06

### Added
- Initial release
- Basic MCP server implementation
- 6 core tools: check_installation, list_supported_models, load_model, finetune_model, generate_text, export_model
- TypeScript implementation
- Basic error handling
