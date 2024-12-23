# Appian-IITM-AI-Challenge


# 🚀 Smart Document Categorizer & Analyzer

A powerful document management system that leverages AI to automatically categorize, analyze, and interact with documents. Built with state-of-the-art language models and efficient caching mechanisms.

## 🌟 Core Features

### Document Processing
- **Intelligent Classification**: Automated categorization using GPT-4 with custom category mapping
- **Information Extraction**: Smart extraction of key document information (dates, names, numbers)
- **Dynamic Summarization**: AI-powered document summarization with regeneration capability
- **Interactive Chat**: Context-aware document chat using RAG (Retrieval Augmented Generation)

### Technical Architecture
- **Multi-Level Caching**:
  - Redis distributed caching for:
    - Document metadata
    - Chat history
    - Classification results
    - Generated summaries
  - Local LRU cache for:
    - Frequent database queries
    - Document retrieval
    - User session data
  
- **Database Architecture**:
  ```sql
  -- Main Document Table
  CREATE TABLE documents_new (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL,
      category TEXT NOT NULL,
      subcategory TEXT NOT NULL,
      important_info TEXT,
      pdf_data BLOB,
      upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      all_extracted TEXT,
      summary TEXT
  );

  -- Document Versions Table
  CREATE TABLE document_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_id INTEGER,
      version_number INTEGER,
      changes TEXT,
      modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (doc_id) REFERENCES documents_new(id)
  );

  -- Chat History Table
  CREATE TABLE chat_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_id INTEGER,
      user_message TEXT,
      ai_response TEXT,
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (doc_id) REFERENCES documents_new(id)
  );
  ```

### Performance Features
- **Vector Search**: FAISS-based similarity search for document retrieval
- **Async Processing**: Background processing for large documents
- **Cache Invalidation**: Smart cache management with TTL
- **Rate Limiting**: Request throttling for API endpoints

## 🛠️ Technical Stack

### Frontend
- **UI Framework**: Streamlit
- **Styling**: Custom CSS with dark mode
- **Components**: Custom React components for visualization

### Backend
- **Core**: Python 3.8+
- **API Layer**: FastAPI endpoints
- **ML Models**: 
  - GPT-4 (OpenAI)
  - FAISS Vector DB
- **Document Processing**: LlamaParse
- **Frameworks**:
  - LangChain for RAG
  - LangGraph for chat
  
### Storage & Caching
- **Database**: SQLite with custom indices
- **Cache Layers**:
  - Redis (distributed)
  - Local LRU (in-memory)
- **Vector Store**: FAISS

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 8GB RAM minimum
- Redis Server 6.0+
- SQLite 3.30+

### API Keys
```bash
OPENAI_API_KEY=your_openai_key
LLAMA_CLOUD_API_KEY=your_llama_key
```

## 🚀 Installation

1. **Clone & Setup**:
```bash
git clone https://github.com/yourusername/smart-doc-categorizer.git
cd smart-doc-categorizer
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Environment Setup**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Database Initialization**:
```bash
python scripts/init_db.py
```

5. **Redis Setup**:
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu
brew install redis  # macOS

# Start Redis
redis-server
```

6. **Run Application**:
```bash
streamlit run app.py
```

## 💾 Caching Architecture

### Redis Cache Structure
```python
# Document Cache
doc_cache_key = f"doc:{doc_id}"
summary_cache_key = f"summary:{doc_id}"
chat_cache_key = f"chat:{doc_id}"

# Cache TTL
DOC_CACHE_TTL = 3600  # 1 hour
SUMMARY_CACHE_TTL = 86400  # 24 hours
CHAT_CACHE_TTL = 1800  # 30 minutes
```

### Cache Invalidation Strategy
- Time-based expiration
- Smart invalidation on updates
- Partial cache updates
- Cache warming for frequent access

## 🔍 Performance Monitoring

### Metrics Tracked
- Cache hit/miss rates
- Response times
- Memory usage
- API latency
- Document processing time

### Monitoring Tools
- Built-in performance dashboard
- Redis monitoring
- SQLite query analyzer

## 🛡️ Security

- API key encryption
- Redis password protection
- Secure file handling
- Rate limiting
- Input sanitization

## 🔧 Troubleshooting

### Common Issues

#### Redis Connection
```bash
# Check Redis status
redis-cli ping
# Should return PONG
```

#### Database Issues
```bash
# Reset database
python scripts/reset_db.py

# Verify tables
sqlite3 documents_new.db ".tables"
```

#### Cache Problems
```bash
# Clear Redis cache
redis-cli FLUSHALL

# Monitor Redis
redis-cli MONITOR
```

## 📚 API Documentation

Detailed API documentation available at `/docs` when running the application.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
