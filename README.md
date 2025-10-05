# AI Performance Review System

An intelligent performance review system that conducts dynamic AI-powered interviews 
and generates comprehensive reports by aggregating insights from multiple review sessions.

## Features

- **Dynamic AI Interviews**: Conducts adaptive interviews with contextual follow-up questions
- **Multi-Source Reviews**: Handles self, peer, and manager reviews separately
- **Semantic Storage**: Stores reviews in Qdrant with rich metadata for filtering
- **Historical Tracking**: Analyzes trends across multiple review cycles
- **Intelligent Reports**: Synthesizes insights from multiple sessions into actionable feedback

## Tech Stack

- **LangGraph**: Stateful conversational agents with autonomous completion
- **Qdrant**: Vector database for semantic search and retrieval
- **Ollama**: Local LLM deployment (llama3.1:8b)
- **Streamlit**: Interactive web interface
- **LangChain**: LLM orchestration and document processing

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Install and start Ollama:
   ```bash
   # Install from https://ollama.ai
   ollama pull llama3.1:8b
   ollama serve
   ```

6. Start Qdrant (using Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at http://localhost:8501

## Project Structure

- `config/`: Configuration and settings
- `src/agents/`: LangGraph interview agents
- `src/storage/`: Qdrant vector store integration
- `src/aggregation/`: Multi-session report generation
- `src/llm/`: Ollama LLM client
- `app/`: Streamlit UI components
- `tests/`: Unit tests

## Configuration

Edit `config/settings.py` to customize:
- Review competency areas
- Question templates
- Completion criteria
- Report formats

## License

MIT License