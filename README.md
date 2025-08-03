# FastRTC App

A real-time audio/video chat application using Gemini AI and WebRTC.

## Setup

1. **Install uv** :
   ```bash
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install dependencies** :
   ```bash
   uv add gradio numpy websockets python-dotenv fastrtc google-genai Pillow
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example file
   copy .env.example .env
   
   # Edit .env and add your Gemini API key
   # GEMINI_API_KEY=your_actual_api_key_here
   ```

## Running the Application

```bash
# Run with uv
C:\Users\rony.thekkan\.local\bin\uv.exe run python app.py

# Or activate the virtual environment first
.venv\Scripts\activate
python app.py
```

## Project Structure

- `app.py` - Main application file
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions
- `.env` - Environment variables (create from .env.example)
- `.venv/` - Virtual environment (created by uv)

## Dependencies Installed

- gradio>=5.38.1 - Web UI framework
- numpy>=2.2.6 - Numerical computing
- websockets>=15.0.1 - WebSocket support
- python-dotenv>=1.1.1 - Environment variable loading
- fastrtc>=0.0.29 - WebRTC functionality
- google-genai>=1.27.0 - Google Gemini AI
- pillow>=11.3.0 - Image processing
