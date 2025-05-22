---
layout: page
title: Alfred
description: Your Local AI Coding Butler
img: assets/img/alfred.png
importance: 1
category: work
---

AIFred is your local AI coding buddy! It runs discreetly on your screen, ready to help explain code, suggest fixes, answer programming questions, or even take voice commands, using AI models that run **entirely on your own machine**. Privacy first!

<div class="row">
    <div class="col-sm-12 mt-3 mt-md-0">
        <img src="https://github.com/user-attachments/assets/49a28264-ed3a-4947-b2a8-7f32dc837cac" alt="AIFred Preview" class="img-fluid rounded z-depth-1">
    </div>
</div>
<div class="caption">
    The main interface of AIFred, showcasing its on-screen terminal-styled overlay.
</div>

## ✨ Key Features

-   **On-Screen Helper**: A sleek, terminal-styled overlay window that stays conveniently on top.
-   **Local LLM Power**: Integrates with **Ollama** to run large language models (like Gemma, Llama 3, etc.) locally. Your code and queries remain private.
-   **🗣️ Voice Input**: Click the microphone icon, speak your query, and AIFred transcribes it locally using the Parakeet ASR model, sending it to the selected Ollama LLM.
-   **Model Selection**: Easily choose which installed Ollama model you want to use via a dropdown menu.
-   **Syntax Highlighting**: Displays code snippets in AI responses with proper highlighting for better readability.
-   **Quick Access**: Toggle the window's visibility instantly with a global hotkey (`Ctrl+Alt+C` by default).

## 🛠️ Technical Implementation

AIFred combines a modern frontend with a local backend to ensure privacy and performance:

-   **Frontend**: Electron, HTML5, CSS3, JavaScript.
-   **Backend**: Python 3.x, FastAPI.
-   **LLM Integration**: Ollama API for local model execution.
-   **ASR/STT**: NVIDIA NeMo Toolkit (Parakeet-TDT Model) and pydub for local speech-to-text.
-   **Audio Handling**: Web Audio API (Frontend), FFmpeg (Backend dependency via pydub).
-   **UI**: Marked for Markdown parsing, Highlight.js for syntax highlighting.

## 🚀 Future Enhancements

AIFred is continuously evolving. Here are some planned improvements:

1.  System tray icon for background operation and quick access.
2.  Ability to capture code/text directly from clipboard or screen selection.
3.  Settings UI for easier configuration (Ollama URL, default model, hotkeys).
4.  Support for multimodal Ollama models (sending images).
5.  Packaging for easier distribution (Installers).

## Try It Out / Get Involved

Ready to have your own local AI coding assistant?

For developers interested in contributing, running the project locally, or learning more, check out the [GitHub repository](https://github.com/emharsha1812/alfred) for installation instructions, prerequisites, and further details.

## 🙏 Acknowledgements
- Ollama Team
- NVIDIA NeMo Team (for the Parakeet model)
- Electron Team
- FastAPI Team