---
name: vision
description: Analyze images using a Vision-Language Model via the wu CLI. Use when user wants to describe an image, ask questions about a picture, caption a photo, or analyze visual content with a VLM. Trigger phrases include "describe this image", "what is in this image", "analyze this picture", "caption this photo", "ask about image".
---

# Analyze Images with VLM

Sends images to a Vision-Language Model using the `wu vision` CLI command. Supports describing, captioning, question-answering, and any other image+text prompt task.

## Prerequisites

- The `wu` CLI must be installed and available on PATH
- Environment variables from `.env` must be set:
  - `NVIDIA_API_KEY` for NIM service (default — uses https://integrate.api.nvidia.com/v1)
  - `OPENAI_API_KEY` for OpenAI service
  - `ANTHROPIC_API_KEY` for Anthropic service
  - `GOOGLE_API_KEY` for Google Gemini service

## Command Reference

```bash
wu vision <image_path> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `image_path` | Yes | Path to image file (PNG, JPG, etc.) |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | "Describe this image in detail." | Prompt/question about the image |
| `--service` | `-s` | `nim` | VLM service: `nim`, `openai`, `anthropic`, `gemini` |
| `--model` | `-m` | None | Model to use (service default if omitted) |
| `--system-prompt` | | (helpful assistant) | System prompt for the VLM |
| `--temperature` | `-t` | 0.7 | Sampling temperature |
| `--max-tokens` | | 1024 | Maximum tokens in response |
| `--format` | `-f` | `text` | Output format: `text` or `json` |
| `--verbose` | `-v` | False | Enable verbose logging |

## Common Workflows

### Describe an image

```bash
wu vision image.png
```

### Ask a specific question

```bash
wu vision image.png -p "What objects are visible in this image?"
```

### Get JSON output for programmatic use

```bash
wu vision image.png -f json
```

### Use OpenAI instead of NIM

```bash
wu vision image.png -s openai
```

### Use a specific model

```bash
wu vision image.png -m "qwen/qwen3.5-397b-a17b"
```

## Common Issues

### "API key required" error
Cause: Missing environment variable for the selected service.
Solution: Set `NVIDIA_API_KEY` (for nim, default), `OPENAI_API_KEY` (for OpenAI), `ANTHROPIC_API_KEY` (for Anthropic), or `GOOGLE_API_KEY` (for Google Gemini) in your `.env` file.

### Timeout or slow response
Cause: Large images or cold-start on the service side.
Solution: Try a smaller image or a different service with `-s`.
