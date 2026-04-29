---
name: image-gen
description: Generate images from text prompts using the wu CLI. Use when user wants to generate an image, create a picture from a description, text-to-image, generate with conditioning images, or create AI art. Trigger phrases include "generate an image", "create a picture", "text to image", "image generation", "generate from prompt".
---

# Generate Images from Text Prompts

Generates images from text prompts using the `wu image-gen` CLI command. Supports text-to-image generation and conditioned generation with reference images. Public backends include Gemini (default), OpenAI, and NVIDIA NIM.

## Prerequisites

- The `wu` CLI must be installed and available on PATH
- Environment variables from `.env` must be set:
  - `GOOGLE_API_KEY` for Gemini (default)
  - `OPENAI_API_KEY` for OpenAI image generation
  - `NVIDIA_API_KEY` for NIM image generation

## Command Reference

```bash
wu image-gen <prompt> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `prompt` | Yes | Text prompt describing the desired image |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `generated.png` | Output file path for the generated image |
| `--image` | `-i` | None | Conditioning image(s) — can be repeated for multiple images |
| `--backend` | `-b` | `gemini` | Backend: `gemini`, `openai`, `nim` |
| `--model` | `-m` | None | Model to use (backend-specific default if omitted) |
| `--base-url` | | None | API base URL override for OpenAI-compatible endpoints |
| `--verbose` | `-v` | False | Enable verbose logging |

## Common Workflows

### Simple text-to-image

```bash
wu image-gen "A photorealistic red sports car on a mountain road" -o car.png
```

### Generate with a conditioning image

```bash
wu image-gen "Apply realistic materials to this 3D render" -i render.png -o result.png
```

### Multiple conditioning images

```bash
wu image-gen "Generate with applied materials" -i target.png -i depth.png -o output.png
```

### Use Gemini backend

```bash
wu image-gen "A sunset over the ocean" --backend gemini -o sunset.png
```

### Specify a model

```bash
wu image-gen "A cute cat" --backend gemini -m "gemini-3-pro-image-preview" -o cat.png
```

### Use a local OpenAI-compatible image-gen endpoint

```bash
wu image-gen "A cute cat" --backend openai \
  --model black-forest-labs/flux.2-klein-4b \
  --base-url http://localhost:8000/v1 \
  -o cat.png
```

## Common Issues

### "API key required" error
Cause: Missing environment variable for the selected backend.
Solution: Set `GOOGLE_API_KEY` (Gemini default), `OPENAI_API_KEY` (OpenAI), or `NVIDIA_API_KEY` (NIM) in your `.env` file.

### "No image found in response" error
Cause: The model returned text instead of an image.
Solution: Make your prompt more explicitly request image generation. Some prompts may be refused by content filters.

### Timeout or slow response
Cause: Image generation can take 10-30 seconds depending on the model and prompt complexity.
Solution: Use `--verbose` to see request progress. Try a simpler prompt or different model.
