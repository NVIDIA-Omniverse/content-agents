---
name: image-gen
description: Generate images from text prompts using the wu CLI. Use when the user wants text-to-image generation, image-conditioned generation, visual material references, prompt-based concept images, or a local OpenAI-compatible image-generation endpoint through Gemini, OpenAI, or NVIDIA NIM backends.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - image-generation
  - cli
  - genai
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI, write access for the output image path, and provider credentials for the selected image-generation backend such as GOOGLE_API_KEY, OPENAI_API_KEY, or NVIDIA_API_KEY; local OpenAI-compatible endpoints may use --base-url.
---

# Image Generation

Generate images from prompts with `wu image-gen`.

## When to Use

- Use when the user asks to generate an image from a text prompt.
- Use when the user wants one or more conditioning images passed to the image
  generation backend.
- Use when Material or Texture workflows need a visual reference image created
  outside the service pipeline.
- Use the service-specific skills when image generation should happen inside a
  Material or Texture Agent REST run.

## Limitations

- Generated content is sent to the selected backend. Confirm provider choice
  for sensitive prompts or conditioning images.
- Backend support for conditioning images, model names, and content filters
  varies.
- `--base-url` is for OpenAI-compatible endpoints and is used with
  `--backend openai`.
- The command writes one output image path per invocation.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- Set the credential for the selected backend: `GOOGLE_API_KEY`,
  `OPENAI_API_KEY`, or `NVIDIA_API_KEY`.
- Ensure conditioning image paths exist when using `--image`.
- Ensure the output directory is writable.

## Instructions

1. Confirm the prompt and intended output path.
2. Choose the backend. Use `gemini` by default unless the user asks for another
   provider or a local OpenAI-compatible endpoint.
3. Add one or more `--image` arguments only when conditioning images are part
   of the request.
4. Add `--model` when the user names a model or the endpoint requires one.
5. Add `--base-url` only for a local or custom OpenAI-compatible endpoint.
6. Report the generated output path and any provider-side refusal or missing
   credential.

## Command Reference

```bash
wu image-gen <prompt> [OPTIONS]
```

| Option | Description |
|---|---|
| `--output`, `-o` | Output image path. Default is `generated.png`. |
| `--image`, `-i` | Conditioning image path. Repeat for multiple images. |
| `--backend`, `-b` | Backend: `gemini`, `openai`, or `nim`. |
| `--model`, `-m` | Optional backend-specific model name. |
| `--base-url` | OpenAI-compatible endpoint URL, such as a local NIM container. |
| `--verbose`, `-v` | Enable debug logging. |

## Common Workflows

```bash
# Text to image.
wu image-gen "A photorealistic red sports car on a mountain road" \
  --output car.png

# Condition on one image.
wu image-gen "Apply realistic brushed aluminum to this render" \
  --image render.png \
  --output reference.png

# Multiple conditioning images.
wu image-gen "Generate a material reference from these views" \
  --image front.png \
  --image side.png \
  --output material_reference.png

# Gemini backend.
wu image-gen "A sunset over the ocean" --backend gemini --output sunset.png

# Local OpenAI-compatible image-generation endpoint.
wu image-gen "A small robot on a workbench" \
  --backend openai \
  --model black-forest-labs/flux.2-klein-4b \
  --base-url http://localhost:8000/v1 \
  --output robot.png
```

## Output Format

Report:

- Command executed, backend, model when provided, and output path.
- Conditioning image paths when used.
- Whether the output image was written.
- Any blocker such as missing credential, missing conditioning image, provider
  refusal, empty image response, endpoint connection failure, or timeout.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| API key required | Credential for the selected backend is missing. | Set the matching provider key in the local environment or `.env`. |
| Conditioning image not found | One `--image` path is wrong. | Use absolute paths or rerun from the correct directory. |
| No image found in response | Provider returned text or filtered the request. | Make the prompt explicitly ask for image generation or try another backend/model. |
| Custom endpoint auth error | `--base-url` points to a secured endpoint without matching endpoint credentials. | Configure the endpoint-specific key or use a local no-auth endpoint. |
| Slow response | Image generation can take tens of seconds. | Use `--verbose`, simplify the prompt, or try a faster model. |
