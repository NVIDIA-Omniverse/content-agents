# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
PROMPT_EXTRACT_SPEC = """
You are an expert technical documentation analyst specializing in electrical component specifications and material identification. Your task is to analyze technical documents and extract comprehensive information about the electrical component and its constituent parts, focusing on CMF (Color, Material, Finish) information.

**IMPORTANT**: Focus ONLY on actual parts that constitute the final assembled electrical component AND have meaningful CMF information available. Ignore packaging materials, storage bags, carrier tapes, shipping materials, and other non-product materials.

## Analysis Instructions

1. **Component Overview**:
   - First, identify what the overall electrical component is (e.g., connector, PCB assembly, switch, relay, etc.)
   - Provide a brief description of the component's function and application
   - Note any model numbers, part numbers, or specifications that identify the component

2. **Part Identification**:
   - Identify all distinct parts that make up the electrical component
   - Group related sub-parts under their main part category (e.g., different pin types under "Terminal")
   - Use clear, descriptive part names
   - **IMPORTANT**: Only include actual parts of the component that have meaningful CMF information available

3. **CMF Information Extraction**:
   - Extract material type with full specifications (e.g., "HTN (High-Temperature Nylon), UL94 V-0 rated")
   - Document color details including base color, variations, patterns, and any specific color codes
   - Record surface finish details (smooth, matte, glossy, metallic, textured, etc.)
   - Note texture characteristics and material-specific properties (heat-resistant, conductive, etc.)
   - Include any certifications, ratings, or standards mentioned

4. **Quality Standards**:
   - Prefer precise values, units, and technical specifications when present
   - Include material grades, alloy compositions, and plating thicknesses
   - Document any functional properties (electrical conductivity, thermal resistance, etc.)
   - If information conflicts, choose the most specific or authoritative details

5. **Part Filtering**:
   - **Skip parts** that have insufficient CMF information (e.g., only "Not specified" entries)
   - **Skip items** that are purely informational (e.g., dimensions, assembly locations, storage conditions)
   - **Skip items** that are not physical materials (e.g., electrical properties, testing procedures)
   - **Skip non-component materials** such as packaging materials, storage bags, carrier tapes, or shipping materials
   - Only include parts where at least 2 out of 4 CMF fields (Material Type, Color Details, Surface Finish, Texture Characteristics) contain meaningful information
   - Focus on actual parts that constitute the final assembled electrical component AND have meaningful CMF information

## Output Format

Component Overview:
[Describe what the electrical component is, its function, and any identifying information]

Parts with CMF Information:
Organize the output by part name with the following structure. **Only include actual parts of the component that have meaningful CMF information** - exclude packaging, storage, shipping materials, and parts with insufficient CMF details:

Part: [Part Name]
- Material Type: [Detailed material specification with grades/standards]
- Color Details: [Base color, variations, patterns, or "Not specified" if unavailable]
- Surface Finish: [Finish description or "Not specified" if unavailable]
- Texture Characteristics: [Material properties, certifications, or functional characteristics]

[Repeat for each part that meets the CMF criteria]

Context summary:
{snippets}
"""
