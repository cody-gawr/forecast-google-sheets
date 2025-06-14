# End-to-End Static Ad Concept Generation Pipeline

## Tools & Stack

| Component    | Stack                     |
| ------------ | ------------------------- |
| LLMs         | `gpt-3.5-turbo`, `gpt-4o` |
| Embeddings   | `SentenceTransformers`    |
| Storage      | JSON / local cache / DB   |
| QA Interface | `Streamlit` (optional)    |
| Deployment   | FastAPI / Airflow / CLI   |
| Language     | Python                    |

## Pipeline Architecture

```css
┌────────────────────────────┐
│  [1] Product URL Ingest    │ ◄──── user input / batch config
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [2] GPT-3.5 Product Analysis │ ◄──── analyzes URL + extracts themes/features
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [3] Prompt Builder (GPT-4o)│ ◄──── dynamic templates + product context
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [4] Concept Generation     │ ◄──── 10+ diverse ad concepts
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [5] Diversity & Tone QA    │ ◄──── semantic similarity filter + tone match
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [6] Sora Prompt Formatter  │ ◄──── Inject product URL + visual prompt
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [7] Streamlit QA Review    │ ◄──── human-in-the-loop QA (optional)
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│  [8] JSON Export → Sora     │ ◄──── format-ready structured output
└────────────────────────────┘
```

## Product URL Ingest

**Function**: Accept product URLs as input to bootstrap the pipeline

**Why**:

- The product page is the source of truth for the brand’s messaging, imagery, and benefits.

- Starting with a URL enables the system to operate dynamically across different SKUs and brands.

**Technical Considerations**:

- Accept via config file, API endpoint, CLI param, Database, Batch file(CSV)

- Normalize + validate URLs

- Supports batch mode for scaling multiple products in parallel

| Source Type             | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| ✅ **Config file**      | Local `.yaml` or `.json` for single-product testing / dev runs |
| ✅ **CLI param**        | Simple scripting / ad-hoc usage                                |
| ✅ **API endpoint**     | Triggered externally by other services (e.g. CMS, job queue)   |
| ✅ **Database**         | **Best for scale + automation** — central source of truth      |
| ✅ **Batch file (CSV)** | Good for one-time loads or QA review jobs                      |

```python
product_url = "https://purdyandfigg.com/pages/starterkit-offer"
```

## GPT-3.5 Product Analyzer (Fast Context Extractor)

### Function:

Use the OpenAI gpt-3.5-turbo model to extract:

- Brand tone

- Emotional positioning

- Key product features

- Value propositions

- Unique selling points (USPs)

### Technical Flow

```text
[product_url]
   ↓
[gpt-3.5 prompt: summarize product features, tone, emotional pitch]
   ↓
[structured summary block for LLM prompt injection]
```

### Prompt Example

```python
def analyze_product_page(product_url):
    prompt = f"""
You are a product strategist. Analyze the product page at this URL: {product_url}

Summarize it in 5 sections:
1. Product Type
2. Core Features
3. Value Propositions (Why buy it?)
4. Emotional Tone / Brand Voice
5. Notable Differentiators

Keep it clear, brand-aligned, and ad-friendly.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
```

**Technical Considerations**:

- Add retry logic and rate limit guards for high-volume workloads.

- Cache analysis results by URL to avoid duplicate billing and API use.

## Generate Concepts with GPT-4o

**Function**: Generate 10 diverse ad concepts using the structured prompt.
**Why**:

- LLMs are excellent at generating creative ideation, but without guidance may be repetitive or off-brand

- Using GPT-4o allows rich scene description, well-formatted copy, and high variation

**Technical Considerations**:

- Set temperature=0.9+ for creative variance

- Use max tokens judiciously to ensure all 10 outputs fit

- Parse or validate LLM output as structured blocks (using regex or pydantic)

```python
def generate_concepts(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.95,  # boost creative diversity
        max_tokens=3000
    )
    return response.choices[0].message.content
```

## Diversity Scoring (Sentence Transformers)

**Function**: Enforce semantic diversity between concepts using cosine similarity on embeddings.
**Why**:

- LLMs tend to repeat structures/patterns, especially under similar prompts

- Without enforcement, output will have near duplicates → lowers performance on ads

**Technical Considerations**:

- Use all-MiniLM-L6-v2 (lightweight + fast) or OpenAI embeddings

- Compare visual descriptions and filter based on a cosine sim threshold (~0.85)

- Run this check before QA or final output formatting

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def filter_diverse_concepts(concepts, threshold=0.85):
    texts = [c["visual_description"] for c in concepts]
    embeddings = model.encode(texts)
    filtered = []
    for i, vec in enumerate(embeddings):
        if max(np.delete(util.cos_sim(vec, embeddings).numpy()[0], i)) < threshold:
            filtered.append(concepts[i])
    return filtered
```

## Brand Tone Alignment

**Function**: Validate whether each concept aligns with the brand’s tone-of-voice.
**Why**:

- Consistency in tone is critical for brand trust, especially on platforms like Meta

- Automating tone checks reduces manual QA load

**Technical Considerations**:

- Build tone profiles using example sentences (brand-defined)

- Average tone embeddings as reference vector

- Score each concept’s value prop → reject or flag if similarity < threshold (~0.75)

- Cache brand tone vectors for efficiency

```python
def check_tone_alignment(concept, tone_embedding, threshold=0.75):
    value_embedding = model.encode(concept["value_prop"])
    sim_score = util.cos_sim(value_embedding, tone_embedding).item()
    return sim_score >= threshold
```

## Sora Prompt Formatting

**Function**: Convert concept + URL into a structured, human-readable Sora visual prompt.
**Why**:

- Sora needs a clear, contextual prompt to generate ad images

- Including product URL and styling metadata improves fidelity and performance

**Technical Considerations**:

- Use strict text layout with label blocks (Sora-specific style)

- Validate prompt contains:

  - Visual Description

  - In-Image Copy

  - Font Strategy

  - Product URL

  - Creative instructions (no humans, white bg, etc.)

```python
  def format_sora_prompt(concept, product_url):
    return f"""
  Input Product URL: {product_url}


Visual Description: {concept['visual_description']}
Font Strategy: {concept['font_strategy']}

In-Image Copy:
Attention Line: {concept['attention_line']}
Value Prop: {concept['value_prop']}
CTA Badge: {concept['cta_badge']}

Rules:

- No humans
- Meta Feed & Story compatibility
- Typography must follow design system
  """

def build_sora_inputs(concepts, product_url, asset_folder_url):
    return [{
        "concept_id": f"purdyfigg_{i+1:03}",
        "product_url": product_url,
        "sora_prompt": format_sora_prompt(c, product_url),
        "image_assets": [
            f"{asset_folder_url}/main.jpg",
            f"{asset_folder_url}/alt.jpg"
        ]
    } for i, c in enumerate(concepts)]

```

## Streamlit QA Review (Optional)

**Function**: Human-in-the-loop interface for final review, approval, or edits.
**Why**:

- Automation + LLMs ≠ perfect → allows creative leads to manually inspect high-priority outputs

- Adds guardrails before sending to Sora

**Technical Considerations**:

- Fast load, paginated per concept

- Approve/Reject buttons per concept

- Save review decisions (e.g., to DB or export)

- Optional filters by tone score, diversity, concept type

## Export to Sora JSON

**Function**: Write final output in structured format (with assets) for downstream visual generation
**Why**:

- Sora expects well-formed input with clear visual instruction and copy

- Exporting in batch JSON format supports job queueing, API triggering, or dashboard loading

**Technical Considerations**:

- Validate JSON with `pydantic` or schema rules

- Include asset links (image paths, fallback options)

- Append metadata like concept_id, brand_id, or batch_id

- Upload to S3, save to Dropbox or Airtable, or trigger job pipeline

```python

import json

def save_final_output(sora_concepts, path="sora_concepts.json"):
    with open(path, "w") as f:
        json.dump(sora_concepts, f, indent=2)
```
