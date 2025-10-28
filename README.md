# Erebus – Authenticity & Post-Processing Agent

**Mission**: Remove detectable AI fingerprints and reintroduce human irregularity while retaining meaning.

## Core Capabilities

- Detect AI-generated patterns using ensemble classifiers
- Apply linguistic perturbations (clause deformation, rhythm irregularity, entropy injection)
- Generate differential reports (diff_summary) to track text changes
- Output AI-likelihood and voice deviation metrics

## API Endpoints

- `POST /v1/clean` - Refine drafts to remove AI fingerprints
- `GET /healthz` - Health check endpoint

## Key Deliverables

- `cleaned.json` schema for refined content and metrics
- AI detection ensemble classifiers
- Linguistic perturbation algorithms
- Differential reporting system

## Dependencies

- Python 3.11+
- FastAPI, uvicorn
- OpenAI or custom heuristic-based detector
- Access to Mnemosyne for stylistic baseline comparison

## Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the service
uvicorn main:app --reload --port 8003

# Install additional dependencies
pip install <package>
pip freeze > requirements.txt
```

## Data Flow

Erebus receives drafts from IRIS and sends cleaned content to Kairos:

```
IRIS → Erebus → Kairos
         ↓
    Mnemosyne (style baseline)
```

## Authenticity Techniques

Erebus employs:
- **Clause deformation**: Subtle restructuring while preserving meaning
- **Rhythm irregularity**: Breaking uniform sentence patterns
- **Entropy injection**: Adding natural linguistic variation
- **Pattern disruption**: Removing detectable AI markers

## Quality Metrics

- AI-likelihood score (target: < 0.25)
- Voice deviation from baseline (target: < 0.35)
- Edit distance from original draft
- Meaning preservation validation

## Related Repositories

- [agent-iris](https://github.com/stephenpeters/agent-iris) - Provides drafts for refinement
- [agent-kairos](https://github.com/stephenpeters/agent-kairos) - Receives cleaned content for scheduling
- [agent-mnemosyne](https://github.com/stephenpeters/agent-mnemosyne) - Provides style baseline
- [agent-sdk](https://github.com/stephenpeters/agent-sdk) - Shared schemas and utilities
