# TumorBoard Quick Start Guide

This guide will get you up and running with TumorBoard in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key (or Anthropic/other LLM provider)

## Installation

1. Clone and navigate to the directory:
```bash
cd tumor_board
```

2. Create a virtual environment with Python 3.11:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e ".[dev]"
```

4. Create a `.env` file in the project root with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

The `.env` file is automatically loaded when you run `tumorboard` commands!

## Your First Assessment

Assess the BRAF V600E mutation in melanoma:

```bash
tumorboard assess BRAF V600E --tumor "Melanoma"
```

You should see output like:
```
================================================================================
VARIANT ACTIONABILITY ASSESSMENT REPORT
================================================================================

Variant: BRAF V600E
Tumor Type: Melanoma

Tier: Tier I
Confidence: 95.0%
Evidence Strength: Strong

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
BRAF V600E is a well-established actionable mutation in melanoma...

--------------------------------------------------------------------------------
RECOMMENDED THERAPIES (2)
--------------------------------------------------------------------------------

1. Vemurafenib
   Evidence Level: FDA-approved
   Approval Status: Approved
   Clinical Context: First-line therapy

2. Dabrafenib + Trametinib
   Evidence Level: FDA-approved
   Approval Status: Approved
   Clinical Context: First-line therapy
...
```

## Batch Processing

1. Create a file `my_variants.json`:
```json
[
  {
    "gene": "BRAF",
    "variant": "V600E",
    "tumor_type": "Melanoma"
  },
  {
    "gene": "EGFR",
    "variant": "L858R",
    "tumor_type": "Lung Adenocarcinoma"
  },
  {
    "gene": "KRAS",
    "variant": "G12C",
    "tumor_type": "Non-Small Cell Lung Cancer"
  }
]
```

2. Run batch assessment:
```bash
tumorboard batch my_variants.json --output my_results.json
```

3. View results:
```bash
cat my_results.json
```

## Validate Against Gold Standard

Run validation to see how well the model performs:

```bash
tumorboard validate benchmarks/gold_standard.json
```

This will output metrics like:
```
================================================================================
VALIDATION REPORT
================================================================================

Total Cases: 15
Correct Predictions: 13
Overall Accuracy: 86.67%
Average Confidence: 87.50%

--------------------------------------------------------------------------------
PER-TIER METRICS
--------------------------------------------------------------------------------

Tier I:
  Precision: 90.00%
  Recall: 90.00%
  F1 Score: 90.00%
...
```

## Try Different Models

Use Claude instead of GPT:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
tumorboard assess BRAF V600E --tumor "Melanoma" --model claude-3-sonnet-20240229
```

Or GPT-4 for potentially better accuracy:

```bash
tumorboard assess BRAF V600E --tumor "Melanoma" --model gpt-4o
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [gold_standard.json](benchmarks/gold_standard.json) to understand the benchmark dataset
- Try the [sample_batch.json](benchmarks/sample_batch.json) for more examples
- Run the test suite: `pytest`
- Experiment with different prompts by modifying [src/tumorboard/llm/prompts.py](src/tumorboard/llm/prompts.py)

## Common Issues

**Issue**: `ModuleNotFoundError: No module named 'tumorboard'`
**Solution**: Make sure you installed with `pip install -e .` from the project root

**Issue**: `litellm.exceptions.AuthenticationError`
**Solution**: Check that your API key is set correctly: `echo $OPENAI_API_KEY`

**Issue**: `MyVariantAPIError`
**Solution**: This is usually a network issue or the variant isn't in the database. Check your internet connection.

## Getting Help

- Check the [README.md](README.md) for full documentation
- Look at test files in `tests/` for usage examples
- Open an issue on GitHub if you find bugs
