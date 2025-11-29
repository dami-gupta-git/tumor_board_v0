# TumorBoard

An LLM-powered cancer variant actionability assessment tool with built-in validation framework.

## Overview

TumorBoard combines clinical evidence from multiple genomic databases (CIViC, ClinVar, COSMIC) with large language models to assess the clinical actionability of cancer variants using the **AMP/ASCO/CAP 4-tier classification system**.

### Key Features

- **Evidence Aggregation**: Automatically fetches variant evidence from MyVariant.info API
- **LLM Assessment**: Uses state-of-the-art LLMs to interpret evidence and assign actionability tiers
- **Validation Framework**: Built-in benchmarking against gold standard datasets
- **Multiple LLM Support**: Works with OpenAI, Anthropic, and other providers via litellm
- **Async Throughout**: Fast, concurrent processing for batch assessments
- **Rich CLI**: Beautiful command-line interface with progress indicators

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tumor_board

# Install dependencies (requires Python 3.11+)
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### 1. Set up API keys

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Assess a single variant

```bash
tumorboard assess BRAF V600E --tumor "Melanoma"
```

Output:
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
BRAF V600E is a well-established actionable mutation in melanoma with
FDA-approved targeted therapies including vemurafenib, dabrafenib, and
encorafenib.

...
```

### 3. Batch processing

```bash
tumorboard batch benchmarks/sample_batch.json --output results.json
```

### 4. Run validation

```bash
tumorboard validate benchmarks/gold_standard.json
```

## CLI Commands

### `assess` - Single Variant Assessment

Assess the clinical actionability of a single variant.

```bash
tumorboard assess <GENE> <VARIANT> --tumor <TUMOR_TYPE> [OPTIONS]

Arguments:
  GENE        Gene symbol (e.g., BRAF)
  VARIANT     Variant notation (e.g., V600E)

Options:
  -t, --tumor TEXT       Tumor type (required)
  -m, --model TEXT       LLM model [default: gpt-4o-mini]
  -o, --output PATH      Save results to JSON file
  -v, --verbose          Enable debug logging
```

**Examples:**

```bash
# Basic assessment
tumorboard assess BRAF V600E --tumor "Melanoma"

# Use Claude instead of GPT
tumorboard assess EGFR L858R --tumor "Lung Cancer" --model claude-3-sonnet-20240229

# Save to file
tumorboard assess KRAS G12C --tumor "NSCLC" --output assessment.json
```

### `batch` - Batch Processing

Process multiple variants from a JSON file.

```bash
tumorboard batch <INPUT_FILE> [OPTIONS]

Arguments:
  INPUT_FILE    JSON file with variant list

Options:
  -o, --output PATH        Output file [default: results.json]
  -m, --model TEXT         LLM model [default: gpt-4o-mini]
  -c, --max-concurrent N   Max concurrent requests [default: 5]
  -v, --verbose            Enable debug logging
```

**Input Format:**

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
  }
]
```

**Example:**

```bash
tumorboard batch variants.json --output results.json --max-concurrent 10
```

### `validate` - Validation

Validate LLM assessments against a gold standard dataset.

```bash
tumorboard validate <GOLD_STANDARD_FILE> [OPTIONS]

Arguments:
  GOLD_STANDARD_FILE    Gold standard JSON file

Options:
  -m, --model TEXT         LLM model [default: gpt-4o-mini]
  -o, --output PATH        Save detailed results to JSON
  -c, --max-concurrent N   Max concurrent validations [default: 3]
  -v, --verbose            Enable debug logging
```

**Gold Standard Format:**

```json
{
  "entries": [
    {
      "gene": "BRAF",
      "variant": "V600E",
      "tumor_type": "Melanoma",
      "expected_tier": "Tier I",
      "notes": "FDA-approved therapies available",
      "references": ["PMID:12345"]
    }
  ]
}
```

**Example:**

```bash
tumorboard validate benchmarks/gold_standard.json --output validation_results.json
```

## AMP/ASCO/CAP Tier System

- **Tier I**: Variants with strong clinical significance
  - FDA-approved therapies for specific variant + tumor type
  - Professional guideline recommendations
  - Strong evidence from clinical trials

- **Tier II**: Variants with potential clinical significance
  - FDA-approved therapies for different tumor types
  - Clinical trial evidence
  - Case reports or smaller studies

- **Tier III**: Variants of unknown clinical significance
  - Preclinical evidence only
  - Uncertain biological significance
  - Conflicting evidence

- **Tier IV**: Benign or likely benign variants
  - Known benign polymorphisms
  - No oncogenic evidence

## Validation Framework

The validation framework is a key differentiator, allowing you to:

1. **Benchmark Performance**: Test LLM accuracy against known correct classifications
2. **Identify Weaknesses**: See exactly where and why the model makes mistakes
3. **Track Improvements**: Monitor performance as you refine prompts or change models
4. **Build Confidence**: Demonstrate reliability with metrics

### Validation Metrics

- **Overall Accuracy**: Percentage of correct tier assignments
- **Per-Tier Metrics**: Precision, recall, and F1 score for each tier
- **Failure Analysis**: Detailed breakdown of incorrect predictions
- **Tier Distance**: How far off incorrect predictions are

### Example Validation Output

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
  TP: 9, FP: 1, FN: 1

Tier II:
  Precision: 75.00%
  Recall: 75.00%
  F1 Score: 75.00%
  TP: 3, FP: 1, FN: 1

...
```

## Project Structure

```
tumor_board/
├── src/tumorboard/
│   ├── api/              # MyVariant.info API client
│   ├── llm/              # LLM service and prompts
│   ├── models/           # Pydantic data models
│   ├── validation/       # Validation framework
│   ├── engine.py         # Core assessment engine
│   └── cli.py            # CLI interface
├── tests/                # Test suite
├── benchmarks/           # Gold standard datasets
│   ├── gold_standard.json
│   └── sample_batch.json
├── pyproject.toml        # Project configuration
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tumorboard --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

### Adding New Gold Standard Entries

Edit `benchmarks/gold_standard.json`:

```json
{
  "gene": "YOUR_GENE",
  "variant": "YOUR_VARIANT",
  "tumor_type": "YOUR_TUMOR_TYPE",
  "expected_tier": "Tier I|II|III|IV",
  "notes": "Clinical rationale",
  "references": ["Citation 1", "Citation 2"]
}
```

## Supported LLM Models

Via litellm, TumorBoard supports:

- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Google**: `gemini-pro`, `gemini-1.5-pro`
- **Azure OpenAI**: `azure/<deployment>`
- **And many more...**

Specify the model with the `--model` flag:

```bash
tumorboard assess BRAF V600E --tumor Melanoma --model claude-3-sonnet-20240229
```

## Data Sources

- **MyVariant.info**: Aggregates data from multiple sources
  - **CIViC**: Clinical Interpretations of Variants in Cancer
  - **ClinVar**: Clinical significance of variants
  - **COSMIC**: Catalogue of Somatic Mutations in Cancer

No authentication required for MyVariant.info API.

## Performance Considerations

- **API Rate Limits**: MyVariant.info has rate limits; use batch processing for large datasets
- **LLM Costs**: GPT-4 is more accurate but expensive; gpt-4o-mini is a good balance
- **Concurrency**: Adjust `--max-concurrent` based on API limits and costs
- **Caching**: Consider caching evidence data for repeated assessments

## Limitations

- **Evidence Quality**: Assessment quality depends on available database evidence
- **LLM Hallucination**: LLMs may occasionally generate incorrect information
- **Novel Variants**: Limited evidence for rare/novel variants
- **Context Window**: Very long evidence may be truncated

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use TumorBoard in your research, please cite:

```
[Citation information to be added]
```

## References

- [AMP/ASCO/CAP Guidelines](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5707196/)
- [MyVariant.info](https://myvariant.info/)
- [CIViC](https://civicdb.org/)
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)
- [COSMIC](https://cancer.sanger.ac.uk/cosmic)

## Support

For issues, questions, or contributions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs

---

**Note**: This tool is for research purposes only. Clinical decisions should always be made by qualified healthcare professionals based on comprehensive patient evaluation.
