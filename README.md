# TumorBoard v0
An LLM-powered cancer variant actionability assessment tool with a built-in validation framework.

**TL;DR**:   
Molecular tumor boards manually review cancer variants to assign clinical actionability—a time-consuming process requiring expert panels. This research tool automates that workflow by fetching variant evidence from genomic databases (CIViC, ClinVar, COSMIC) and using LLMs to assign AMP/ASCO/CAP tier classifications, mimicking expert judgment. Includes a validation framework to benchmark LLM accuracy against gold-standard classifications. This is a research prototype exploring whether LLMs can approximate clinical decision-making; not for actual clinical use.

## Overview

TumorBoard combines clinical evidence from multiple genomic databases (CIViC, ClinVar, COSMIC). It then uses large language models to approximate expert application of the **AMP/ASCO/CAP 4-tier classification system**.

### Key Features

- **Evidence Aggregation**: Automatically fetches variant evidence from MyVariant.info API
- **LLM Assessment**: Uses LLMs to interpret evidence and assign actionability tiers
- **Validation Framework**: Benchmarks against gold standard datasets
- **Multiple LLM Support**: Works with OpenAI, Anthropic, and other providers via litellm
- **Async Throughout**: Fast, concurrent processing for batch assessments
- **Rich CLI**: Command-line interface with progress indicators

## Why This Tool Exists

Molecular tumor boards face significant challenges:

1. **Resource Intensive**: Expert panels must manually review variants and apply 
   classification frameworks - a time-consuming process requiring coordinated expertise.
2. **Coverage Gaps**: Curated databases like CIViC don't cover every variant-tumor 
   combination, especially rare or novel variants.
3. **Evidence Fragmentation**: Relevant evidence is scattered across multiple 
   databases (CIViC, ClinVar, COSMIC), requiring manual synthesis.
4. **Rapid Evolution**: New trials and approvals constantly change variant 
   actionability.

## Disclaimer

**Limitations:**
- LLMs may hallucinate or misinterpret evidence 
- Pattern matching ≠ expert clinical judgment  
- Requires validation against gold standards (hence the built-in framework)
- Evidence quality: Depends on database coverage
- Novel variants: Limited data for rare variants
- Context windows: Very long evidence may be truncated

**This tool is for research purposes only.** Clinical decisions should always 
be made by qualified healthcare professionals.

## Getting Started

**Installation:**
```bash
git clone <repository-url>
cd tumor_board
pip install -e .
```

**Setup API Key** (choose one):
LLMs are cloud-based AI services that need authentication.

```bash
# OpenAI (default, uses gpt-4o-mini)
export OPENAI_API_KEY="your-key-here"

# Or Anthropic (use with --model claude-3-sonnet-20240229)
export ANTHROPIC_API_KEY="your-key-here"
```

Alternatively, you can use env.example - rename it to .env, and specify your keys there.

**Basic Usage:**

```bash
# Single variant
tumorboard assess BRAF V600E --tumor "Melanoma"

# Batch processing
tumorboard batch benchmarks/sample_batch.json --output results.json

# Validate performance
tumorboard validate benchmarks/gold_standard.json
```

## CLI Reference

### `assess` - Single Variant
Specify a single variant, then run this command to fetch variant evidence and use the LLM to assign an AMP/ASCO/CAP tier classification.

```bash
tumorboard assess <GENE> <VARIANT> --tumor <TUMOR_TYPE> [OPTIONS]

Options:
  -m, --model TEXT    LLM model [default: gpt-4o-mini]
  -o, --output PATH   Save to JSON file
```

Example output:
```
Assessing BRAF V600E in Melanoma...

Variant: BRAF V600E | Tumor: Melanoma
Tier: Tier I | Confidence: 95.0%

BRAF V600E is a well-established actionable mutation in melanoma...

Therapies: Vemurafenib, Dabrafenib
```

### `batch` - Multiple Variants
Specify a JSON file with variant details (gene, variant, tumor type), then run this command to process them concurrently and generate batch results.

```bash
tumorboard batch <INPUT_FILE> [OPTIONS]

Options:
  -o, --output PATH        Output file [default: results.json]
  -m, --model TEXT         LLM model [default: gpt-4o-mini]
  -c, --max-concurrent N   Concurrent requests [default: 5]
```

Input format: `[{"gene": "BRAF", "variant": "V600E", "tumor_type": "Melanoma"}, ...]`

### `validate` - Test Accuracy
Specify a gold standard dataset with known correct tier classifications, then run this command to benchmark the LLM's performance and identify where it agrees or disagrees with expert consensus—this is critical for evaluating reliability before using the tool for research.

Provides:
- Overall accuracy and per-tier precision/recall/F1
- Failure analysis showing where and why mistakes occur
- Tier distance metrics

```bash
tumorboard validate <GOLD_STANDARD_FILE> [OPTIONS]

Options:
  -m, --model TEXT         LLM model
  -o, --output PATH        Save detailed results
  -c, --max-concurrent N   Concurrent validations [default: 3]
```

Gold standard format: `{"entries": [{"gene": "BRAF", "variant": "V600E", "tumor_type": "Melanoma", "expected_tier": "Tier I", ...}]}`

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


## Configuration

**Supported Models:** OpenAI (gpt-4, gpt-4o, gpt-4o-mini), Anthropic (claude-3 series), Google (gemini), Azure OpenAI

**Data Sources:** MyVariant.info aggregates CIViC, ClinVar, and COSMIC databases

**Performance:** Adjust `--max-concurrent` for batch processing. GPT-4 is more accurate but expensive; gpt-4o-mini offers good balance.


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code quality guidelines.

## License & Citation

**Author:** Dami Gupta (dami.gupta@gmail.com)

**License:** MIT License

**Citation:** If you use TumorBoard in your research, please cite:

```bibtex
@software{tumorboard2025,
  author = {Gupta, Dami},
  title = {TumorBoard: LLM-Powered Cancer Variant Actionability Assessment},
  year = {2025},
  url = {https://github.com/dami-gupta-git/tumor_board_v0}
}
```

## References

- [AMP/ASCO/CAP Guidelines](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5707196/)
- [MyVariant.info](https://myvariant.info/) | [CIViC](https://civicdb.org/) | [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | [COSMIC](https://cancer.sanger.ac.uk/cosmic)

---

**Note**: This tool is for research purposes only. Clinical decisions should always be made by qualified healthcare professionals.
