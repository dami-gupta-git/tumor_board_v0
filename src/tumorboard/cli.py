"""Command-line interface for TumorBoard."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from tumorboard.engine import AssessmentEngine
from tumorboard.models.variant import VariantInput
from tumorboard.validation.validator import Validator

# Load environment variables from .env file
load_dotenv()

# Create Typer app
app = typer.Typer(
    name="tumorboard",
    help="LLM-powered cancer variant actionability assessment with validation",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@app.command()
def assess(
    gene: str = typer.Argument(..., help="Gene symbol (e.g., BRAF)"),
    variant: str = typer.Argument(..., help="Variant notation (e.g., V600E)"),
    tumor: str = typer.Option(..., "--tumor", "-t", help="Tumor type (e.g., Melanoma)"),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LLM model to use (e.g., gpt-4, claude-3-sonnet-20240229)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Assess clinical actionability of a single variant.

    Example:
        tumorboard assess BRAF V600E --tumor "Melanoma"
    """
    setup_logging(verbose)

    async def run_assessment() -> None:
        """Run the assessment asynchronously."""
        variant_input = VariantInput(gene=gene, variant=variant, tumor_type=tumor)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                description=f"Assessing {gene} {variant} in {tumor}...",
                total=None,
            )

            async with AssessmentEngine(llm_model=model) as engine:
                try:
                    assessment = await engine.assess_variant(variant_input)

                    # Display report
                    console.print("\n")
                    console.print(assessment.to_report())
                    console.print("\n")

                    # Save to file if requested
                    if output:
                        output_data = assessment.model_dump(mode="json")
                        with open(output, "w") as f:
                            json.dump(output_data, f, indent=2)
                        console.print(f"[green]Saved to {output}[/green]")

                except Exception as e:
                    console.print(f"[red]Error: {str(e)}[/red]")
                    raise typer.Exit(1)

    asyncio.run(run_assessment())


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="Input JSON file with variants"),
    output: Path = typer.Option(
        "results.json",
        "--output",
        "-o",
        help="Output JSON file path",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LLM model to use",
    ),
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        "-c",
        help="Maximum concurrent assessments",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Batch process multiple variants from a JSON file.

    Input JSON format:
    [
        {"gene": "BRAF", "variant": "V600E", "tumor_type": "Melanoma"},
        {"gene": "EGFR", "variant": "L858R", "tumor_type": "Lung Adenocarcinoma"}
    ]

    Example:
        tumorboard batch variants.json --output results.json
    """
    setup_logging(verbose)

    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    async def run_batch() -> None:
        """Run batch assessment asynchronously."""
        # Load variants from file
        try:
            with open(input_file, "r") as f:
                data = json.load(f)

            variants = [VariantInput(**item) for item in data]
            console.print(f"[blue]Loaded {len(variants)} variants from {input_file}[/blue]\n")

        except Exception as e:
            console.print(f"[red]Error loading variants: {str(e)}[/red]")
            raise typer.Exit(1)

        # Run batch assessment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                description=f"Assessing {len(variants)} variants...",
                total=len(variants),
            )

            async with AssessmentEngine(llm_model=model) as engine:
                try:
                    assessments = await engine.batch_assess(
                        variants,
                        max_concurrent=max_concurrent,
                    )

                    progress.update(task, completed=len(variants))

                    # Save results
                    output_data = [assessment.model_dump(mode="json") for assessment in assessments]
                    with open(output, "w") as f:
                        json.dump(output_data, f, indent=2)

                    console.print(
                        f"\n[green]Successfully assessed {len(assessments)}/{len(variants)} "
                        f"variants[/green]"
                    )
                    console.print(f"[green]Results saved to {output}[/green]\n")

                    # Summary statistics
                    tier_counts = {}
                    for assessment in assessments:
                        tier = assessment.tier.value
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1

                    console.print("[bold]Tier Distribution:[/bold]")
                    for tier, count in sorted(tier_counts.items()):
                        console.print(f"  {tier}: {count}")

                except Exception as e:
                    console.print(f"[red]Error during batch assessment: {str(e)}[/red]")
                    raise typer.Exit(1)

    asyncio.run(run_batch())


@app.command()
def validate(
    gold_standard: Path = typer.Argument(..., help="Gold standard JSON file"),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LLM model to use",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for detailed results",
    ),
    max_concurrent: int = typer.Option(
        3,
        "--max-concurrent",
        "-c",
        help="Maximum concurrent validations",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Validate LLM assessments against gold standard dataset.

    Gold standard JSON format:
    {
      "entries": [
        {
          "gene": "BRAF",
          "variant": "V600E",
          "tumor_type": "Melanoma",
          "expected_tier": "Tier I",
          "notes": "FDA-approved for melanoma",
          "references": ["PMID:12345"]
        }
      ]
    }

    Example:
        tumorboard validate benchmarks/gold_standard.json
    """
    setup_logging(verbose)

    if not gold_standard.exists():
        console.print(f"[red]Error: Gold standard file not found: {gold_standard}[/red]")
        raise typer.Exit(1)

    async def run_validation() -> None:
        """Run validation asynchronously."""
        async with AssessmentEngine(llm_model=model) as engine:
            validator = Validator(engine)

            try:
                # Load gold standard
                entries = validator.load_gold_standard(gold_standard)
                console.print(f"[blue]Loaded {len(entries)} gold standard entries[/blue]\n")

                # Run validation
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        description="Running validation...",
                        total=len(entries),
                    )

                    metrics = await validator.validate_dataset(
                        entries,
                        max_concurrent=max_concurrent,
                    )

                    progress.update(task, completed=len(entries))

                # Display report
                console.print("\n")
                console.print(metrics.to_report())
                console.print("\n")

                # Save detailed results if requested
                if output:
                    output_data = metrics.model_dump(mode="json")
                    with open(output, "w") as f:
                        json.dump(output_data, f, indent=2)
                    console.print(f"[green]Detailed results saved to {output}[/green]")

            except Exception as e:
                console.print(f"[red]Error during validation: {str(e)}[/red]")
                raise typer.Exit(1)

    asyncio.run(run_validation())


@app.command()
def version() -> None:
    """Show version information."""
    from tumorboard import __version__

    console.print(f"TumorBoard version {__version__}")


if __name__ == "__main__":
    app()
