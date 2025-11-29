"""Core assessment engine combining API and LLM services."""

import logging
from typing import Any

from tumorboard.api.myvariant import MyVariantClient
from tumorboard.llm.service import LLMService
from tumorboard.models.assessment import ActionabilityAssessment
from tumorboard.models.evidence import Evidence
from tumorboard.models.variant import VariantInput

logger = logging.getLogger(__name__)


class AssessmentEngine:
    """Core engine for variant actionability assessment.

    Combines evidence gathering from MyVariant.info and
    LLM-based assessment.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.1,
        api_timeout: float = 30.0,
    ) -> None:
        """Initialize the assessment engine.

        Args:
            llm_model: LLM model to use for assessment
            llm_temperature: Temperature for LLM sampling
            api_timeout: Timeout for API requests
        """
        self.myvariant_client = MyVariantClient(timeout=api_timeout)
        self.llm_service = LLMService(
            model=llm_model,
            temperature=llm_temperature,
        )

    async def __aenter__(self) -> "AssessmentEngine":
        """Async context manager entry."""
        await self.myvariant_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.myvariant_client.__aexit__(exc_type, exc_val, exc_tb)

    async def assess_variant(self, variant_input: VariantInput) -> ActionabilityAssessment:
        """Assess a single variant.

        Args:
            variant_input: Variant information

        Returns:
            Complete actionability assessment

        Raises:
            Exception: If assessment fails
        """
        logger.info(
            f"Assessing variant: {variant_input.gene} {variant_input.variant} "
            f"in {variant_input.tumor_type}"
        )

        # Step 1: Fetch evidence from MyVariant.info
        logger.info("Fetching evidence from MyVariant.info...")
        evidence = await self.myvariant_client.fetch_evidence(
            gene=variant_input.gene,
            variant=variant_input.variant,
        )

        if not evidence.has_evidence():
            logger.warning("No evidence found in databases")
        else:
            logger.info(
                f"Found evidence: CIViC={len(evidence.civic)}, "
                f"ClinVar={len(evidence.clinvar)}, "
                f"COSMIC={len(evidence.cosmic)}"
            )

        # Step 2: Assess with LLM
        logger.info("Generating LLM assessment...")
        assessment = await self.llm_service.assess_variant(
            gene=variant_input.gene,
            variant=variant_input.variant,
            tumor_type=variant_input.tumor_type,
            evidence=evidence,
        )

        logger.info(
            f"Assessment complete: {assessment.tier.value} "
            f"(confidence: {assessment.confidence_score:.1%})"
        )

        return assessment

    async def batch_assess(
        self,
        variants: list[VariantInput],
        max_concurrent: int = 5,
    ) -> list[ActionabilityAssessment]:
        """Assess multiple variants in batch.

        Args:
            variants: List of variant inputs
            max_concurrent: Maximum concurrent assessments

        Returns:
            List of assessments
        """
        import asyncio

        logger.info(f"Starting batch assessment of {len(variants)} variants")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def assess_with_semaphore(variant: VariantInput) -> ActionabilityAssessment:
            async with semaphore:
                return await self.assess_variant(variant)

        # Execute all assessments with concurrency limit
        tasks = [assess_with_semaphore(variant) for variant in variants]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        assessments = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to assess variant {idx}: {str(result)}")
            else:
                assessments.append(result)

        logger.info(
            f"Batch assessment complete: {len(assessments)}/{len(variants)} successful"
        )

        return assessments

    async def close(self) -> None:
        """Close all clients."""
        await self.myvariant_client.close()
