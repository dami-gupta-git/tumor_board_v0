"""Core assessment engine combining API and LLM services."""

import asyncio
from tumorboard.api.myvariant import MyVariantClient
from tumorboard.llm.service import LLMService
from tumorboard.models.assessment import ActionabilityAssessment
from tumorboard.models.variant import VariantInput


class AssessmentEngine:
    """Simple engine for variant assessment."""

    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.1):
        self.myvariant_client = MyVariantClient()
        self.llm_service = LLMService(model=llm_model, temperature=llm_temperature)

    async def __aenter__(self):
        await self.myvariant_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.myvariant_client.__aexit__(exc_type, exc_val, exc_tb)

    async def assess_variant(self, variant_input: VariantInput) -> ActionabilityAssessment:
        """Assess a single variant."""

        # Fetch evidence
        evidence = await self.myvariant_client.fetch_evidence(
            gene=variant_input.gene,
            variant=variant_input.variant,
        )

        # Assess with LLM
        assessment = await self.llm_service.assess_variant(
            gene=variant_input.gene,
            variant=variant_input.variant,
            tumor_type=variant_input.tumor_type,
            evidence=evidence,
        )

        return assessment

    async def batch_assess(
        self, variants: list[VariantInput], max_concurrent: int = 5
    ) -> list[ActionabilityAssessment]:
        """Assess multiple variants."""

        tasks = [self.assess_variant(variant) for variant in variants]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        assessments = [r for r in results if not isinstance(r, Exception)]
        return assessments
