"""LLM service for variant actionability assessment."""

import json
import logging
from typing import Any

from litellm import acompletion
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tumorboard.llm.prompts import ACTIONABILITY_SYSTEM_PROMPT, create_assessment_prompt
from tumorboard.models.assessment import ActionabilityAssessment, ActionabilityTier
from tumorboard.models.evidence import Evidence

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Exception raised for LLM service errors."""

    pass


class LLMService:
    """Service for LLM-based variant assessment using litellm.

    Supports multiple LLM providers through litellm abstraction.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LLM service.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet", "gpt-4o-mini")
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM API with retry logic.

        Args:
            messages: List of message dictionaries

        Returns:
            LLM response content

        Raises:
            LLMServiceError: If the API call fails
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract content from response
            content = response.choices[0].message.content

            if not content:
                raise LLMServiceError("Empty response from LLM")

            return content

        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise LLMServiceError(f"LLM API call failed: {str(e)}")

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            LLMServiceError: If JSON parsing fails
        """
        # Remove markdown code blocks if present
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:]  # Remove ```json
        elif response.startswith("```"):
            response = response[3:]  # Remove ```

        if response.endswith("```"):
            response = response[:-3]  # Remove trailing ```

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response}")
            raise LLMServiceError(f"Invalid JSON response from LLM: {str(e)}")

    def _validate_and_create_assessment(
        self,
        response_data: dict[str, Any],
        gene: str,
        variant: str,
        tumor_type: str,
    ) -> ActionabilityAssessment:
        """Validate LLM response and create ActionabilityAssessment.

        Args:
            response_data: Parsed JSON response from LLM
            gene: Gene symbol
            variant: Variant notation
            tumor_type: Tumor type

        Returns:
            Validated ActionabilityAssessment

        Raises:
            LLMServiceError: If validation fails
        """
        try:
            # Ensure required fields are present
            if "tier" not in response_data:
                raise LLMServiceError("Missing required field: tier")

            # Parse tier string to enum
            tier_str = response_data["tier"]
            try:
                tier = ActionabilityTier(tier_str)
            except ValueError:
                logger.warning(f"Invalid tier '{tier_str}', defaulting to Unknown")
                tier = ActionabilityTier.UNKNOWN

            # Create assessment with validated data
            assessment = ActionabilityAssessment(
                gene=gene,
                variant=variant,
                tumor_type=tumor_type,
                tier=tier,
                confidence_score=response_data.get("confidence_score", 0.5),
                summary=response_data.get("summary", "No summary provided"),
                rationale=response_data.get("rationale", "No rationale provided"),
                evidence_strength=response_data.get("evidence_strength"),
                clinical_trials_available=response_data.get("clinical_trials_available", False),
                recommended_therapies=response_data.get("recommended_therapies", []),
                references=response_data.get("references", []),
            )

            return assessment

        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise LLMServiceError(f"Failed to validate LLM response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in validation: {str(e)}")
            raise LLMServiceError(f"Failed to create assessment: {str(e)}")

    async def assess_variant(
        self,
        gene: str,
        variant: str,
        tumor_type: str,
        evidence: Evidence,
    ) -> ActionabilityAssessment:
        """Assess variant actionability using LLM.

        Args:
            gene: Gene symbol
            variant: Variant notation
            tumor_type: Tumor type
            evidence: Aggregated evidence from databases

        Returns:
            ActionabilityAssessment with tier, therapies, and rationale

        Raises:
            LLMServiceError: If assessment fails
        """
        # Generate evidence summary
        evidence_summary = evidence.summary()

        # Create prompt
        user_prompt = create_assessment_prompt(
            gene=gene,
            variant=variant,
            tumor_type=tumor_type,
            evidence_summary=evidence_summary,
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": ACTIONABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM
        logger.info(f"Assessing {gene} {variant} in {tumor_type}")
        response = await self._call_llm(messages)

        # Parse response
        response_data = self._parse_json_response(response)

        # Validate and create assessment
        assessment = self._validate_and_create_assessment(
            response_data=response_data,
            gene=gene,
            variant=variant,
            tumor_type=tumor_type,
        )

        logger.info(
            f"Assessment complete: {assessment.tier.value} "
            f"(confidence: {assessment.confidence_score:.2%})"
        )

        return assessment

    async def batch_assess(
        self,
        variants: list[tuple[str, str, str, Evidence]],
    ) -> list[ActionabilityAssessment]:
        """Assess multiple variants in batch.

        Args:
            variants: List of (gene, variant, tumor_type, evidence) tuples

        Returns:
            List of ActionabilityAssessments
        """
        import asyncio

        tasks = [
            self.assess_variant(gene, variant, tumor_type, evidence)
            for gene, variant, tumor_type, evidence in variants
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)
