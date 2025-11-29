"""Tests for LLM service."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from tumorboard.llm.service import LLMService, LLMServiceError
from tumorboard.models.assessment import ActionabilityTier


class TestLLMService:
    """Tests for LLMService."""

    def test_parse_json_response(self):
        """Test JSON parsing from LLM response."""
        service = LLMService()

        # Test plain JSON
        response = '{"tier": "Tier I", "confidence_score": 0.95}'
        parsed = service._parse_json_response(response)
        assert parsed["tier"] == "Tier I"

        # Test with markdown code blocks
        response = '```json\n{"tier": "Tier I", "confidence_score": 0.95}\n```'
        parsed = service._parse_json_response(response)
        assert parsed["tier"] == "Tier I"

        # Test with just backticks
        response = '```\n{"tier": "Tier I", "confidence_score": 0.95}\n```'
        parsed = service._parse_json_response(response)
        assert parsed["tier"] == "Tier I"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        service = LLMService()

        with pytest.raises(LLMServiceError):
            service._parse_json_response("not valid json")

    def test_validate_and_create_assessment(self, mock_llm_response):
        """Test validation and assessment creation."""
        service = LLMService()

        response_data = json.loads(mock_llm_response)
        assessment = service._validate_and_create_assessment(
            response_data=response_data,
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
        )

        assert assessment.tier == ActionabilityTier.TIER_I
        assert assessment.confidence_score == 0.95
        assert len(assessment.recommended_therapies) == 2
        assert assessment.recommended_therapies[0].drug_name == "Vemurafenib"

    def test_validate_invalid_tier(self):
        """Test handling invalid tier."""
        service = LLMService()

        response_data = {
            "tier": "Invalid Tier",
            "confidence_score": 0.5,
            "summary": "Test",
            "rationale": "Test",
        }

        # Should default to UNKNOWN for invalid tier
        assessment = service._validate_and_create_assessment(
            response_data=response_data,
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
        )

        assert assessment.tier == ActionabilityTier.UNKNOWN

    @pytest.mark.asyncio
    async def test_assess_variant(self, sample_evidence, mock_llm_response):
        """Test variant assessment."""
        service = LLMService()

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_llm_response

            assessment = await service.assess_variant(
                gene="BRAF",
                variant="V600E",
                tumor_type="Melanoma",
                evidence=sample_evidence,
            )

            assert assessment.tier == ActionabilityTier.TIER_I
            assert assessment.gene == "BRAF"
            assert assessment.variant == "V600E"
            assert assessment.confidence_score == 0.95

    @pytest.mark.asyncio
    async def test_llm_api_error(self, sample_evidence):
        """Test LLM API error handling."""
        service = LLMService()

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = LLMServiceError("API error")

            with pytest.raises(LLMServiceError):
                await service.assess_variant(
                    gene="BRAF",
                    variant="V600E",
                    tumor_type="Melanoma",
                    evidence=sample_evidence,
                )

    @pytest.mark.asyncio
    async def test_batch_assess(self, sample_evidence, mock_llm_response):
        """Test batch assessment."""
        service = LLMService()

        variants = [
            ("BRAF", "V600E", "Melanoma", sample_evidence),
            ("EGFR", "L858R", "Lung Cancer", sample_evidence),
        ]

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_llm_response

            assessments = await service.batch_assess(variants)

            assert len(assessments) == 2
            assert all(a.tier == ActionabilityTier.TIER_I for a in assessments)
