"""Tests for data models."""

import pytest
from pydantic import ValidationError

from tumorboard.models.assessment import ActionabilityAssessment, ActionabilityTier, RecommendedTherapy
from tumorboard.models.evidence import CIViCEvidence, Evidence
from tumorboard.models.validation import GoldStandardEntry, ValidationMetrics, ValidationResult
from tumorboard.models.variant import VariantInput


class TestVariantInput:
    """Tests for VariantInput model."""

    def test_variant_input_creation(self):
        """Test creating a variant input."""
        variant = VariantInput(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
        )
        assert variant.gene == "BRAF"
        assert variant.variant == "V600E"
        assert variant.tumor_type == "Melanoma"

    def test_to_hgvs(self):
        """Test HGVS conversion."""
        variant = VariantInput(gene="BRAF", variant="V600E", tumor_type="Melanoma")
        assert variant.to_hgvs() == "BRAF:V600E"


class TestEvidence:
    """Tests for Evidence models."""

    def test_civic_evidence_creation(self):
        """Test creating CIViC evidence."""
        civic = CIViCEvidence(
            evidence_type="Predictive",
            evidence_level="A",
            clinical_significance="Sensitivity/Response",
            disease="Melanoma",
            drugs=["Vemurafenib"],
            description="Test evidence",
        )
        assert civic.evidence_type == "Predictive"
        assert "Vemurafenib" in civic.drugs

    def test_evidence_has_evidence(self):
        """Test has_evidence method."""
        evidence = Evidence(
            variant_id="BRAF:V600E",
            gene="BRAF",
            variant="V600E",
        )
        assert not evidence.has_evidence()

        evidence.civic = [CIViCEvidence(evidence_type="Predictive")]
        assert evidence.has_evidence()


class TestActionabilityAssessment:
    """Tests for ActionabilityAssessment model."""

    def test_assessment_creation(self):
        """Test creating an assessment."""
        assessment = ActionabilityAssessment(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            tier=ActionabilityTier.TIER_I,
            confidence_score=0.95,
            summary="Test summary",
            rationale="Test rationale",
        )
        assert assessment.tier == ActionabilityTier.TIER_I
        assert assessment.confidence_score == 0.95

    def test_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            ActionabilityAssessment(
                gene="BRAF",
                variant="V600E",
                tumor_type="Melanoma",
                tier=ActionabilityTier.TIER_I,
                confidence_score=1.5,  # Invalid
                summary="Test",
                rationale="Test",
            )

    def test_to_report(self):
        """Test report generation."""
        assessment = ActionabilityAssessment(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            tier=ActionabilityTier.TIER_I,
            confidence_score=0.95,
            summary="Test summary",
            rationale="Test rationale",
            recommended_therapies=[
                RecommendedTherapy(
                    drug_name="Vemurafenib",
                    evidence_level="FDA-approved",
                )
            ],
        )
        report = assessment.to_report()
        assert "BRAF V600E" in report
        assert "Tier I" in report
        assert "Vemurafenib" in report


class TestValidationModels:
    """Tests for validation models."""

    def test_gold_standard_entry(self):
        """Test gold standard entry creation."""
        entry = GoldStandardEntry(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            expected_tier=ActionabilityTier.TIER_I,
        )
        assert entry.expected_tier == ActionabilityTier.TIER_I

    def test_validation_result_tier_distance(self):
        """Test tier distance calculation."""
        assessment = ActionabilityAssessment(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            tier=ActionabilityTier.TIER_II,
            confidence_score=0.8,
            summary="Test",
            rationale="Test",
        )

        result = ValidationResult(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            expected_tier=ActionabilityTier.TIER_I,
            predicted_tier=ActionabilityTier.TIER_II,
            is_correct=False,
            confidence_score=0.8,
            assessment=assessment,
        )

        assert result.tier_distance == 1

    def test_tier_metrics_calculation(self):
        """Test tier metrics calculation."""
        from tumorboard.models.validation import TierMetrics

        metrics = TierMetrics(
            tier=ActionabilityTier.TIER_I,
            true_positives=8,
            false_positives=2,
            false_negatives=1,
        )
        metrics.calculate()

        assert metrics.precision == 0.8  # 8/(8+2)
        assert metrics.recall == 8 / 9  # 8/(8+1)
        assert metrics.f1_score > 0

    def test_validation_metrics(self, sample_gold_standard_entry):
        """Test validation metrics calculation."""
        # Create mock results
        assessment = ActionabilityAssessment(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            tier=ActionabilityTier.TIER_I,
            confidence_score=0.95,
            summary="Test",
            rationale="Test",
        )

        result = ValidationResult(
            gene="BRAF",
            variant="V600E",
            tumor_type="Melanoma",
            expected_tier=ActionabilityTier.TIER_I,
            predicted_tier=ActionabilityTier.TIER_I,
            is_correct=True,
            confidence_score=0.95,
            assessment=assessment,
        )

        metrics = ValidationMetrics()
        metrics.calculate([result])

        assert metrics.total_cases == 1
        assert metrics.correct_predictions == 1
        assert metrics.accuracy == 1.0
