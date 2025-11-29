"""Validation and benchmarking models."""

from pydantic import BaseModel, Field

from tumorboard.models.assessment import ActionabilityAssessment, ActionabilityTier


class GoldStandardEntry(BaseModel):
    """Gold standard entry for validation."""

    gene: str
    variant: str
    tumor_type: str
    expected_tier: ActionabilityTier
    notes: str | None = None
    references: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of validating a single assessment against gold standard."""

    gene: str
    variant: str
    tumor_type: str
    expected_tier: ActionabilityTier
    predicted_tier: ActionabilityTier
    is_correct: bool
    confidence_score: float
    assessment: ActionabilityAssessment

    @property
    def tier_distance(self) -> int:
        """Calculate distance between predicted and expected tier (0-3)."""
        tier_order = {
            ActionabilityTier.TIER_I: 0,
            ActionabilityTier.TIER_II: 1,
            ActionabilityTier.TIER_III: 2,
            ActionabilityTier.TIER_IV: 3,
            ActionabilityTier.UNKNOWN: -1,
        }
        expected_idx = tier_order.get(self.expected_tier, -1)
        predicted_idx = tier_order.get(self.predicted_tier, -1)

        if expected_idx == -1 or predicted_idx == -1:
            return 999  # Unknown tier
        return abs(expected_idx - predicted_idx)


class TierMetrics(BaseModel):
    """Metrics for a specific tier."""

    tier: ActionabilityTier
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def calculate(self) -> None:
        """Calculate precision, recall, and F1 score."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0

        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0

        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


class ValidationMetrics(BaseModel):
    """Overall validation metrics."""

    total_cases: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    average_confidence: float = 0.0
    tier_metrics: dict[str, TierMetrics] = Field(default_factory=dict)
    failure_analysis: list[dict[str, str]] = Field(default_factory=list)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result and update metrics."""
        self.total_cases += 1

        if result.is_correct:
            self.correct_predictions += 1

        # Update tier-specific metrics
        expected_key = result.expected_tier.value
        predicted_key = result.predicted_tier.value

        # Initialize tier metrics if needed
        for tier in [result.expected_tier, result.predicted_tier]:
            if tier.value not in self.tier_metrics:
                self.tier_metrics[tier.value] = TierMetrics(tier=tier)

        # Update confusion matrix counts
        if result.is_correct:
            self.tier_metrics[expected_key].true_positives += 1
        else:
            self.tier_metrics[expected_key].false_negatives += 1
            self.tier_metrics[predicted_key].false_positives += 1

            # Add to failure analysis
            self.failure_analysis.append(
                {
                    "variant": f"{result.gene} {result.variant}",
                    "tumor_type": result.tumor_type,
                    "expected": result.expected_tier.value,
                    "predicted": result.predicted_tier.value,
                    "tier_distance": str(result.tier_distance),
                    "confidence": f"{result.confidence_score:.2%}",
                    "summary": result.assessment.summary[:200] + "..."
                    if len(result.assessment.summary) > 200
                    else result.assessment.summary,
                }
            )

    def calculate(self, results: list[ValidationResult]) -> None:
        """Calculate overall metrics from results."""
        if not results:
            return

        # Add all results
        for result in results:
            self.add_result(result)

        # Calculate overall metrics
        if self.total_cases > 0:
            self.accuracy = self.correct_predictions / self.total_cases

        total_confidence = sum(r.confidence_score for r in results)
        self.average_confidence = total_confidence / len(results) if results else 0.0

        # Calculate per-tier metrics
        for metrics in self.tier_metrics.values():
            metrics.calculate()

    def to_report(self) -> str:
        """Generate a formatted validation report."""
        lines = [
            "=" * 80,
            "VALIDATION REPORT",
            "=" * 80,
            f"\nTotal Cases: {self.total_cases}",
            f"Correct Predictions: {self.correct_predictions}",
            f"Overall Accuracy: {self.accuracy:.2%}",
            f"Average Confidence: {self.average_confidence:.2%}",
            f"\n{'-' * 80}",
            "PER-TIER METRICS",
            f"{'-' * 80}",
        ]

        # Sort tiers in order
        tier_order = [
            ActionabilityTier.TIER_I,
            ActionabilityTier.TIER_II,
            ActionabilityTier.TIER_III,
            ActionabilityTier.TIER_IV,
        ]

        for tier in tier_order:
            if tier.value in self.tier_metrics:
                metrics = self.tier_metrics[tier.value]
                lines.append(f"\n{tier.value}:")
                lines.append(f"  Precision: {metrics.precision:.2%}")
                lines.append(f"  Recall: {metrics.recall:.2%}")
                lines.append(f"  F1 Score: {metrics.f1_score:.2%}")
                lines.append(
                    f"  TP: {metrics.true_positives}, "
                    f"FP: {metrics.false_positives}, "
                    f"FN: {metrics.false_negatives}"
                )

        if self.failure_analysis:
            lines.append(f"\n{'-' * 80}")
            lines.append(f"FAILURE ANALYSIS ({len(self.failure_analysis)} errors)")
            lines.append(f"{'-' * 80}")
            for idx, failure in enumerate(self.failure_analysis[:10], 1):  # Show top 10
                lines.append(f"\n{idx}. {failure['variant']} in {failure['tumor_type']}")
                lines.append(
                    f"   Expected: {failure['expected']} | "
                    f"Predicted: {failure['predicted']} | "
                    f"Distance: {failure['tier_distance']}"
                )
                lines.append(f"   Confidence: {failure['confidence']}")
                lines.append(f"   Summary: {failure['summary']}")

            if len(self.failure_analysis) > 10:
                lines.append(f"\n... and {len(self.failure_analysis) - 10} more errors")

        lines.append(f"\n{'=' * 80}")
        return "\n".join(lines)
