"""Assessment and actionability models."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ActionabilityTier(str, Enum):
    """AMP/ASCO/CAP clinical actionability tiers.

    Tier I: Variants with strong clinical significance
    Tier II: Variants with potential clinical significance
    Tier III: Variants with unknown clinical significance
    Tier IV: Variants deemed benign or likely benign
    """

    TIER_I = "Tier I"
    TIER_II = "Tier II"
    TIER_III = "Tier III"
    TIER_IV = "Tier IV"
    UNKNOWN = "Unknown"


class RecommendedTherapy(BaseModel):
    """Recommended therapy based on variant."""

    drug_name: str = Field(..., description="Name of the therapeutic agent")
    evidence_level: str | None = Field(None, description="Level of supporting evidence")
    approval_status: str | None = Field(None, description="FDA approval status for this indication")
    clinical_context: str | None = Field(
        None, description="Clinical context (e.g., first-line, resistant)"
    )


class ActionabilityAssessment(BaseModel):
    """Complete actionability assessment for a variant."""

    gene: str
    variant: str
    tumor_type: str | None
    tier: ActionabilityTier = Field(..., description="AMP/ASCO/CAP tier classification")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the assessment (0-1)"
    )
    summary: str = Field(..., description="Human-readable summary of the assessment")
    recommended_therapies: list[RecommendedTherapy] = Field(default_factory=list)
    rationale: str = Field(..., description="Detailed rationale for tier assignment")
    evidence_strength: str | None = Field(
        None, description="Overall strength of evidence (Strong/Moderate/Weak)"
    )
    clinical_trials_available: bool = Field(
        default=False, description="Whether relevant clinical trials exist"
    )
    references: list[str] = Field(
        default_factory=list, description="Key references supporting the assessment"
    )

    def to_report(self) -> str:
        """Simple report output."""
        tumor_display = self.tumor_type if self.tumor_type else "Not specified"
        report = f"\nVariant: {self.gene} {self.variant} | Tumor: {tumor_display}\n"
        report += f"Tier: {self.tier.value} | Confidence: {self.confidence_score:.1%}\n\n"
        report += f"{self.summary}\n"

        if self.recommended_therapies:
            report += f"\nTherapies: {', '.join([t.drug_name for t in self.recommended_therapies])}\n"

        return report
