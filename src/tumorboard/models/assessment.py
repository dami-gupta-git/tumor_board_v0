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
    tumor_type: str
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

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        return round(v, 3)

    def to_report(self) -> str:
        """Generate a formatted report."""
        lines = [
            "=" * 80,
            "VARIANT ACTIONABILITY ASSESSMENT REPORT",
            "=" * 80,
            f"\nVariant: {self.gene} {self.variant}",
            f"Tumor Type: {self.tumor_type}",
            f"\nTier: {self.tier.value}",
            f"Confidence: {self.confidence_score:.1%}",
            f"Evidence Strength: {self.evidence_strength or 'Not specified'}",
            f"\n{'-' * 80}",
            f"SUMMARY\n{'-' * 80}",
            self.summary,
            f"\n{'-' * 80}",
            f"RATIONALE\n{'-' * 80}",
            self.rationale,
        ]

        if self.recommended_therapies:
            lines.append(f"\n{'-' * 80}")
            lines.append(f"RECOMMENDED THERAPIES ({len(self.recommended_therapies)})")
            lines.append(f"{'-' * 80}")
            for idx, therapy in enumerate(self.recommended_therapies, 1):
                lines.append(f"\n{idx}. {therapy.drug_name}")
                if therapy.evidence_level:
                    lines.append(f"   Evidence Level: {therapy.evidence_level}")
                if therapy.approval_status:
                    lines.append(f"   Approval Status: {therapy.approval_status}")
                if therapy.clinical_context:
                    lines.append(f"   Clinical Context: {therapy.clinical_context}")

        if self.clinical_trials_available:
            lines.append(f"\n{'-' * 80}")
            lines.append("Clinical trials may be available for this variant.")

        if self.references:
            lines.append(f"\n{'-' * 80}")
            lines.append(f"KEY REFERENCES ({len(self.references)})")
            lines.append(f"{'-' * 80}")
            for idx, ref in enumerate(self.references, 1):
                lines.append(f"{idx}. {ref}")

        lines.append(f"\n{'=' * 80}")
        return "\n".join(lines)
