"""Variant data models."""

from pydantic import BaseModel, Field


class VariantInput(BaseModel):
    """Input for variant assessment."""

    gene: str = Field(..., description="Gene symbol (e.g., BRAF)")
    variant: str = Field(..., description="Variant notation (e.g., V600E)")
    tumor_type: str = Field(..., description="Tumor type (e.g., Melanoma)")

    def to_hgvs(self) -> str:
        """Convert to HGVS-like notation for API queries."""
        return f"{self.gene}:{self.variant}"

    class Config:
        json_schema_extra = {
            "example": {
                "gene": "BRAF",
                "variant": "V600E",
                "tumor_type": "Melanoma",
            }
        }


class Variant(VariantInput):
    """Extended variant model with additional metadata."""

    chromosome: str | None = None
    position: int | None = None
    reference: str | None = None
    alternate: str | None = None
    transcript_id: str | None = None
