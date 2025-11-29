"""MyVariant.info API client for fetching variant evidence."""

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tumorboard.models.evidence import (
    CIViCEvidence,
    ClinVarEvidence,
    COSMICEvidence,
    Evidence,
)


class MyVariantAPIError(Exception):
    """Exception raised for MyVariant API errors."""

    pass


class MyVariantClient:
    """Client for MyVariant.info API.

    MyVariant.info aggregates variant annotations from multiple sources
    including CIViC, ClinVar, COSMIC, and more.
    """

    BASE_URL = "https://myvariant.info/v1"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        """Initialize the MyVariant client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MyVariantClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _query(self, query: str, fields: list[str] | None = None) -> dict[str, Any]:
        """Execute a query against MyVariant API.

        Args:
            query: Query string (e.g., "BRAF:V600E" or "chr7:140453136")
            fields: Specific fields to retrieve

        Returns:
            API response as dictionary

        Raises:
            MyVariantAPIError: If the API request fails
        """
        client = self._get_client()
        params: dict[str, str] = {"q": query}

        if fields:
            params["fields"] = ",".join(fields)

        try:
            response = await client.get(f"{self.BASE_URL}/query", params=params)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise MyVariantAPIError(f"API error: {data['error']}")

            return data

        except httpx.HTTPStatusError as e:
            raise MyVariantAPIError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.TimeoutException:
            raise MyVariantAPIError(f"Request timed out after {self.timeout}s")
        except httpx.HTTPError as e:
            raise MyVariantAPIError(f"HTTP error: {str(e)}")
        except Exception as e:
            raise MyVariantAPIError(f"Unexpected error: {str(e)}")

    async def get_variant(self, variant_id: str) -> dict[str, Any]:
        """Get variant by ID.

        Args:
            variant_id: Variant identifier (HGVS, dbSNP, etc.)

        Returns:
            Variant data
        """
        client = self._get_client()

        try:
            response = await client.get(f"{self.BASE_URL}/variant/{variant_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise MyVariantAPIError(f"Failed to fetch variant: {str(e)}")

    def _parse_civic_evidence(self, civic_data: dict[str, Any] | list[Any]) -> list[CIViCEvidence]:
        """Parse CIViC data into evidence objects.

        Args:
            civic_data: Raw CIViC data from API

        Returns:
            List of CIViC evidence objects
        """
        evidence_list: list[CIViCEvidence] = []

        # Handle both single dict and list of dicts
        items = civic_data if isinstance(civic_data, list) else [civic_data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # CIViC can have nested evidence items
            if "evidence_items" in item:
                for ev_item in item.get("evidence_items", []):
                    evidence_list.append(
                        CIViCEvidence(
                            evidence_type=ev_item.get("evidence_type"),
                            evidence_level=ev_item.get("evidence_level"),
                            evidence_direction=ev_item.get("evidence_direction"),
                            clinical_significance=ev_item.get("clinical_significance"),
                            disease=ev_item.get("disease", {}).get("name")
                            if isinstance(ev_item.get("disease"), dict)
                            else None,
                            drugs=[
                                drug.get("name", "")
                                for drug in ev_item.get("drugs", [])
                                if isinstance(drug, dict)
                            ],
                            description=ev_item.get("description"),
                            source=ev_item.get("source", {}).get("name")
                            if isinstance(ev_item.get("source"), dict)
                            else None,
                            rating=ev_item.get("rating"),
                        )
                    )
            else:
                # Direct evidence object
                evidence_list.append(
                    CIViCEvidence(
                        evidence_type=item.get("evidence_type"),
                        evidence_level=item.get("evidence_level"),
                        evidence_direction=item.get("evidence_direction"),
                        clinical_significance=item.get("clinical_significance"),
                        disease=item.get("disease"),
                        drugs=item.get("drugs", []) if isinstance(item.get("drugs"), list) else [],
                        description=item.get("description"),
                        source=item.get("source"),
                        rating=item.get("rating"),
                    )
                )

        return evidence_list

    def _parse_clinvar_evidence(
        self, clinvar_data: dict[str, Any] | list[Any]
    ) -> list[ClinVarEvidence]:
        """Parse ClinVar data into evidence objects.

        Args:
            clinvar_data: Raw ClinVar data from API

        Returns:
            List of ClinVar evidence objects
        """
        evidence_list: list[ClinVarEvidence] = []

        # Handle both single dict and list of dicts
        items = clinvar_data if isinstance(clinvar_data, list) else [clinvar_data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Extract clinical significance
            clin_sig = item.get("clinical_significance")
            if isinstance(clin_sig, list):
                clin_sig = ", ".join(str(s) for s in clin_sig)

            # Extract conditions
            conditions = []
            if "conditions" in item:
                cond_data = item["conditions"]
                if isinstance(cond_data, list):
                    for cond in cond_data:
                        if isinstance(cond, dict):
                            conditions.append(cond.get("name", ""))
                        else:
                            conditions.append(str(cond))
                elif isinstance(cond_data, dict):
                    conditions.append(cond_data.get("name", ""))

            evidence_list.append(
                ClinVarEvidence(
                    clinical_significance=str(clin_sig) if clin_sig else None,
                    review_status=item.get("review_status"),
                    conditions=conditions,
                    last_evaluated=item.get("last_evaluated"),
                    variation_id=str(item.get("variation_id")) if "variation_id" in item else None,
                )
            )

        return evidence_list

    def _parse_cosmic_evidence(
        self, cosmic_data: dict[str, Any] | list[Any]
    ) -> list[COSMICEvidence]:
        """Parse COSMIC data into evidence objects.

        Args:
            cosmic_data: Raw COSMIC data from API

        Returns:
            List of COSMIC evidence objects
        """
        evidence_list: list[COSMICEvidence] = []

        # Handle both single dict and list of dicts
        items = cosmic_data if isinstance(cosmic_data, list) else [cosmic_data]

        for item in items:
            if not isinstance(item, dict):
                continue

            evidence_list.append(
                COSMICEvidence(
                    mutation_id=item.get("mutation_id"),
                    primary_site=item.get("primary_site"),
                    site_subtype=item.get("site_subtype"),
                    primary_histology=item.get("primary_histology"),
                    histology_subtype=item.get("histology_subtype"),
                    sample_count=item.get("sample_count"),
                    mutation_somatic_status=item.get("mutation_somatic_status"),
                )
            )

        return evidence_list

    async def fetch_evidence(self, gene: str, variant: str) -> Evidence:
        """Fetch evidence for a variant from multiple sources.

        Args:
            gene: Gene symbol (e.g., "BRAF")
            variant: Variant notation (e.g., "V600E")

        Returns:
            Aggregated evidence from all sources

        Raises:
            MyVariantAPIError: If the API request fails
        """
        query = f"{gene}:{variant}"

        # Request specific fields from CIViC, ClinVar, and COSMIC
        fields = [
            "civic",
            "clinvar",
            "cosmic",
            "dbsnp",
            "cadd",
        ]

        try:
            result = await self._query(query, fields=fields)

            # Extract hits (MyVariant returns a list of hits)
            hits = result.get("hits", [])

            if not hits:
                # No data found, return empty evidence
                return Evidence(
                    variant_id=query,
                    gene=gene,
                    variant=variant,
                    raw_data=result,
                )

            # Use the first hit (most relevant)
            hit_data = hits[0] if hits else {}

            # Parse evidence from each source
            civic_evidence = []
            if "civic" in hit_data:
                civic_evidence = self._parse_civic_evidence(hit_data["civic"])

            clinvar_evidence = []
            if "clinvar" in hit_data:
                clinvar_evidence = self._parse_clinvar_evidence(hit_data["clinvar"])

            cosmic_evidence = []
            if "cosmic" in hit_data:
                cosmic_evidence = self._parse_cosmic_evidence(hit_data["cosmic"])

            return Evidence(
                variant_id=hit_data.get("_id", query),
                gene=gene,
                variant=variant,
                civic=civic_evidence,
                clinvar=clinvar_evidence,
                cosmic=cosmic_evidence,
                raw_data=hit_data,
            )

        except MyVariantAPIError:
            raise
        except Exception as e:
            raise MyVariantAPIError(f"Failed to parse evidence: {str(e)}")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
