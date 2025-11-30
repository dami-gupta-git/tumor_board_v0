"""MyVariant.info API client for fetching variant evidence.

ARCHITECTURE:
    Gene + Variant → MyVariant.info API → Evidence (CIViC/ClinVar/COSMIC)

Aggregates variant information from multiple databases for LLM assessment.

Key Design:
- Async HTTP with connection pooling (httpx.AsyncClient)
- Retry with exponential backoff (tenacity)
- Structured parsing to typed Evidence models
- Context manager for session cleanup
"""

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
        # Request specific fields from CIViC, ClinVar, COSMIC, and identifiers
        fields = [
            "civic",
            "clinvar",
            "cosmic",
            "dbsnp",
            "cadd",
            "entrezgene",  # NCBI Gene ID
            "cosmic.cosmic_id",  # COSMIC mutation ID
            "clinvar.variant_id",  # ClinVar variation ID
            "clinvar.clinical_significance",  # ClinVar clinical significance
            "clinvar.accession",  # ClinVar accession
            "dbsnp.rsid",  # dbSNP rs number
            "hgvs",  # HGVS notations (genomic, protein, transcript)
            "snpeff",  # SnpEff effect prediction
            "dbnsfp.polyphen2.hdiv.pred",  # PolyPhen2 prediction
            "dbnsfp.cadd.phred",  # CADD phred score
            "gnomad_exome.af.af",  # gnomAD exome allele frequency
            "vcf.alt",  # VCF alternative allele
            "vcf.ref",  # VCF reference allele
        ]

        try:
            # Try multiple query strategies to find the variant
            # Strategy 1: Gene with protein notation (e.g., "BRAF p.V600E")
            # This works best with MyVariant API
            protein_notation = f"p.{variant}" if not variant.startswith("p.") else variant
            query = f"{gene} {protein_notation}"
            result = await self._query(query, fields=fields)

            # Strategy 2: If no hits, try simple gene:variant (e.g., "BRAF:V600E")
            if result.get("total", 0) == 0:
                query = f"{gene}:{variant}"
                result = await self._query(query, fields=fields)

            # Strategy 3: If still no hits, try searching by gene name and variant without prefix
            if result.get("total", 0) == 0:
                query = f"{gene} {variant}"
                result = await self._query(query, fields=fields)

            # Extract hits (MyVariant returns a list of hits)
            hits = result.get("hits", [])

            if not hits:
                # No data found, return empty evidence
                return Evidence(
                    variant_id=query,
                    gene=gene,
                    variant=variant,
                    cosmic_id=None,
                    ncbi_gene_id=None,
                    dbsnp_id=None,
                    clinvar_id=None,
                    clinvar_clinical_significance=None,
                    clinvar_accession=None,
                    hgvs_genomic=None,
                    hgvs_protein=None,
                    hgvs_transcript=None,
                    snpeff_effect=None,
                    polyphen2_prediction=None,
                    cadd_score=None,
                    gnomad_exome_af=None,
                    transcript_id=None,
                    transcript_consequence=None,
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

            # Extract database identifiers
            cosmic_id = None
            if "cosmic" in hit_data:
                cosmic_data = hit_data["cosmic"]
                if isinstance(cosmic_data, dict):
                    cosmic_id = cosmic_data.get("cosmic_id")
                elif isinstance(cosmic_data, list) and cosmic_data:
                    cosmic_id = cosmic_data[0].get("cosmic_id") if isinstance(cosmic_data[0], dict) else None

            ncbi_gene_id = None
            # Try multiple places for gene ID
            if "entrezgene" in hit_data:
                ncbi_gene_id = str(hit_data["entrezgene"])
            elif "dbsnp" in hit_data and isinstance(hit_data["dbsnp"], dict):
                gene_info = hit_data["dbsnp"].get("gene")
                if isinstance(gene_info, dict) and "geneid" in gene_info:
                    ncbi_gene_id = str(gene_info["geneid"])

            dbsnp_id = None
            if "dbsnp" in hit_data:
                dbsnp_data = hit_data["dbsnp"]
                if isinstance(dbsnp_data, dict):
                    rsid = dbsnp_data.get("rsid")
                    if rsid:
                        dbsnp_id = f"rs{rsid}" if not str(rsid).startswith("rs") else str(rsid)

            clinvar_id = None
            clinvar_clinical_significance = None
            clinvar_accession = None
            if "clinvar" in hit_data:
                clinvar_data = hit_data["clinvar"]
                if isinstance(clinvar_data, dict):
                    clinvar_id = str(clinvar_data.get("variant_id")) if "variant_id" in clinvar_data else None
                    # Extract clinical significance
                    clin_sig = clinvar_data.get("clinical_significance")
                    if clin_sig:
                        if isinstance(clin_sig, list):
                            clinvar_clinical_significance = ", ".join(str(s) for s in clin_sig)
                        else:
                            clinvar_clinical_significance = str(clin_sig)
                    # Extract accession
                    accession = clinvar_data.get("accession")
                    if accession:
                        clinvar_accession = str(accession)
                elif isinstance(clinvar_data, list) and clinvar_data:
                    if isinstance(clinvar_data[0], dict):
                        if "variant_id" in clinvar_data[0]:
                            clinvar_id = str(clinvar_data[0]["variant_id"])
                        clin_sig = clinvar_data[0].get("clinical_significance")
                        if clin_sig:
                            if isinstance(clin_sig, list):
                                clinvar_clinical_significance = ", ".join(str(s) for s in clin_sig)
                            else:
                                clinvar_clinical_significance = str(clin_sig)
                        accession = clinvar_data[0].get("accession")
                        if accession:
                            clinvar_accession = str(accession)

            # Extract HGVS notations
            hgvs_genomic = None
            hgvs_protein = None
            hgvs_transcript = None

            # The variant _id is often in HGVS genomic format (e.g., "chr7:g.140453136A>T")
            variant_id = hit_data.get("_id", query)
            if variant_id and (variant_id.startswith("chr") or variant_id.startswith("NC_")):
                hgvs_genomic = variant_id

            if "hgvs" in hit_data:
                hgvs_data = hit_data["hgvs"]
                if isinstance(hgvs_data, str):
                    # Single HGVS notation
                    if hgvs_data.startswith("chr") or hgvs_data.startswith("NC_"):
                        hgvs_genomic = hgvs_data
                    elif ":p." in hgvs_data:
                        hgvs_protein = hgvs_data
                    elif ":c." in hgvs_data:
                        hgvs_transcript = hgvs_data
                elif isinstance(hgvs_data, list):
                    # Multiple HGVS notations
                    for hgvs in hgvs_data:
                        if isinstance(hgvs, str):
                            if (hgvs.startswith("chr") or hgvs.startswith("NC_")) and not hgvs_genomic:
                                hgvs_genomic = hgvs
                            elif ":p." in hgvs and not hgvs_protein:
                                hgvs_protein = hgvs
                            elif ":c." in hgvs and not hgvs_transcript:
                                hgvs_transcript = hgvs

            # Try to extract protein notation from CIViC or other sources
            if not hgvs_protein and "civic" in hit_data:
                civic_data = hit_data["civic"]
                if isinstance(civic_data, dict) and "name" in civic_data:
                    name = civic_data["name"]
                    if ":p." in name:
                        hgvs_protein = name

            # Extract functional annotations
            snpeff_effect = None
            if "snpeff" in hit_data:
                snpeff_data = hit_data["snpeff"]
                if isinstance(snpeff_data, dict):
                    ann = snpeff_data.get("ann")
                    if isinstance(ann, dict):
                        snpeff_effect = ann.get("effect")
                    elif isinstance(ann, list) and ann:
                        if isinstance(ann[0], dict):
                            snpeff_effect = ann[0].get("effect")

            polyphen2_prediction = None
            if "dbnsfp" in hit_data:
                dbnsfp_data = hit_data["dbnsfp"]
                if isinstance(dbnsfp_data, dict):
                    polyphen2_data = dbnsfp_data.get("polyphen2", {})
                    if isinstance(polyphen2_data, dict):
                        hdiv = polyphen2_data.get("hdiv", {})
                        if isinstance(hdiv, dict):
                            polyphen2_prediction = hdiv.get("pred")

            cadd_score = None
            if "dbnsfp" in hit_data:
                dbnsfp_data = hit_data["dbnsfp"]
                if isinstance(dbnsfp_data, dict):
                    cadd_data = dbnsfp_data.get("cadd", {})
                    if isinstance(cadd_data, dict):
                        phred = cadd_data.get("phred")
                        if phred is not None:
                            try:
                                cadd_score = float(phred)
                            except (ValueError, TypeError):
                                pass
            # Also check top-level cadd field
            if cadd_score is None and "cadd" in hit_data:
                cadd_data = hit_data["cadd"]
                if isinstance(cadd_data, dict):
                    phred = cadd_data.get("phred")
                    if phred is not None:
                        try:
                            cadd_score = float(phred)
                        except (ValueError, TypeError):
                            pass

            gnomad_exome_af = None
            if "gnomad_exome" in hit_data:
                gnomad_data = hit_data["gnomad_exome"]
                if isinstance(gnomad_data, dict):
                    af_data = gnomad_data.get("af", {})
                    if isinstance(af_data, dict):
                        af = af_data.get("af")
                        if af is not None:
                            try:
                                gnomad_exome_af = float(af)
                            except (ValueError, TypeError):
                                pass

            # Extract transcript information
            transcript_id = None
            transcript_consequence = None
            if "snpeff" in hit_data:
                snpeff_data = hit_data["snpeff"]
                if isinstance(snpeff_data, dict):
                    ann = snpeff_data.get("ann")
                    if isinstance(ann, dict):
                        transcript_id = ann.get("feature_id")
                        transcript_consequence = ann.get("effect")
                    elif isinstance(ann, list) and ann:
                        if isinstance(ann[0], dict):
                            transcript_id = ann[0].get("feature_id")
                            transcript_consequence = ann[0].get("effect")

            return Evidence(
                variant_id=hit_data.get("_id", query),
                gene=gene,
                variant=variant,
                cosmic_id=cosmic_id,
                ncbi_gene_id=ncbi_gene_id,
                dbsnp_id=dbsnp_id,
                clinvar_id=clinvar_id,
                clinvar_clinical_significance=clinvar_clinical_significance,
                clinvar_accession=clinvar_accession,
                hgvs_genomic=hgvs_genomic,
                hgvs_protein=hgvs_protein,
                hgvs_transcript=hgvs_transcript,
                snpeff_effect=snpeff_effect,
                polyphen2_prediction=polyphen2_prediction,
                cadd_score=cadd_score,
                gnomad_exome_af=gnomad_exome_af,
                transcript_id=transcript_id,
                transcript_consequence=transcript_consequence,
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
