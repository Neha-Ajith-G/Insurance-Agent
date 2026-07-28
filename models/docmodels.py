"""
Shared Pydantic models for the Smart Insurance Document Router pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Union

from pydantic import BaseModel, Field, computed_field, model_validator


class DocumentCategory(str, Enum):
    POLICY_DOCUMENTS = "POLICY_DOCUMENTS"
    PROPOSAL_UNDERWRITING = "PROPOSAL_UNDERWRITING"
    QUOTES_SALES = "QUOTES_SALES"
    PREMIUM_BILLING = "PREMIUM_BILLING"
    CLAIMS = "CLAIMS"
    ENDORSEMENTS = "ENDORSEMENTS"
    RENEWALS = "RENEWALS"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    LEGAL_COMPLIANCE = "LEGAL_COMPLIANCE"
    MEDICAL_INSURANCE = "MEDICAL_INSURANCE"
    MOTOR_INSURANCE = "MOTOR_INSURANCE"
    LIFE_INSURANCE = "LIFE_INSURANCE"
    TRAVEL_INSURANCE = "TRAVEL_INSURANCE"
    PROPERTY_INSURANCE = "PROPERTY_INSURANCE"
    OTHER = "OTHER"


class DocumentType(str, Enum):
    # Policy Documents
    POLICY_SCHEDULE = "POLICY_SCHEDULE"
    POLICY_WORDING = "POLICY_WORDING"
    CERTIFICATE_OF_INSURANCE = "CERTIFICATE_OF_INSURANCE"
    POLICY_BOND = "POLICY_BOND"

    # Proposal & Underwriting
    PROPOSAL_FORM = "PROPOSAL_FORM"
    APPLICATION_FORM = "APPLICATION_FORM"
    HEALTH_QUESTIONNAIRE = "HEALTH_QUESTIONNAIRE"
    INSPECTION_REPORT = "INSPECTION_REPORT"
    KYC_DOCUMENT = "KYC_DOCUMENT"

    # Quotes & Sales
    QUOTATION = "QUOTATION"
    PREMIUM_ILLUSTRATION = "PREMIUM_ILLUSTRATION"
    BENEFIT_ILLUSTRATION = "BENEFIT_ILLUSTRATION"
    COMPARISON_SHEET = "COMPARISON_SHEET"

    # Premium & Billing
    PREMIUM_INVOICE = "PREMIUM_INVOICE"
    PREMIUM_RECEIPT = "PREMIUM_RECEIPT"
    TAX_INVOICE = "TAX_INVOICE"

    # Claims
    CLAIM_FORM = "CLAIM_FORM"
    CLAIM_INTIMATION = "CLAIM_INTIMATION"
    SURVEY_REPORT = "SURVEY_REPORT"
    DAMAGE_ASSESSMENT_REPORT = "DAMAGE_ASSESSMENT_REPORT"
    SETTLEMENT_LETTER = "SETTLEMENT_LETTER"
    DISCHARGE_VOUCHER = "DISCHARGE_VOUCHER"

    # Endorsements
    ENDORSEMENT_CERTIFICATE = "ENDORSEMENT_CERTIFICATE"
    CHANGE_REQUEST_FORM = "CHANGE_REQUEST_FORM"
    ADDRESS_CHANGE = "ADDRESS_CHANGE"
    NOMINEE_CHANGE = "NOMINEE_CHANGE"
    SUM_INSURED_REVISION = "SUM_INSURED_REVISION"

    # Renewals
    RENEWAL_NOTICE = "RENEWAL_NOTICE"
    RENEWAL_CERTIFICATE = "RENEWAL_CERTIFICATE"
    LAPSE_NOTICE = "LAPSE_NOTICE"

    # Customer Service
    CANCELLATION_REQUEST = "CANCELLATION_REQUEST"
    FREE_LOOK_CANCELLATION_FORM = "FREE_LOOK_CANCELLATION_FORM"
    REFUND_LETTER = "REFUND_LETTER"
    REINSTATEMENT_REQUEST = "REINSTATEMENT_REQUEST"

    # Legal & Compliance
    CONSENT_FORM = "CONSENT_FORM"
    PRIVACY_NOTICE = "PRIVACY_NOTICE"
    AML_KYC_DOCUMENT = "AML_KYC_DOCUMENT"
    FATCA_CRS_DECLARATION = "FATCA_CRS_DECLARATION"

    # Medical Insurance Specific
    HOSPITAL_CASHLESS_REQUEST = "HOSPITAL_CASHLESS_REQUEST"
    PRE_AUTHORIZATION_FORM = "PRE_AUTHORIZATION_FORM"
    MEDICAL_REPORT = "MEDICAL_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    MEDICAL_BILL = "MEDICAL_BILL"
    PRESCRIPTION = "PRESCRIPTION"

    # Motor Insurance Specific
    RC_COPY = "RC_COPY"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    FIR = "FIR"
    REPAIR_ESTIMATE = "REPAIR_ESTIMATE"
    GARAGE_INVOICE = "GARAGE_INVOICE"
    SURVEYOR_REPORT = "SURVEYOR_REPORT"

    # Life Insurance Specific
    NOMINATION_FORM = "NOMINATION_FORM"
    BENEFICIARY_DECLARATION = "BENEFICIARY_DECLARATION"
    DEATH_CLAIM_FORM = "DEATH_CLAIM_FORM"
    DEATH_CERTIFICATE = "DEATH_CERTIFICATE"
    SURVIVAL_BENEFIT_FORM = "SURVIVAL_BENEFIT_FORM"

    # Travel Insurance Specific
    TRAVEL_ITINERARY = "TRAVEL_ITINERARY"
    BOARDING_PASS = "BOARDING_PASS"
    PASSPORT_COPY = "PASSPORT_COPY"
    VISA_COPY = "VISA_COPY"
    BAGGAGE_LOSS_REPORT = "BAGGAGE_LOSS_REPORT"

    # Property Insurance Specific
    PROPERTY_VALUATION_REPORT = "PROPERTY_VALUATION_REPORT"
    FIRE_INCIDENT_REPORT = "FIRE_INCIDENT_REPORT"
    INVENTORY_LIST = "INVENTORY_LIST"

    OTHER = "OTHER"


CATEGORY_TYPE_MAP: dict[DocumentCategory, list[DocumentType]] = {
    DocumentCategory.POLICY_DOCUMENTS: [
        DocumentType.POLICY_SCHEDULE, DocumentType.POLICY_WORDING,
        DocumentType.CERTIFICATE_OF_INSURANCE, DocumentType.POLICY_BOND,
    ],
    DocumentCategory.PROPOSAL_UNDERWRITING: [
        DocumentType.PROPOSAL_FORM, DocumentType.APPLICATION_FORM,
        DocumentType.HEALTH_QUESTIONNAIRE, DocumentType.INSPECTION_REPORT,
        DocumentType.KYC_DOCUMENT,
    ],
    DocumentCategory.QUOTES_SALES: [
        DocumentType.QUOTATION, DocumentType.PREMIUM_ILLUSTRATION,
        DocumentType.BENEFIT_ILLUSTRATION, DocumentType.COMPARISON_SHEET,
    ],
    DocumentCategory.PREMIUM_BILLING: [
        DocumentType.PREMIUM_INVOICE, DocumentType.PREMIUM_RECEIPT,
        DocumentType.RENEWAL_NOTICE, DocumentType.TAX_INVOICE,
    ],
    DocumentCategory.CLAIMS: [
        DocumentType.CLAIM_FORM, DocumentType.CLAIM_INTIMATION,
        DocumentType.SURVEY_REPORT, DocumentType.DAMAGE_ASSESSMENT_REPORT,
        DocumentType.SETTLEMENT_LETTER, DocumentType.DISCHARGE_VOUCHER,
    ],
    DocumentCategory.ENDORSEMENTS: [
        DocumentType.ENDORSEMENT_CERTIFICATE, DocumentType.CHANGE_REQUEST_FORM,
        DocumentType.ADDRESS_CHANGE, DocumentType.NOMINEE_CHANGE,
        DocumentType.SUM_INSURED_REVISION,
    ],
    DocumentCategory.RENEWALS: [
        DocumentType.RENEWAL_NOTICE, DocumentType.RENEWAL_CERTIFICATE,
        DocumentType.LAPSE_NOTICE,
    ],
    DocumentCategory.CUSTOMER_SERVICE: [
        DocumentType.CANCELLATION_REQUEST, DocumentType.FREE_LOOK_CANCELLATION_FORM,
        DocumentType.REFUND_LETTER, DocumentType.REINSTATEMENT_REQUEST,
    ],
    DocumentCategory.LEGAL_COMPLIANCE: [
        DocumentType.CONSENT_FORM, DocumentType.PRIVACY_NOTICE,
        DocumentType.AML_KYC_DOCUMENT, DocumentType.FATCA_CRS_DECLARATION,
    ],
    DocumentCategory.MEDICAL_INSURANCE: [
        DocumentType.HOSPITAL_CASHLESS_REQUEST, DocumentType.PRE_AUTHORIZATION_FORM,
        DocumentType.MEDICAL_REPORT, DocumentType.DISCHARGE_SUMMARY,
        DocumentType.MEDICAL_BILL, DocumentType.PRESCRIPTION,
    ],
    DocumentCategory.MOTOR_INSURANCE: [
        DocumentType.RC_COPY, DocumentType.DRIVING_LICENSE, DocumentType.FIR,
        DocumentType.REPAIR_ESTIMATE, DocumentType.GARAGE_INVOICE,
        DocumentType.SURVEYOR_REPORT,
    ],
    DocumentCategory.LIFE_INSURANCE: [
        DocumentType.NOMINATION_FORM, DocumentType.BENEFICIARY_DECLARATION,
        DocumentType.DEATH_CLAIM_FORM, DocumentType.DEATH_CERTIFICATE,
        DocumentType.SURVIVAL_BENEFIT_FORM,
    ],
    DocumentCategory.TRAVEL_INSURANCE: [
        DocumentType.TRAVEL_ITINERARY, DocumentType.BOARDING_PASS,
        DocumentType.PASSPORT_COPY, DocumentType.VISA_COPY,
        DocumentType.BAGGAGE_LOSS_REPORT,
    ],
    DocumentCategory.PROPERTY_INSURANCE: [
        DocumentType.PROPERTY_VALUATION_REPORT, DocumentType.FIRE_INCIDENT_REPORT,
        DocumentType.INVENTORY_LIST, DocumentType.REPAIR_ESTIMATE,
    ],
    DocumentCategory.OTHER: [DocumentType.OTHER],
}


class ClassificationResult(BaseModel):
    category: DocumentCategory
    document_type: DocumentType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None

    @model_validator(mode="after")
    def _check_type_matches_category(self) -> "ClassificationResult":
        if self.document_type == DocumentType.OTHER:
            return self
        allowed = CATEGORY_TYPE_MAP.get(self.category, [])
        if self.document_type not in allowed:
            raise ValueError(
                f"document_type {self.document_type.value!r} is not valid "
                f"for category {self.category.value!r}"
            )
        return self


class PolicyFields(BaseModel):
    policy_number: str | None = None
    insured_name: str | None = None
    policy_start_date: str | None = None
    policy_end_date: str | None = None
    premium: float | None = None
    sum_insured: float | None = None
    plan_name: str | None = None


class ClaimFields(BaseModel):
    claim_number: str | None = None
    policy_number: str | None = None
    claimant_name: str | None = None
    incident_date: str | None = None
    claim_amount: float | None = None
    incident_description: str | None = None


class SurveyReportFields(BaseModel):
    survey_number: str | None = None
    policy_number: str | None = None
    claim_number: str | None = None
    surveyor_name: str | None = None
    assessed_loss_amount: float | None = None
    survey_date: str | None = None


class PremiumBillingFields(BaseModel):
    document_number: str | None = None
    policy_number: str | None = None
    premium_amount: float | None = None
    currency: str | None = None
    due_date: str | None = None
    payment_date: str | None = None


class MedicalReportFields(BaseModel):
    patient_name: str | None = None
    policy_number: str | None = None
    diagnosis: str | None = None
    treating_doctor: str | None = None
    hospital_name: str | None = None
    report_date: str | None = None


class GenericInsuranceFields(BaseModel):
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)


ExtractionFields = Union[
    PolicyFields,
    ClaimFields,
    SurveyReportFields,
    PremiumBillingFields,
    MedicalReportFields,
    GenericInsuranceFields,
]

EXTRACTION_MODEL_MAP: dict[DocumentType, type[BaseModel]] = {
    DocumentType.POLICY_SCHEDULE: PolicyFields,
    DocumentType.POLICY_WORDING: PolicyFields,
    DocumentType.CERTIFICATE_OF_INSURANCE: PolicyFields,
    DocumentType.POLICY_BOND: PolicyFields,
    DocumentType.CLAIM_FORM: ClaimFields,
    DocumentType.CLAIM_INTIMATION: ClaimFields,
    DocumentType.DAMAGE_ASSESSMENT_REPORT: ClaimFields,
    DocumentType.SETTLEMENT_LETTER: ClaimFields,
    DocumentType.DISCHARGE_VOUCHER: ClaimFields,
    DocumentType.SURVEY_REPORT: SurveyReportFields,
    DocumentType.SURVEYOR_REPORT: SurveyReportFields,
    DocumentType.PREMIUM_INVOICE: PremiumBillingFields,
    DocumentType.PREMIUM_RECEIPT: PremiumBillingFields,
    DocumentType.TAX_INVOICE: PremiumBillingFields,
    DocumentType.RENEWAL_NOTICE: PremiumBillingFields,
    DocumentType.RENEWAL_CERTIFICATE: PremiumBillingFields,
    DocumentType.MEDICAL_REPORT: MedicalReportFields,
    DocumentType.DISCHARGE_SUMMARY: MedicalReportFields,
    DocumentType.HOSPITAL_CASHLESS_REQUEST: MedicalReportFields,
    DocumentType.PRE_AUTHORIZATION_FORM: MedicalReportFields,
}

DEFAULT_EXTRACTION_MODEL: type[BaseModel] = GenericInsuranceFields


class ExtractionResult(BaseModel):
    category: DocumentCategory
    document_type: DocumentType
    fields: ExtractionFields


class PipelineStage(str, Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"


class TokenStats(BaseModel):
    turn: PipelineStage
    input_tokens: int
    output_tokens: int
    total_tokens: int
    inference_time: float


class PipelineResult(BaseModel):
    file_path: str
    filename: str
    classification: ClassificationResult
    extraction: ExtractionResult
    token_stats: list[TokenStats]
    pipeline_elapsed: float

    @computed_field
    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.token_stats)

    @computed_field
    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.token_stats)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.token_stats)

    @computed_field
    @property
    def total_inference_time(self) -> float:
        return sum(s.inference_time for s in self.token_stats)


# --- Pydantic Metadata Mapping Models -----------------------------------

class InsuranceDomainMetadata(BaseModel):
    category: str = Field(default="OTHER", description="Broad category: CLAIMS, POLICY_DOCUMENTS, etc.")
    document_type: str = Field(default="OTHER", description="Specific type: CLAIM_FORM, SURVEY_REPORT, etc.")
    company_name: str | None = Field(default=None, description="Insurer / Provider name")
    insured_name: str | None = Field(default=None, description="Name of policyholder / patient / claimant")
    policy_number: str | None = Field(default=None, description="Policy reference number")
    claim_number: str | None = Field(default=None, description="Claim reference number")
    fiscal_year: str | None = Field(default=None, description="Fiscal / Tax year (e.g. 2025, FY2026)")
    effective_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    total_amount: float | None = Field(default=None, description="Financial amount (claim, premium, bill)")
    currency: str = Field(default="USD", description="Currency code (USD, INR, EUR)")


class DocumentFileMetadata(BaseModel):
    doc_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original file name")
    file_extension: str = Field(default="", description="Extension e.g. .pdf, .docx, .png")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    upload_timestamp: str = Field(default="", description="ISO 8601 UTC upload timestamp")
    user_id: int | None = Field(default=None, description="Owner user ID for access control")
    needs_ocr: bool = Field(default=False, description="True if document required OCR processing")


class PipelineAuditMetadata(BaseModel):
    llm_model: str = Field(default="qwen2.5:3b", description="Local LLM model used")
    classification_confidence: float | None = Field(default=None, description="Classifier confidence score")
    responder_verified: bool = Field(default=True, description="True if passed factual grounding check")
    grounding_score: float = Field(default=1.0, description="Grounding verification score 0.0-1.0")
    total_tokens_used: int = Field(default=0, description="Total tokens consumed across turns")
    total_inference_time_sec: float = Field(default=0.0, description="Total LLM inference time in seconds")


class EnrichedDocumentPayload(BaseModel):
    file_meta: DocumentFileMetadata
    domain_meta: InsuranceDomainMetadata
    audit_meta: PipelineAuditMetadata
    chunks_count: int = 0
