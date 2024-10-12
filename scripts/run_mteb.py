"""
Run the MTEB benchmarks for all selected models.

TODO: fix all this with version issues
"""

# imports
from pathlib import Path

# packages
import mteb
import polars

# packages
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

# imports


def load_model(model_id: str) -> tuple[AutoModel, PreTrainedTokenizerFast]:
    """
    Load the model and tokenizer for the given model identifier.

    Args:
        model_id (str): The model identifier.

    Returns:
        tuple[AutoModel, PreTrainedTokenizerFast]: The model and tokenizer.
    """
    # load the model and tokenizer
    if "-onnx" in model_id:
        model = ORTModelForFeatureExtraction.from_pretrained(model_id).to("cpu")
    else:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def run_mteb(model_id: str, task_names: list[str]) -> list[dict]:
    """
    Run the selected model and tasks and return a simplified dictionary of results.

    Args:
        model_id (str): The model identifier.
        task_names (list[str]): The list of task names.

    Returns:
        list[dict]: The list of results
    """
    # load the model
    model = SentenceTransformer(model_id, trust_remote_code=True, device="cpu")

    # track results across all tasks
    # NOTE: we run them separately as some tasks throw errors/exceptions
    results = []
    for task in task_names:
        try:
            evaluation = mteb.MTEB(
                tasks=mteb.get_tasks(tasks=[task], languages=["eng"])
            )
            mteb_results = evaluation.run(model, verbosity=0)  # type: ignore

            for result in mteb_results:
                results.append(
                    {
                        "model_name": model_id,
                        "task_name": result.task_name,
                        "score": result.get_score(),
                    }
                )
        except Exception as e:
            print(f"Error running task {task}: {e}")

    # free the gpu ram
    model = None

    return results


DEFAULT_TASKS = [
    "CUADAffiliateLicenseLicenseeLegalBenchClassification",
    "CUADAffiliateLicenseLicensorLegalBenchClassification",
    "CUADAntiAssignmentLegalBenchClassification",
    "CUADAuditRightsLegalBenchClassification",
    "CUADCapOnLiabilityLegalBenchClassification",
    "CUADChangeOfControlLegalBenchClassification",
    "CUADCompetitiveRestrictionExceptionLegalBenchClassification",
    "CUADCovenantNotToSueLegalBenchClassification",
    "CUADEffectiveDateLegalBenchClassification",
    "CUADExclusivityLegalBenchClassification",
    "CUADExpirationDateLegalBenchClassification",
    "CUADGoverningLawLegalBenchClassification",
    "CUADIPOwnershipAssignmentLegalBenchClassification",
    "CUADInsuranceLegalBenchClassification",
    "CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
    "CUADJointIPOwnershipLegalBenchClassification",
    "CUADLicenseGrantLegalBenchClassification",
    "CUADLiquidatedDamagesLegalBenchClassification",
    "CUADMinimumCommitmentLegalBenchClassification",
    "CUADMostFavoredNationLegalBenchClassification",
    "CUADNoSolicitOfCustomersLegalBenchClassification",
    "CUADNoSolicitOfEmployeesLegalBenchClassification",
    "CUADNonCompeteLegalBenchClassification",
    "CUADNonDisparagementLegalBenchClassification",
    "CUADNonTransferableLicenseLegalBenchClassification",
    "CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
    "CUADPostTerminationServicesLegalBenchClassification",
    "CUADPriceRestrictionsLegalBenchClassification",
    "CUADRenewalTermLegalBenchClassification",
    "CUADRevenueProfitSharingLegalBenchClassification",
    "CUADRofrRofoRofnLegalBenchClassification",
    "CUADSourceCodeEscrowLegalBenchClassification",
    "CUADTerminationForConvenienceLegalBenchClassification",
    "CUADThirdPartyBeneficiaryLegalBenchClassification",
    "CUADUncappedLiabilityLegalBenchClassification",
    "CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
    "CUADVolumeRestrictionLegalBenchClassification",
    "CUADWarrantyDurationLegalBenchClassification",
    "CanadaTaxCourtOutcomesLegalBenchClassification",
    "ContractNLIConfidentialityOfAgreementLegalBenchClassification",
    "ContractNLIExplicitIdentificationLegalBenchClassification",
    "ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
    "ContractNLILimitedUseLegalBenchClassification",
    "ContractNLINoLicensingLegalBenchClassification",
    "ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
    "ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
    "ContractNLIPermissibleCopyLegalBenchClassification",
    "ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
    "ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
    "ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
    "ContractNLISharingWithEmployeesLegalBenchClassification",
    "ContractNLISharingWithThirdPartiesLegalBenchClassification",
    "ContractNLISurvivalOfObligationsLegalBenchClassification",
    "CorporateLobbyingLegalBenchClassification",
    "DefinitionClassificationLegalBenchClassification",
    "Diversity1LegalBenchClassification",
    "Diversity2LegalBenchClassification",
    "Diversity3LegalBenchClassification",
    "Diversity4LegalBenchClassification",
    "Diversity5LegalBenchClassification",
    "Diversity6LegalBenchClassification",
    "FunctionOfDecisionSectionLegalBenchClassification",
    "InsurancePolicyInterpretationLegalBenchClassification",
    "InternationalCitizenshipQuestionsLegalBenchClassification",
    "JCrewBlockerLegalBenchClassification",
    "LearnedHandsBenefitsLegalBenchClassification",
    "LearnedHandsBusinessLegalBenchClassification",
    "LearnedHandsConsumerLegalBenchClassification",
    "LearnedHandsCourtsLegalBenchClassification",
    "LearnedHandsCrimeLegalBenchClassification",
    "LearnedHandsDivorceLegalBenchClassification",
    "LearnedHandsDomesticViolenceLegalBenchClassification",
    "LearnedHandsEducationLegalBenchClassification",
    "LearnedHandsEmploymentLegalBenchClassification",
    "LearnedHandsEstatesLegalBenchClassification",
    "LearnedHandsFamilyLegalBenchClassification",
    "LearnedHandsHealthLegalBenchClassification",
    "LearnedHandsHousingLegalBenchClassification",
    "LearnedHandsImmigrationLegalBenchClassification",
    "LearnedHandsTortsLegalBenchClassification",
    "LearnedHandsTrafficLegalBenchClassification",
    "LegalReasoningCausalityLegalBenchClassification",
    "MAUDLegalBenchClassification",
    "NYSJudicialEthicsLegalBenchClassification",
    "OPP115DataRetentionLegalBenchClassification",
    "OPP115DataSecurityLegalBenchClassification",
    "OPP115DoNotTrackLegalBenchClassification",
    "OPP115FirstPartyCollectionUseLegalBenchClassification",
    "OPP115InternationalAndSpecificAudiencesLegalBenchClassification",
    "OPP115PolicyChangeLegalBenchClassification",
    "OPP115ThirdPartySharingCollectionLegalBenchClassification",
    "OPP115UserAccessEditAndDeletionLegalBenchClassification",
    "OPP115UserChoiceControlLegalBenchClassification",
    "OralArgumentQuestionPurposeLegalBenchClassification",
    "OverrulingLegalBenchClassification",
    "PROALegalBenchClassification",
    "PersonalJurisdictionLegalBenchClassification",
    "SCDBPAccountabilityLegalBenchClassification",
    "SCDBPAuditsLegalBenchClassification",
    "SCDBPCertificationLegalBenchClassification",
    "SCDBPTrainingLegalBenchClassification",
    "SCDBPVerificationLegalBenchClassification",
    "SCDDAccountabilityLegalBenchClassification",
    "SCDDAuditsLegalBenchClassification",
    "SCDDCertificationLegalBenchClassification",
    "SCDDTrainingLegalBenchClassification",
    "SCDDVerificationLegalBenchClassification",
    "TelemarketingSalesRuleLegalBenchClassification",
    "TextualismToolDictionariesLegalBenchClassification",
    "TextualismToolPlainLegalBenchClassification",
    "UCCVCommonLawLegalBenchClassification",
    "UnfairTOSLegalBenchClassification",
]

DEFAULT_MODELS = [
    "alea-institute/kl3m-embedding-001",
    "alea-institute/kl3m-embedding-002",
    "alea-institute/kl3m-embedding-003",
    "alea-institute/kl3m-embedding-004",
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "nlpaueb/legal-bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v3",
    "mixedbread-ai/mxbai-embed-large-v1",
]

if __name__ == "__main__":
    all_results = []
    for model in DEFAULT_MODELS:
        try:
            all_results.extend(run_mteb(model, DEFAULT_TASKS))
        except Exception as e:
            print(f"Error running model {model}: {e}")

    # get output data path
    data_path = Path(__file__).parent.parent / "data"
    if not data_path.exists():
        data_path.mkdir(exist_ok=True)

    # convert to polars df, sort by task name and score, then output to csv
    df = polars.DataFrame(all_results)
    df = df.sort("task_name", "score")
    df.write_csv(data_path / "mteb_results.csv")
