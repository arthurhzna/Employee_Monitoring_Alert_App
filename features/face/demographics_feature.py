# domain logic - Gender & Age prediction service
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DemographicsResult:
    gender: str
    gender_confidence: float
    age: str
    age_confidence: float
    category: str = "unknown" 

class DemographicsFeature:
    def __init__(
        self,
        min_confidence_gender: float = 0.5,
        min_confidence_age: float = 0.5
    ) -> None:
        self.min_confidence_gender = min_confidence_gender
        self.min_confidence_age = min_confidence_age
    
    def process_model_output(
        self,
        age_output: Dict[str, Any],
        gender_output: Dict[str, Any]
    ) -> DemographicsResult:

        category = self.get_age_gender_category(gender_output, age_output)
        
        return DemographicsResult(
            gender = gender_output.get("label", "unknown"),
            gender_confidence = gender_output.get("confidence", 0.0),
            age = age_output.get("label", "unknown"),
            age_confidence = age_output.get("confidence", 0.0),
            category = category
        )

    def get_age_gender_category(
        self, 
        gender_output: Dict[str, Any], 
        age_output: Dict[str, Any]
    ) -> str:
        gender_label = gender_output.get("label", "")
        gender_conf = gender_output.get("confidence", 0.0)
        if gender_conf < self.min_confidence_gender:
            return "unknown"
        gender = self._normalize_gender_group(gender_label)
        if gender is None:
            return "unknown"

        age_label = age_output.get("label", "")
        age_conf = age_output.get("confidence", 0.0)
        if age_conf < self.min_confidence_age or not age_label:
            return "unknown"

        age_num = self._normalize_age_group(age_label)
        return f"{gender}{age_num}"

    def _normalize_gender_group(self, label: str) -> Optional[str]:
        label_lower = label.lower()
        if label_lower in ["male", "m", "man", "men"]:
            return "m"
        if label_lower in ["female", "f", "woman", "women"]:
            return "f"
        return None

    def _normalize_age_group(self, age: str) -> int:
        if age in ("0-2", "3-9", "10-19"):
            return 0
        if age in ("20-29", "30-39"):
            return 1
        if age == "40-49":
            return 2
        return 3