import json
import re
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from loguru import logger
from config import settings

# List of known standard industries for validation
KNOWN_INDUSTRIES = [
    "Healthcare & Medical",
    "Retail & Supermarkets",
    "Banking, Financial Services & Insurance (BFSI)",
    "E-Commerce & Digital Shopping",
    "Hospitality & Hotel Management",
    "Software & SaaS",
    "Education & Academics",
    "Telecommunications",
    "Automotive & Transportation",
    "Real Estate & Construction",
    "Food & Beverage / Restaurants",
    "Travel, Aviation & Tourism",
    "Energy & Utilities",
    "Manufacturing & Industrial",
    "Media, Entertainment & Gaming",
    "Logistics & Supply Chain",
    "Professional Services & Consulting",
    "Fashion & Apparel",
    "Fitness & Wellness",
    "Government & Public Sector"
]


class CategoryGenerationError(Exception):
    """
    Raised when category generation cannot be completed via OpenAI.
    Carries user-friendly and developer-oriented diagnostic messages.
    """
    def __init__(self, user_message: str, developer_message: str, status_code: int = 503):
        super().__init__(developer_message)
        self.user_message = user_message
        self.developer_message = developer_message
        self.status_code = status_code


def validate_industry_name(industry_name: Any) -> Tuple[bool, str]:
    """
    Validates and cleans the industry name.
    Returns (is_valid, cleaned_name_or_error_message).
    """
    if not industry_name or not isinstance(industry_name, str):
        return False, "Industry name must be a non-empty string."
    
    cleaned = industry_name.strip()
    if len(cleaned) < 2:
        return False, "Industry name must be at least 2 characters long."
    
    if len(cleaned) > 100:
        return False, "Industry name must not exceed 100 characters."

    # Check for invalid control characters or script injection
    if re.search(r"[<>{}\\]", cleaned):
        return False, "Industry name contains invalid or special characters."
    
    return True, cleaned


def generate_industry_categories(
    industry_name: str,
    client_id: int,
    survey_id: int,
    category_creation_count: int = 5,
    subcategory_creation_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates category_creation_count main categories and subcategory_creation_count sub-categories per category
    tailored to the given industry using OpenAI.
    Raises CategoryGenerationError if OpenAI fails, is unconfigured, or credits are exhausted.
    """
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your") or not settings.openai_api_key.strip():
        dev_msg = "OpenAI API credits or API key is unconfigured. Please configure a valid OPENAI_API_KEY in .env."
        user_msg = "This service is currently unavailable. Please try again later."
        logger.error(f"❌ {dev_msg}")
        raise CategoryGenerationError(
            user_message=user_msg,
            developer_message=dev_msg,
            status_code=503
        )

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        prompt = f"""
You are a Customer Experience (CX) expert specializing in survey design and classification frameworks.
Generate EXACTLY {category_creation_count} main categories relevant to customer feedback for the industry: "{industry_name}".
For EACH of the {category_creation_count} main categories, generate EXACTLY {subcategory_creation_count} relevant sub-categories.

Output MUST be strictly a valid JSON object matching this schema:
{{
  "categories": [
    {{
      "category_id": 1,
      "category_name": "Main Category Title",
      "sub_categories": ["Sub Category 1", "Sub Category 2", "Sub Category 3"]
    }},
    ... (exactly {category_creation_count} items, category_id 1 to {category_creation_count})
  ]
}}

Requirements:
- Main categories must be highly relevant and distinct for the "{industry_name}" industry.
- Each main category must contain EXACTLY {subcategory_creation_count} concise sub-categories.
- category_id should be integers 1 to {category_creation_count}.
- Do NOT add markdown code blocks or explanatory text outside the JSON.
"""

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        categories = data.get("categories", [])

        # Validate structure returned by AI
        if not isinstance(categories, list) or len(categories) != category_creation_count:
            logger.warning(f"AI returned category count ({len(categories) if isinstance(categories, list) else 0}) != requested count ({category_creation_count}). Adjusting...")

        sanitized_categories = []
        for idx in range(1, category_creation_count + 1):
            cat_obj = categories[idx - 1] if isinstance(categories, list) and idx <= len(categories) and isinstance(categories[idx - 1], dict) else {}
            cat_name = str(cat_obj.get("category_name", f"Category {idx}")).strip()
            
            sub_cats = cat_obj.get("sub_categories", [])
            if not isinstance(sub_cats, list):
                sub_cats = []
            
            sanitized_sub_cats = []
            for sub_idx in range(1, subcategory_creation_count + 1):
                if sub_idx <= len(sub_cats) and sub_cats[sub_idx - 1]:
                    sanitized_sub_cats.append(str(sub_cats[sub_idx - 1]).strip())
                else:
                    sanitized_sub_cats.append(f"{cat_name} Feature {sub_idx}")

            sanitized_categories.append({
                "category_id": idx,
                "category_name": cat_name,
                "sub_categories": sanitized_sub_cats
            })

        return sanitized_categories

    except Exception as e:
        err_msg = str(e).lower()
        if any(k in err_msg for k in ["quota", "rate_limit", "429", "insufficient_quota", "credit", "billing"]):
            dev_msg = f"OpenAI API credits are exhausted or billing limit reached ({e}). Please top up your OpenAI billing account."
            user_msg = "This service is currently unavailable. Please try again later."
            logger.error(f"❌ {dev_msg}")
            raise CategoryGenerationError(
                user_message=user_msg,
                developer_message=dev_msg,
                status_code=503
            )
        else:
            dev_msg = f"OpenAI category generation failed: {e}"
            user_msg = "This service is currently unavailable. Please try again later."
            logger.error(f"❌ {dev_msg}")
            raise CategoryGenerationError(
                user_message=user_msg,
                developer_message=dev_msg,
                status_code=500
            )
