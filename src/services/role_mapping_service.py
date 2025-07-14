"""
Role Mapping Service

This service handles competency framework mapping using LLM.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

# Import settings
from src.core.config import settings

load_dotenv()
logger = logging.getLogger("uvicorn.error")


def map_role_to_competencies_gemini(
    prompt_text: str, 
    competency_framework_json: str, 
    organization: str, 
    role_title: str, 
    department: Optional[str] = None
) -> str:
    """
    Map role to competencies using Gemini LLM.
    
    Args:
        prompt_text: The prompt template for mapping
        competency_framework_json: JSON string of competency framework
        organization: Organization name
        role_title: Role title to map
        department: Optional department name
        
    Returns:
        JSON string with mapping results
    """
    logger.info("Starting Gemini LLM mapping call")
    
    try:
        # Check if Gemini API key is available
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY in your environment.")
        
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash-preview-04-17"
        
        # Prepare the user prompt
        user_prompt = prompt_text.replace("[Insert the entire competency framework JSON here]", competency_framework_json)
        user_prompt = user_prompt.replace("[organization]", organization)
        user_prompt = user_prompt.replace("[role_title]", role_title)
        user_prompt = user_prompt.replace("[department]", department or "")
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="INSERT_INPUT_HERE"),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["organization", "role_title", "mapped_competencies", "mapping_rationale"],
                properties={
                    "organization": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The name of the organization",
                    ),
                    "role_title": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The title of the role",
                    ),
                    "mapped_competencies": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        description="List of mapped competencies",
                        items=genai.types.Schema(
                            type=genai.types.Type.OBJECT,
                            required=["category", "theme", "sub_themes", "confidence"],
                            properties={
                                "category": genai.types.Schema(
                                    type=genai.types.Type.STRING,
                                    description="The category of the competency",
                                    enum=["Behavioural", "Functional", "Domain"],
                                ),
                                "theme": genai.types.Schema(
                                    type=genai.types.Type.STRING,
                                    description="The name of the competency theme",
                                ),
                                "sub_themes": genai.types.Schema(
                                    type=genai.types.Type.ARRAY,
                                    description="List of competency sub-themes",
                                    items=genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                    ),
                                ),
                                "confidence": genai.types.Schema(
                                    type=genai.types.Type.INTEGER,
                                    description="Confidence level (0 to 100) of the competency mapping for the role",
                                ),
                            },
                        ),
                    ),
                    "mapping_rationale": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Explanation of why these competencies were selected",
                    ),
                },
            ),
            system_instruction=[
                types.Part.from_text(text=user_prompt),
            ],
        )
        
        output = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output += chunk.text
            
        logger.info("Gemini LLM mapping call completed successfully.")
        return output
        
    except Exception as e:
        logger.error(f"Error during Gemini LLM call: {e}")
        raise


def map_role_to_competencies(
    organization: str, 
    role_title: str, 
    department: Optional[str] = None
) -> Dict[str, Any]:
    """
    Map a role to competencies using the competency framework.
    
    Args:
        organization: Organization name
        role_title: Role title to map
        department: Optional department name
        
    Returns:
        Dictionary with mapping results
    """
    logger.info(f"Mapping role: {role_title} at {organization}")
    
    try:
        # Default competency framework (this should ideally come from a database or config)
        competency_framework = {
            "behavioral": {
                "leadership": ["Strategic Thinking", "Team Management", "Decision Making"],
                "communication": ["Verbal Communication", "Written Communication", "Presentation Skills"],
                "collaboration": ["Teamwork", "Cross-functional Collaboration", "Stakeholder Management"]
            },
            "functional": {
                "technical": ["Domain Expertise", "Technical Skills", "Tool Proficiency"],
                "analytical": ["Data Analysis", "Problem Solving", "Critical Thinking"],
                "project_management": ["Planning", "Execution", "Risk Management"]
            },
            "domain": {
                "industry_knowledge": ["Market Understanding", "Regulatory Knowledge", "Best Practices"],
                "business_acumen": ["Financial Literacy", "Strategy", "Operations"]
            }
        }
        
        # Default prompt template
        prompt_template = f"""
        You are an expert in competency mapping. Based on the provided competency framework, 
        map the most relevant competencies for the role of {role_title} at {organization}.
        
        Organization: {organization}
        Role: {role_title}
        Department: {department or 'Not specified'}
        
        Competency Framework: {json.dumps(competency_framework, indent=2)}
        
        Please analyze this role and return the top 5-8 most relevant competencies with confidence scores.
        """
        
        # Call Gemini for mapping
        result = map_role_to_competencies_gemini(
            prompt_template,
            json.dumps(competency_framework),
            organization,
            role_title,
            department
        )
        
        # Parse the result
        mapping_data = json.loads(result)
        
        return {
            "status": "success",
            "message": "Role mapping completed successfully",
            "data": mapping_data
        }
        
    except Exception as e:
        logger.error(f"Error in role mapping service: {str(e)}")
        return {
            "status": "error",
            "message": f"Role mapping error: {str(e)}",
            "data": None
        }
