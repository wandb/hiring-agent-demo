# Pydantic
from typing import Optional, List
from pydantic import BaseModel, Field

# server.py
## extraction prompts
extract_application_prompt = """
# Role
You are an assistant tasked with extracting and organizing detailed information from a job application for comparison with a job offer.

# Instructions
Carefully analyze the job application and extract the following details. Present the output in a structured format.

# Steps
1. Extract **Applicant Details**:
   - Full name.
   - Contact information (email, phone, address).
   - LinkedIn profile or other relevant online presence (if provided).

2. Extract **Job Preferences**:
   - Desired position or role.
   - Preferred work location(s) or remote work preferences.
   - Expected salary or compensation range (if stated).
   - Availability or start date.

3. Extract **Qualifications**:
   - Educational background, including degrees, institutions, and graduation dates.
   - Certifications, licenses, or professional credentials.
   - Relevant technical skills or specialized knowledge.

4. Extract **Professional Experience**:
   - Previous job titles, companies, and employment dates.
   - Key responsibilities and achievements in each role.
   - Notable projects, accomplishments, or contributions.

5. Extract **Career Objectives**:
   - Stated career goals or aspirations.
   - Areas of interest or focus within the industry.

6. Extract **Other Information**:
   - References (if included).
   - Additional notes, personal statements, or relevant extracurricular activities.

# Output
Summarize the extracted information clearly, organized under the headings above. Highlight specific skills, qualifications, or experiences relevant to the desired position. Ensure completeness and clarity in your response.

# Candidate Application
{application}
"""

extract_offer_prompt = """
# Role
You are an assistant tasked with extracting and organizing detailed information from a job offer for comparison with a job application.

# Instructions
Carefully analyze the job offer and extract the following details. Present the output in a structured format.

# Steps
1. Extract **Company Details**:
   - Name of the company.
   - Industry and sector.
   - Location(s) of the job (including remote/hybrid options).

2. Extract **Job Role**:
   - Title of the position.
   - Department or team.
   - Key responsibilities and duties.
   - Reporting structure (e.g., to whom the role reports).

3. Extract **Compensation and Benefits**:
   - Base salary or hourly wage.
   - Bonuses or commissions (if applicable).
   - Equity, stock options, or profit-sharing opportunities.
   - Benefits package (e.g., healthcare, retirement plans, insurance).
   - Paid time off (PTO), vacation days, and sick leave.
   - Relocation assistance or perks (if any).

4. Extract **Work Environment and Schedule**:
   - Work hours (full-time, part-time, shift requirements).
   - Office, remote, or hybrid work expectations.
   - Flexibility or specific schedule details.

5. Extract **Career Development**:
   - Training programs, mentorship opportunities, or professional development.
   - Promotion and career growth potential.

6. Analyze **Application Comparison**:
   - Skills or qualifications explicitly required or preferred.
   - Experience level or certifications sought.
   - Alignment with any personal career goals or stated aspirations.

7. Include **Other Information**:
   - Start date or onboarding timeline.
   - Contract type (permanent, temporary, internship, etc.).
   - Any additional expectations or commitments (e.g., travel, overtime).

# Output
Summarize the extracted information clearly, organized under the headings above. Highlight discrepancies or alignments between the job offer and the application details. Ensure completeness and clarity in your response.

# Job Offer Details: 
{job_offer} 
"""

class CV(BaseModel):
    """Class to extract information from a CV."""

    first_name: str = Field(description="The first name of the candidate")
    last_name: str = Field(description="The last name of the candidate")
    date_of_birth: Optional[str] = Field(default=None, description="The date of birth of the candidate in YYYY-MM-DD format")
    phone: Optional[str] = Field(default=None, description="The phone number of the candidate")
    motivation_points: Optional[List[str]] = Field(
        default=None, description="List of key points describing the candidate's motivation for applying"
    )
    relevant_experience: Optional[List[str]] = Field(
        default=None, description="List of relevant previous work experiences and accomplishments"
    )
    highest_qualification: Optional[str] = Field(default=None, description="The highest educational qualification of the candidate")
    current_profession: Optional[str] = Field(default=None, description="The current profession of the candidate")
    known_programming_languages: Optional[List[str]] = Field(
        default=None, description="A list of programming languages known by the candidate, applicable for IT candidates"
    )
    num_previous_employers: Optional[int] = Field(
        default=None, description="The number of previous employers the candidate has worked with"
    )
    years_of_experience: Optional[float] = Field(
        default=None, description="The total number of years of professional experience"
    )
    salary_expectations: Optional[float] = Field(
        default=None, description="The salary expectations of the candidate in numeric format"
    )
    availability: Optional[str] = Field(
        default=None, description="The availability date of the candidate in a human-readable format"
    )


class Offer(BaseModel):
    """Class to extract information from a Job position."""

    job_title: str = Field(description="The job title")
    duration: str = Field(description="The duration of the job (limited in time or unlimited)")
    role_summary: str = Field(description="A short summary of the role")
    responsibilities: str = Field(description="The responsibilities the role entails and a good candidate should have experience with")
    requirements: str = Field(description="The required experience and skills a good candidate should have")
    offer_summary: str = Field(description="A short summary about the benefits and opportunities of the role")
    salary: Optional[float] = Field(
        default=None, description="The salary offered for the position in numeric format"
    )
    benefits: Optional[List[str]] = Field(
        default=None, description="A list of benefits offered with the position, such as flexible hours, remote work, etc."
    )
    company_info: Optional[str] = Field(
        default=None, description="Information about the company, such as size, industry, culture and values"
    )

## comparison
compare_offer_application_prompt = """
# Role
You are an assistant tasked with comparing extracted information from a job application to the details of a job offer. Your goal is to identify alignments, discrepancies, and potential areas for discussion or negotiation.

# Instructions
Carefully analyze the extracted information from both the job application and the job offer. Compare each category side-by-side and provide a detailed evaluation.

# Steps
1. **Compare Applicant Details and Job Preferences**:
   - Verify if the applicant's desired role matches the offered position.
   - Check if the applicant's preferred work location aligns with the job's location or remote/hybrid options.
   - Confirm if the applicant's expected salary aligns with the offered compensation.
   - Evaluate the alignment of the applicant's availability/start date with the job's start date.

2. **Compare Qualifications**:
   - Match the applicant's educational background and certifications to the qualifications required or preferred in the job offer.
   - Identify any additional qualifications in the application that exceed or fall short of the job's requirements.

3. **Compare Professional Experience**:
   - Assess whether the applicant's past roles and responsibilities align with the key responsibilities of the offered role.
   - Highlight any relevant projects, achievements, or skills from the application that directly align with the job offer.

4. **Compare Career Objectives**:
   - Evaluate how the applicant's stated career goals align with the growth opportunities or objectives outlined in the job offer.
   - Note any areas where the applicant's goals diverge from the role's focus.

5. **Identify Gaps and Alignments**:
   - Highlight areas where the applicant meets, exceeds, or falls short of the job requirements.
   - Note any potential for negotiation or clarification based on the applicant's expectations and the job's offerings.

# Output
Provide a detailed, category-by-category comparison in a clear and structured format. Summarize key alignments, major discrepancies, and areas for potential discussion or negotiation. Ensure that the analysis is concise, accurate, and actionable.

Job Offer Details:
{job_offer_extract}

Candidate Application:
{application_extract}

# Answer
Here is the JSON response: ```json
"""

class InterviewDecision(BaseModel):
    """Class representing a first stage hiring decision with explanation. Only considers application and job offer."""
    decision: bool = Field(description="Whether to move on to interview with the candidate or reject application")
    reason: str = Field(description="Explanation for the decision")

## guardrail
context_prompt = """
# Job Offer Details
{job_offer_extract}

# Candidate Application
{application_extract}
"""

guardrail_prompt = """
Based ONLY on the job requirements and candidate qualifications above, determine if we should proceed with an interview.
Previous response had hallucinations. Feedback: {guardrail_conclusion}
Consider ONLY factors mentioned in the documents like education, experience, skills match and availability.
"""

# for evaluate.py
reason_comp_prompt = """You are a Senior Hiring Manager at Weights & Biases. Your task is to evaluate the decision rationale provided by a Junior Hiring Manager about inviting a candidate to interview. You have:

  • A **reference reasoning** (expert senior hiring manager)  
  • A **junior reasoning** (the candidate's evaluator)  

First, use the following **Explainable 1-5 Rating Scale** to score the junior reasoning on each metric:

5 - Outstanding: Exemplary, comprehensive, perfectly aligned with W&B's mission and values.  
4 - Exceeds Standards: Strong, well-structured, minor omissions only.  
3 - Meets Standards: Acceptable, covers basics but lacks depth.  
2 - Below Standards: Superficial or incomplete, missing critical aspects.  
1 - Unsatisfactory: Fails to address core criteria, includes irrelevant or biased reasoning.  

Then, score the junior reasoning on **eight specific metrics**:

1. Role & Domain Fit  
2. Technical Rigor  
3. Values Alignment  
4. Collaboration Style  
5. Communication Clarity  
6. Fairness & Objectivity  
7. Impact Orientation  
8. Decision Consistency  

For each metric, output:
- **Score**: integer 1-5  
- **Comment**: brief justification citing evidence from the junior reasoning  

Finally, provide an overall **Pass/Fail** recommendation on whether to invite the candidate, plus a one-sentence rationale.

Do **not** output any other text or free-form feedback.

---  
Reference reasoning:  
{p1_reasoning}

Junior reasoning:  
{p2_reasoning}"""

# Old prompt
# """
# You're a senior hiring manager at an international company. 
# Your goal is to evaluate the decision making of a junior hiring manager.

# The junior hiring manager should assess whether to invite a candidate to a job interview based on a job position
# and an application provided by an applicant. To justify their decision the junior hiring manager has to provide a 
# reasoning. 

# Evaluate their reasoning by comparing it to a reference reasoning from a seasoned senior hiring manager. 
# Compare the two reasonings to make sure that the junior hiring manager followed the correct reasoning and thought of the most important points to justify their decision. 

# Reference Reasoning by senior hiring manager: 
# {p1_reasoning}

# Reasoning by junior hiring manager:
# {p2_reasoning}
# """

class MetricEvaluation(BaseModel):
    """Class representing the evaluation of a specific scoring metric."""
    score: int = Field(description="Numeric score from 1-5")
    comment: str = Field(description="Brief justification for the score")

class ReasonComparison(BaseModel):
    """Class representing a detailed evaluation of hiring reasoning."""
    role_domain_fit: MetricEvaluation = Field(description="Evaluation of Role & Domain Fit (score 1-5 and comment)")
    technical_rigor: MetricEvaluation = Field(description="Evaluation of Technical Rigor (score 1-5 and comment)")
    values_alignment: MetricEvaluation = Field(description="Evaluation of Values Alignment (score 1-5 and comment)")
    collaboration_style: MetricEvaluation = Field(description="Evaluation of Collaboration Style (score 1-5 and comment)")
    communication_clarity: MetricEvaluation = Field(description="Evaluation of Communication Clarity (score 1-5 and comment)")
    fairness_objectivity: MetricEvaluation = Field(description="Evaluation of Fairness & Objectivity (score 1-5 and comment)")
    impact_orientation: MetricEvaluation = Field(description="Evaluation of Impact Orientation (score 1-5 and comment)")
    decision_consistency: MetricEvaluation = Field(description="Evaluation of Decision Consistency (score 1-5 and comment)")
    pass_fail: str = Field(description="Overall Pass/Fail recommendation")
    rationale: str = Field(description="One-sentence rationale for the Pass/Fail recommendation")


# for generate.py
app_gen_prompt_pos = """ Job Offer Details: {job_offer} 

{applicant_context}

# Role
You are an assistant tasked with creating a tailored job application, including a CV and a cover letter, based on the details provided in a job offer. The goal is to highlight qualifications, experience, and skills that align closely with the job's requirements.

# Instructions
Using the job offer as a guide, draft a complete job application that includes the following:

1. **Cover Letter**:
   - Address the letter to the hiring manager or company (use a generic salutation if no specific name is provided).
   - Start with a compelling introduction explaining the interest in the role and the company.
   - Highlight relevant skills, qualifications, and achievements that directly align with the job description.
   - Mention specific experiences or projects that demonstrate the ability to excel in the role.
   - Conclude with a confident call to action, expressing enthusiasm for an interview and the opportunity to contribute.

2. **Curriculum Vitae (CV)**:
   - Create a professional summary or objective tailored to the role, summarizing key qualifications and career highlights.
   - List professional experience in reverse chronological order, emphasizing responsibilities and achievements that align with the job requirements.
   - Include educational background, certifications, and any relevant technical or soft skills.
   - Add sections for projects, publications, awards, or extracurricular activities if they enhance the application.
   - Ensure the CV is formatted cleanly and professionally.

# Steps
1. Analyze the job offer to extract key requirements, responsibilities, and qualifications.
2. Match these with suitable experiences, skills, and achievements to include in the application.
3. Structure the cover letter and CV clearly, ensuring they are concise, professional, and directly relevant to the role.

# Output Format
Provide the application in a structured format:
1. Cover Letter:
   - Include the salutation, body paragraphs, and a closing statement.
2. CV:
   - Professional summary.
   - Work experience (listed with job title, company, dates, and bullet points for key achievements).
   - Education (degrees, institutions, and graduation dates).
   - Skills (grouped by technical, soft, or other categories).
   - Additional sections (e.g., certifications, projects, awards) if applicable.

Ensure the tone is professional and tailored to the job offer. Make the application compelling and well-aligned with the employer's expectations.
"""


app_gen_prompt_neg = """ 
Job Offer Details: {job_offer}

{applicant_context}
    
# Role
You are an assistant tasked with creating a tailored job application, including a CV and a cover letter, based on the details provided in a job offer. However, the goal here is to create a job application that does *not* align with the job's requirements, qualifications, and experience. The applicant should lack the skills, qualifications, and experience needed for the role.

# Instructions
Using the job offer as a guide, draft a complete job application that includes the following:

1. **Cover Letter**:
   - Address the letter to the hiring manager or company (use a generic salutation if no specific name is provided).
   - Start with a generic and non-compelling introduction explaining the interest in the role, but ensure it doesn't align with the job's specifics.
   - Highlight irrelevant skills, qualifications, and achievements that don't match the job description. Include experiences or projects that are unrelated or do not demonstrate the applicant's ability to excel in the role.
   - Avoid mentioning any qualifications that are important for the role.
   - Conclude with a vague or overly general closing statement, avoiding enthusiasm for the interview or a clear expression of interest in contributing to the company.

2. **Curriculum Vitae (CV)**:
   - Create a generic professional summary that does not address the role or the qualifications required.
   - List irrelevant professional experience in reverse chronological order, emphasizing responsibilities and achievements that do not relate to the job.
   - Avoid including any relevant educational background, certifications, or technical skills that match the job offer.
   - Include irrelevant skills, such as those unrelated to the role, or include an absence of technical expertise.
   - Avoid any projects, publications, or awards that would help the application for the specific role.
   - Ensure the CV does not meet the professional standards expected for the job in terms of formatting and content.

# Steps
1. Analyze the job offer to extract key requirements, responsibilities, and qualifications, and ensure that the application lacks these entirely.
2. Ensure that the cover letter and CV are full of irrelevant or lacking qualifications, making it clear that the applicant is not suited for the position.
3. Structure the cover letter and CV clearly but ensure the content is unprofessional and misaligned with the job offer.

# Output Format
Provide the application in a structured format:
1. Cover Letter:
   - Include the salutation, body paragraphs, and a closing statement. Avoid addressing any of the key skills or qualifications required for the role.
2. CV:
   - Professional summary that does not reflect the job's needs.
   - Work experience (listed with job title, company, dates, and bullet points for non-relevant or minimal achievements).
   - Education (degrees, institutions, and graduation dates that are unrelated or irrelevant to the job).
   - Skills (grouped by irrelevant or missing categories).
   - Additional sections (e.g., irrelevant certifications or activities).

Ensure the tone is generic, lacks enthusiasm, and makes no effort to align with the employer's expectations. Make the application clearly unsuitable for the role.
"""

class SimpleApplicationGeneration(BaseModel):
    application_text: str = Field(description="A realistic mock application")
    interview: bool = Field(description="Whether this application will be proceeded to the interview stage")
    reason: str = Field(description="The reason why the interview decision was made the way it was")

class EvaluationExample(BaseModel):
    """Class representing a single evaluation example for testing the hiring agent."""
    offer_pdf: str = Field(description="Path to the job offer PDF file")
    offer_text: str = Field(description="Extracted text content from the job offer")
    application_pdf: str = Field(description="Path to the application PDF file") 
    application_text: str = Field(description="Extracted text content from the application")
    interview: bool = Field(description="Expected interview decision (True/False)")
    reason: str = Field(description="Expected reasoning for the interview decision")

class EvaluationDataset(BaseModel):
    """Class representing a collection of evaluation examples."""
    examples: List[EvaluationExample] = Field(description="List of evaluation examples for testing")