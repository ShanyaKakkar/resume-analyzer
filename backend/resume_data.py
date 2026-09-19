from pathlib import Path

from details_extractor import extract_contact_details
from education_extractor import extract_education
from experience_extractor import (extract_achievements, extract_certifications,
                                  extract_experience, extract_projects)
from resume_parser import extract_links, extract_text
from section_splitter import split_sections
from skills_extractor import extract_skills


class ResumeError(Exception):
    """The file could not be read as a resume."""


def parse_resume(path):
    """Read a PDF or DOCX resume and return everything as one structured record."""
    if Path(path).suffix.lower() not in (".pdf", ".docx"):
        raise ValueError("Only PDF and DOCX files are supported")

    try:
        text = extract_text(path)
        links = extract_links(path)
    except Exception as error:
        raise ResumeError(
            "This file could not be opened. It may be damaged or password protected."
        ) from error

    sections = split_sections(text)
    contact = extract_contact_details(sections.get("header", ""), text, links)
    education = extract_education(sections.get("education", ""))
    skills = extract_skills(sections.get("skills", ""), text)
    experience = extract_experience(sections.get("experience", ""))
    projects = extract_projects(sections.get("projects", ""))
    certifications = extract_certifications(sections.get("certifications", ""))
    achievements = extract_achievements(sections.get("achievements", ""))

    # Things the student should check on the review screen
    warnings = []
    if len(text.strip()) < 50:
        warnings.append("No readable text was found. The file may be a scanned image.")
    for key, label in (("name", "a name"), ("email", "an email address"),
                       ("phone", "a phone number")):
        if not contact[key]:
            warnings.append(f"Could not find {label}.")
    if not education:
        warnings.append("Could not find any education details.")
    if not skills:
        warnings.append("Could not find any skills.")
    if not experience and not projects:
        warnings.append("Could not find any experience or projects.")

    # We deliberately do NOT copy the "personal_details" section (date of birth,
    # marital status and so on) into the record.
    return {
        "contact": contact,
        "summary": sections.get("summary"),
        "education": education,
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "achievements": achievements,
        "raw_text": text,
        "meta": {
            "file_type": Path(path).suffix.lower().lstrip("."),
            "characters": len(text),
            "sections_found": [name for name in sections if name != "header"],
            "warnings": warnings,
        },
    }
