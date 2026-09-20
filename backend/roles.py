"""Match a list of skills to tech job roles using skill profiles.

This is rule-based on purpose: each role has a list of typical skills that WE chose
(edit them freely), and the score is how much of that profile the resume covers.
Nothing is learned from data, so it needs no training and always explains itself."""
from skills_extractor import SKILLS

# core skills count double; extra skills count once
ROLE_PROFILES = {
    "Data Scientist / ML Engineer": {
        "core": ["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "Statistics"],
        "extra": ["Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Analysis",
                  "Data Visualization", "Matplotlib", "Jupyter Notebook", "NLP", "Computer Vision"],
    },
    "Data Analyst": {
        "core": ["SQL", "Excel", "Data Analysis", "Data Visualization"],
        "extra": ["Power BI", "Tableau", "Python", "Pandas", "Statistics", "Matplotlib", "MySQL"],
    },
    "Backend Developer": {
        "core": ["REST API", "SQL"],
        "extra": ["Python", "Java", "Node.js", "Django", "Flask", "FastAPI", "Spring Boot",
                  "Express.js", "MySQL", "PostgreSQL", "MongoDB", "Docker", "Git", "Microservices"],
    },
    "Frontend Developer": {
        "core": ["HTML", "CSS", "JavaScript"],
        "extra": ["React", "Angular", "Vue.js", "TypeScript", "Next.js", "Bootstrap",
                  "Tailwind CSS", "jQuery", "Git", "Figma"],
    },
    "Full-Stack Developer": {
        "core": ["JavaScript", "REST API", "SQL"],
        "extra": ["HTML", "CSS", "React", "Node.js", "Express.js", "MongoDB", "MySQL",
                  "Django", "Flask", "Git", "Docker"],
    },
    "Mobile App Developer": {
        "core": ["Android", "Flutter", "React Native", "Kotlin"],
        "extra": ["Java", "Swift", "Dart", "Firebase", "REST API", "Git"],
    },
    "DevOps / Cloud Engineer": {
        "core": ["Docker", "Linux", "CI/CD", "AWS"],
        "extra": ["Kubernetes", "Jenkins", "Terraform", "Azure", "Google Cloud", "GitHub Actions",
                  "Bash", "Nginx", "Git"],
    },
    "Software Engineer (general)": {
        "core": ["Data Structures and Algorithms", "OOP"],
        "extra": ["Java", "C++", "Python", "System Design", "DBMS", "Operating Systems",
                  "Computer Networks", "Git", "Design Patterns"],
    },
    "AI / LLM Engineer": {
        "core": ["Python", "Generative AI", "LLM"],
        "extra": ["NLP", "Hugging Face", "PyTorch", "TensorFlow", "Deep Learning", "Streamlit",
                  "FastAPI", "Machine Learning"],
    },
    "QA / Test Engineer": {
        "core": ["Selenium", "Pytest", "JUnit"],
        "extra": ["Postman", "Agile", "Jira", "Python", "Java", "SQL", "Git"],
    },
}

# catch typing mistakes early: every name must exist in the skills list
_unknown = {s for p in ROLE_PROFILES.values() for group in p.values() for s in group} - set(SKILLS)
if _unknown:
    raise ValueError(f"Skills in ROLE_PROFILES that are not in SKILLS: {sorted(_unknown)}")

MIN_MATCH = 0.25


def suggest_roles(skills, top_k=3):
    """Roles whose skill profile the given skills cover best.
    [{'role', 'match' (0-1), 'matched_skills', 'missing_core'}, ...]"""
    have = set(skills)
    results = []
    for role, profile in ROLE_PROFILES.items():
        core, extra = profile["core"], profile["extra"]
        total = 2 * len(core) + len(extra)
        got = 2 * sum(s in have for s in core) + sum(s in have for s in extra)
        match = got / total
        if match >= MIN_MATCH:
            results.append({
                "role": role,
                "match": round(match, 2),
                "matched_skills": [s for s in core + extra if s in have],
                "missing_core": [s for s in core if s not in have],
            })
    return sorted(results, key=lambda r: r["match"], reverse=True)[:top_k]
