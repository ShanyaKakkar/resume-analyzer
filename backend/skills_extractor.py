import re

# Skill name -> the different ways people write it (all lowercase).
# Add more whenever a resume shows a skill that is missing.
SKILLS = {
    # Programming languages
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Golang": ["golang"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Scala": ["scala"],
    "Rust": ["rust"],
    "Dart": ["dart"],
    "MATLAB": ["matlab"],
    "Bash": ["bash", "shell scripting"],
    "SQL": ["sql"],
    "PL/SQL": ["pl/sql", "plsql"],
    # Web
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angularjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "springboot"],
    "Spring": ["spring framework"],
    "Bootstrap": ["bootstrap"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "jQuery": ["jquery"],
    "REST API": ["rest api", "rest apis", "restful api", "restful apis", "restful"],
    "GraphQL": ["graphql"],
    ".NET": [".net", "dotnet", "asp.net"],
    # Databases
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb"],
    "SQLite": ["sqlite"],
    "Oracle": ["oracle"],
    "SQL Server": ["sql server", "mssql"],
    "Redis": ["redis"],
    "Firebase": ["firebase"],
    "NoSQL": ["nosql"],
    "Elasticsearch": ["elasticsearch"],
    # Data science and AI
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Generative AI": ["generative ai", "genai"],
    "LLM": ["llm", "llms", "large language models"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data visualisation"],
    "Statistics": ["statistics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "PyTorch": ["pytorch"],
    "OpenCV": ["opencv"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "XGBoost": ["xgboost"],
    "Random Forest": ["random forest"],
    "Hugging Face": ["hugging face", "huggingface"],
    "Streamlit": ["streamlit"],
    "BeautifulSoup": ["beautifulsoup", "beautiful soup"],
    "Jupyter Notebook": ["jupyter", "jupyter notebook"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "ms excel", "microsoft excel"],
    # Cloud and DevOps
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Google Cloud": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],
    "Terraform": ["terraform"],
    "Nginx": ["nginx"],
    "Linux": ["linux"],
    # Tools
    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "VS Code": ["vs code", "vscode", "visual studio code"],
    "Postman": ["postman"],
    "Jira": ["jira"],
    "Figma": ["figma"],
    # Testing
    "Pytest": ["pytest"],
    "JUnit": ["junit"],
    "Selenium": ["selenium"],
    # Mobile
    "Android": ["android"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],
    # Computer science basics
    "Data Structures and Algorithms": [
        "data structures and algorithms", "data structures & algorithms",
        "data structures", "dsa",
    ],
    "OOP": ["oop", "oops", "object oriented programming", "object-oriented programming"],
    "DBMS": ["dbms", "database management system", "database management systems"],
    "Operating Systems": ["operating system", "operating systems"],
    "Computer Networks": ["computer networks", "computer networking"],
    "System Design": ["system design"],
    "Design Patterns": ["design patterns"],
    "Microservices": ["microservices"],
    "Agile": ["agile", "scrum"],
}

# A word boundary that also works for names like C++, C#, Node.js, .NET
BEFORE = r"(?<![A-Za-z0-9+#.])"
AFTER = r"(?![A-Za-z0-9+#])"

PATTERNS = {}
for _skill, _aliases in SKILLS.items():
    _parts = [re.escape(a) for a in sorted(_aliases, key=len, reverse=True)]
    PATTERNS[_skill] = re.compile(BEFORE + "(?:" + "|".join(_parts) + ")" + AFTER, re.I)

# "C" and "R" are single letters, so we accept them only when they stand alone
# in a list, like "Languages: C, C++, Java", and only inside the Skills section.
SINGLE_LETTER = {
    "C": re.compile(r"(?:^|[,;/|:\u2022\u00b7])[ \t]*C[ \t]*(?=[,;/|\u2022\u00b7]|$)", re.M),
    "R": re.compile(r"(?:^|[,;/|:\u2022\u00b7])[ \t]*R[ \t]*(?=[,;/|\u2022\u00b7]|$)", re.M),
}


def find_skills(text):
    """Return skills found in the text, in order of first appearance."""
    hits = []
    for skill, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match:
            hits.append((match.start(), skill))
    return [skill for _, skill in sorted(hits)]


def extract_skills(skills_section_text, full_text):
    """Skills from the Skills section first (in the order written),
    then any other skills mentioned elsewhere, like in Projects."""
    found = []

    # C and R only count inside the Skills section
    for skill, pattern in SINGLE_LETTER.items():
        if pattern.search(skills_section_text or ""):
            found.append(skill)

    for text in (skills_section_text or "", full_text or ""):
        for skill in find_skills(text):
            if skill not in found:
                found.append(skill)
    return found
