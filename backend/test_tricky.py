from section_splitter import split_sections

tricky = """RIYA SHARMA
riya@example.com | +91 90000 00000
OBJECTIVE: Motivated CS student looking for a software role.
EDUCATION
B.Tech CSE, Sample Institute, 2022 - 2026
Courses: Data Structures, Operating Systems
Skills: Python, Java, SQL
Tools: Git, Docker
PROJECTS
Chat App
- Built with Flask
STRENGTHS
Teamwork, quick learner
Experience - 2 internships
Backend Intern, Sample Corp
DECLARATION
I hereby declare that the above is true.
"""

for name, content in split_sections(tricky).items():
    print("===", name.upper())
    print(content)
    print()