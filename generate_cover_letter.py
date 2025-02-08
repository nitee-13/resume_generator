import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_knowledge_base():
    """Load both the FAISS index and the original knowledge base content."""
    # Load FAISS index and metadata
    index = faiss.read_index(os.path.join("knowledge_base", "rag_faiss.index"))
    with open(os.path.join("knowledge_base", "rag_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Load original knowledge base content
    with open(os.path.join("knowledge_base", "knowledge_base.json"), "r") as f:
        knowledge_base = json.load(f)
    
    return index, metadata, knowledge_base

def get_content_from_source(source, knowledge_base):
    """Get the actual content from the knowledge base using the source path."""
    try:
        parts = source.split('.')
        current = knowledge_base
        
        for part in parts:
            if '[' in part:
                # Handle array indexing
                name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                if name:
                    current = current[name][idx]
                else:
                    current = current[idx]
            else:
                current = current[part]
        
        return current
    except Exception as e:
        print(f"Error accessing {source}: {str(e)}")
        return None

def get_embedding(text):
    """Get embedding for the job description."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def search_knowledge_base(index, metadata, knowledge_base, query_embedding, k=15):
    """Search the FAISS index for relevant information."""
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    # Get the relevant metadata and their text content
    relevant_info = []
    processed_sources = set()  # Track processed sources to avoid duplicates
    
    # First, explicitly search for projects that match the job requirements
    for project in knowledge_base['Projects']:
        project_text = (
            f"{project['Name']} {project['Description']} " +
            ' '.join(project['Tools']) + ' ' +
            ' '.join(project.get('Keywords', [])) + ' ' +
            ' '.join(project['Outcomes'])
        ).lower()
        
        # Define broader matching criteria based on job requirements
        relevant_terms = {
            # AI/ML terms
            'ai', 'ml', 'bert', 'nlp', 'tensorflow', 'pytorch', 
            'machine learning', 'deep learning', 'computer vision',
            'opencv', 'neural', 'sentiment', 'classification',
            'detection', 'analytics',
            
            # Data processing terms
            'real-time', 'pipeline', 'api', 'scalable', 'processing',
            
            # Cloud/Infrastructure terms
            'aws', 'cloud', 'docker', 'kubernetes', 'deployment',
            
            # Specific technologies mentioned in job
            'python', 'fastapi', 'redis', 'kafka', 'spark'
        }
        
        # Check if project matches any relevant terms
        if any(term in project_text for term in relevant_terms):
            info = {
                'text': f"Project: {project['Name']}\n" +
                       f"Description: {project['Description']}\n" +
                       f"Tools: {', '.join(project['Tools'])}\n" +
                       "Outcomes:\n" +
                       "\n".join(f"- {outcome}" for outcome in project['Outcomes']),
                'type': 'Project',
                'name': project['Name'],
                'tools': project['Tools']
            }
            relevant_info.append(info)
    
    # Then process other entries from FAISS search
    for idx in indices[0]:
        if idx < len(metadata):
            source = metadata[idx]['source']
            base_source = source.split('[')[0]
            
            if base_source in processed_sources:
                continue
                
            if 'WorkExperience' in source:
                # Get the full work experience entry
                work_idx = int(source.split('[')[1].split(']')[0])
                work_exp = knowledge_base['WorkExperience'][work_idx]
                info = {
                    'text': f"Role: {work_exp['Role']}\n" +
                           f"Company: {work_exp['Company']}\n" +
                           f"Dates: {work_exp['Dates']}\n" +
                           "Key Contributions:\n" +
                           "\n".join(f"- {contrib}" for contrib in work_exp['KeyContributions']) +
                           "\nTechnologies: " + ", ".join(work_exp['Technologies']),
                    'type': 'Work Experience',
                    'company': work_exp['Company'],
                    'role': work_exp['Role'],
                    'date': work_exp['Dates']
                }
                relevant_info.append(info)
                processed_sources.add(base_source)
                
            elif 'Skills' in source:
                # Get the full skills category
                category = source.split('.')[1].split('[')[0]
                skills = knowledge_base['Skills'].get(category, [])
                if skills:
                    info = {
                        'text': f"{category} Skills: {', '.join(skills)}",
                        'type': 'Skill'
                    }
                    relevant_info.append(info)
                    processed_sources.add(f"Skills.{category}")
    
    return relevant_info

def extract_technical_requirements(job_description):
    """Extract key technical requirements from the job description using GPT."""
    prompt = f"""
    Please analyze this job description and extract the key requirements.
    Respond ONLY with a JSON object containing these keys:
    {{
        "technical_skills": [list of required technical skills],
        "experience_years": "number or range",
        "education": "required education",
        "bonus_skills": [list of preferred/bonus skills],
        "key_technologies": [list of specific technologies mentioned]
    }}

    Job Description:
    {job_description}
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a technical recruiter expert at analyzing job descriptions. Extract and categorize requirements precisely. Respond only with the requested JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0  # Use 0 temperature for consistent output
    )
    
    # Parse the response text as JSON
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback in case of parsing error
        return {
            "technical_skills": [],
            "experience_years": "Not specified",
            "education": "Not specified",
            "bonus_skills": [],
            "key_technologies": []
        }

def generate_cover_letter(job_description, relevant_experience, requirements):
    """Generate a cover letter using OpenAI's GPT model with extracted requirements."""
    # Get name from knowledge base
    name = knowledge_base.get('PersonalInfo', {}).get('Name', 'Candidate')
    
    prompt = f"""
    Job Description:
    {job_description}

    Key Requirements:
    - Technical Skills Required: {', '.join(requirements['technical_skills'])}
    - Experience Required: {requirements['experience_years']}
    - Key Technologies: {', '.join(requirements['key_technologies'])}
    
    Relevant information about me:
    {relevant_experience}

    My name: {name}

    Please write a professional cover letter following these specific guidelines:

    1. Opening Paragraph:
    - Use my name ({name}) in the signature
    - Show genuine enthusiasm for the specific role and company
    - Mention how you learned about the position
    - Include a brief overview of your relevant technical background, specifically mentioning experience with {', '.join(requirements['technical_skills'][:3])}

    2. Body Paragraphs:
    - Address these key technical requirements: {', '.join(requirements['technical_skills'])}
    - Connect your experience specifically to these technologies: {', '.join(requirements['key_technologies'])}
    - Highlight relevant projects or achievements that demonstrate the required skills
    - Address both technical skills and soft skills mentioned in the job description
    - Use specific examples and metrics where possible
    - If applicable, mention any bonus skills you possess from this list: {', '.join(requirements['bonus_skills'])}

    3. Closing Paragraph:
    - Reiterate your interest and fit for the role
    - Include a call to action
    - Thank the reader for their time
    - End with a proper signature using my name ({name})

    Additional Requirements:
    - Keep the letter concise (around 400 words)
    - Use a professional yet engaging tone
    - Focus on the most relevant experiences that match the job requirements
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional cover letter writer specializing in technical positions. You excel at matching candidate experiences with job requirements and creating compelling narratives that demonstrate technical expertise and soft skills."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def generate_interest_statement(job_description, requirements, relevant_experience):
    """Generate a personalized statement explaining interest in the role."""
    
    # Identify role type based on job description and requirements
    job_desc_lower = job_description.lower()
    tech_stack = set(requirements['technical_skills'] + requirements['key_technologies'])
    
    # Define role type indicators with expanded software development keywords
    role_indicators = {
        'ai_ml': {
            'keywords': {'ai', 'ml', 'machine learning', 'deep learning', 'nlp', 'neural',
                        'tensorflow', 'pytorch', 'bert', 'transformer', 'computer vision',
                        'artificial intelligence', 'model training', 'model deployment'},
            'priority': ['Projects', 'Work Experience', 'Skills']
        },
        'data_engineering': {
            'keywords': {'data engineer', 'etl', 'pipeline', 'kafka', 'spark', 'data warehouse',
                        'airflow', 'data lake', 'streaming', 'data pipeline', 'data processing',
                        'data infrastructure', 'data modeling'},
            'priority': ['Work Experience', 'Projects', 'Skills']
        },
        'software_dev': {
            'keywords': {
                # General software development
                'software engineer', 'software developer', 'application developer',
                
                # Frontend specific
                'frontend', 'front-end', 'front end', 'react', 'vue', 'angular',
                'javascript', 'typescript', 'ui', 'ux', 'web developer',
                'responsive design', 'css', 'html', 'sass', 'less',
                
                # Backend specific
                'backend', 'back-end', 'back end', 'api', 'rest', 'graphql',
                'django', 'flask', 'fastapi', 'node', 'express', 'spring',
                'database', 'sql', 'nosql', 'mongodb',
                
                # Full stack
                'full stack', 'full-stack', 'fullstack', 'end-to-end',
                'web application', 'web development', 'microservices',
                'rest api', 'api development', 'mvc', 'orm'
            },
            'priority': ['Work Experience', 'Projects', 'Skills']
        },
        'cloud_devops': {
            'keywords': {'devops', 'aws', 'azure', 'kubernetes', 'docker', 'ci/cd',
                        'infrastructure', 'cloud', 'terraform', 'jenkins', 'gitlab',
                        'deployment', 'monitoring', 'sre', 'site reliability'},
            'priority': ['Work Experience', 'Skills', 'Projects']
        }
    }
    
    # Determine role type
    role_type = 'software_dev'  # default
    max_matches = 0
    
    for role, indicators in role_indicators.items():
        matches = len(indicators['keywords'] & (
            set(word.lower() for word in job_desc_lower.split()) |
            {tech.lower() for tech in tech_stack}
        ))
        if matches > max_matches:
            max_matches = matches
            role_type = role
    
    # Separate experiences by type
    experiences = {
        'Work Experience': [info for info in relevant_experience if info['type'] == 'Work Experience'],
        'Projects': [info for info in relevant_experience if info['type'] == 'Project'],
        'Skills': [info for info in relevant_experience if info['type'] == 'Skill']
    }
    
    # Score and sort experiences based on relevance to job requirements
    def score_relevance(item, tech_stack):
        text = item.get('text', '').lower()
        return sum(1 for tech in tech_stack if tech.lower() in text)
    
    for exp_type in experiences:
        if experiences[exp_type]:
            experiences[exp_type].sort(
                key=lambda x: score_relevance(x, tech_stack),
                reverse=True
            )
    
    # Format experiences according to role priority
    formatted_experience = ""
    for exp_type in role_indicators[role_type]['priority']:
        if experiences[exp_type]:
            if exp_type == 'Projects':
                formatted_experience += f"\nRelevant {exp_type}:\n" + "\n".join(
                    f"- {proj['name']}: {proj.get('text', '')}" 
                    for proj in experiences[exp_type][:3]  # Limit to top 3 most relevant
                )
            elif exp_type == 'Work Experience':
                formatted_experience += f"\n{exp_type}:\n" + "\n".join(
                    f"- {exp['role']} at {exp['company']}: {exp.get('text', '')}" 
                    for exp in experiences[exp_type]
                )
            else:  # Skills
                formatted_experience += f"\nRelevant {exp_type}:\n" + "\n".join(
                    f"- {skill['text']}" for skill in experiences[exp_type]
                )
    
    # Customize prompt based on role type
    role_specific_prompts = {
        'ai_ml': """
            1. Start with enthusiasm for AI/ML engineering and the company's AI initiatives
            2. Highlight relevant AI projects first, especially those with concrete metrics
            3. Connect AI/ML project experience with the role's specific needs
            4. Mention experience with ML frameworks and model deployment
        """,
        'software_dev': """
            1. Start with enthusiasm for software development and the company's technical challenges
            2. Highlight relevant work experience with similar tech stack
            3. Mention specific projects that demonstrate proficiency in required technologies
            4. Focus on coding practices, architecture, and scalability experience
        """,
        'data_engineering': """
            1. Start with enthusiasm for data engineering and the company's data challenges
            2. Highlight experience with data pipelines and ETL processes
            3. Mention specific projects involving data processing at scale
            4. Focus on database, big data, and data streaming experience
        """,
        'cloud_devops': """
            1. Start with enthusiasm for DevOps and cloud infrastructure
            2. Highlight experience with cloud platforms and containerization
            3. Mention specific projects involving CI/CD and infrastructure
            4. Focus on automation, scalability, and reliability achievements
        """
    }

    prompt = f"""
    Based on this job description and requirements:

    Job Description:
    {job_description}

    Key Requirements:
    - Technical Skills: {', '.join(requirements['technical_skills'])}
    - Technologies: {', '.join(requirements['key_technologies'])}
    
    My relevant experience and background:
    {formatted_experience}
    
    Generate a compelling and specific paragraph (150-200 words) explaining why I am interested in this role.
    The paragraph should:
    {role_specific_prompts.get(role_type, role_specific_prompts['software_dev'])}
    5. Show alignment with the company's technical goals
    6. Reference specific metrics or achievements
    7. End with enthusiasm for contributing to the team

    Keep the tone professional but enthusiastic. Focus on concrete technical experiences and specific achievements.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a technical professional who excels at articulating genuine interest in technical roles. Focus on concrete technical experiences while maintaining authenticity."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def get_multiline_input():
    """Get multi-line input from user until they type 'DONE' on a new line."""
    print("\nPlease paste the job description below.")
    print("After pasting, type 'DONE' on a new line and press Enter:")
    print("=" * 50)
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        lines.append(line)
    
    return "\n".join(lines)

def extract_relevant_info(knowledge_base, job_description, requirements):
    relevant_info = []
    
    # Convert job description and requirements to lowercase for better matching
    job_desc_lower = job_description.lower()
    req_lower = [req.lower() for req in requirements]
    
    # Extract keywords from job description
    keywords = set()
    # Add common technical keywords
    tech_keywords = ['python', 'java', 'javascript', 'ai', 'ml', 'machine learning', 
                    'deep learning', 'nlp', 'data science', 'cloud', 'aws', 
                    'frontend', 'backend', 'full stack', 'database']
    
    for keyword in tech_keywords:
        if keyword in job_desc_lower:
            keywords.add(keyword)
    
    # Add requirements as keywords
    keywords.update(req_lower)
    
    # Extract work experience
    if 'WorkExperience' in knowledge_base:
        for exp in knowledge_base['WorkExperience']:
            # Check if experience matches keywords
            exp_text = ' '.join(str(v) for v in exp.values()).lower()
            if any(keyword in exp_text for keyword in keywords):
                relevant_info.append({
                    'type': 'Work Experience',
                    'role': exp['Role'],
                    'company': exp['Company'],
                    'text': '\n'.join(exp['KeyContributions'])
                })
    
    # Extract relevant projects
    if 'Projects' in knowledge_base:
        for project in knowledge_base['Projects']:
            # Check if project matches keywords
            project_text = ' '.join(str(v) for v in project.values()).lower()
            if any(keyword in project_text for keyword in keywords):
                relevant_info.append({
                    'type': 'Project',
                    'Name': project['Name'],
                    'Description': project['Description'],
                    'Tools': project['Tools'],
                    'Outcomes': project['Outcomes']
                })
    
    # Extract relevant skills
    if 'Skills' in knowledge_base:
        for category, skills in knowledge_base['Skills'].items():
            if isinstance(skills, list):  # Check if skills is a list
                category_text = (category + ' ' + ' '.join(skills)).lower()
                if any(keyword in category_text for keyword in keywords):
                    relevant_info.append({
                        'type': 'Skill',
                        'category': category,
                        'skills': skills
                    })
    
    return relevant_info

# For debugging purposes, you might want to add this:
def debug_relevant_info(relevant_info):
    print("\nDebugging Relevant Information:")
    for info in relevant_info:
        print(f"\nType: {info.get('type', 'Unknown')}")
        
        # Handle different data structures safely using .get()
        if info.get('type') == 'Project':
            print(f"Name: {info.get('name', 'N/A')}")
            print(f"Description: {info.get('text', 'N/A')}")
            print(f"Tools: {', '.join(info.get('tools', []))}")
            print(f"Outcomes: {info.get('Outcomes', 'N/A')}")
        
        elif info.get('type') == 'Work Experience':
            print(f"Role: {info.get('role', 'N/A')}")
            print(f"Company: {info.get('company', 'N/A')}")
            print(f"Text: {info.get('text', 'N/A')}")
        
        elif info.get('type') == 'Skill':
            # Handle both possible skill structures
            if 'category' in info:
                print(f"Category: {info.get('category', 'N/A')}")
                print(f"Skills: {', '.join(info.get('skills', []))}")
            else:
                print(f"Text: {info.get('text', 'N/A')}")
        
        # Fallback for any other type or structure
        else:
            print("Content:")
            for key, value in info.items():
                if key != 'type':
                    print(f"{key}: {value}")

def main():
    # Load both FAISS index and knowledge base
    print("\nLoading knowledge base...")
    index, metadata, knowledge_base = load_knowledge_base()
    
    # Get job description from user
    job_description = get_multiline_input()
    
    if not job_description.strip():
        print("Error: No job description provided. Exiting...")
        return
    
    # Extract technical requirements
    print("\nAnalyzing job requirements...")
    requirements = extract_technical_requirements(job_description)
    
    # Print extracted requirements for verification
    print("\nExtracted Requirements:")
    print("Technical Skills:", ", ".join(requirements['technical_skills']))
    print("Experience:", requirements['experience_years'])
    print("Education:", requirements['education'])
    print("Key Technologies:", ", ".join(requirements['key_technologies']))
    print("Bonus Skills:", ", ".join(requirements['bonus_skills']))
    
    # Get embedding for job description
    job_embedding = get_embedding(job_description)
    
    # Search knowledge base for relevant information
    print("\nRetrieving relevant experiences...")
    relevant_info = search_knowledge_base(index, metadata, knowledge_base, job_embedding)
    
    # Add this line for debugging
    debug_relevant_info(relevant_info)
    
    # Print retrieved information in a more readable format
    print("\nRetrieved Information from Knowledge Base:")
    print("=" * 80)
    for idx, info in enumerate(relevant_info, 1):
        print(f"\nEntry {idx}:")
        if isinstance(info, dict):
            print("Content:")
            print(info.get('text', ''))  # Print the actual content
            print("\nMetadata:")
            for key, value in info.items():
                if key != 'text':  # Skip printing the text again
                    print(f"{key}: {value}")
        else:
            print(info)
        print("-" * 40)
    print("=" * 80)
    
    # Format relevant information focusing on the actual content
    formatted_experience = "\n".join([
        f"Experience at {info.get('company', 'N/A')} - {info.get('role', 'N/A')}:\n{info.get('text', '')}"
        for info in relevant_info
        if isinstance(info, dict)
    ])
    
    # Generate interest statement
    print("\nGenerating interest statement...")
    interest_statement = generate_interest_statement(job_description, requirements, relevant_info)
    print("\nInterest Statement:")
    print("=" * 80)
    print(interest_statement)
    print("=" * 80)
    
    # Ask if user wants to proceed with cover letter
    proceed = input("\nWould you like to generate a full cover letter? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    # Generate cover letter with requirements
    print("\nGenerating cover letter...")
    cover_letter = generate_cover_letter(job_description, formatted_experience, requirements)
    
    # Print the result
    print("\nGenerated Cover Letter:")
    print("=" * 80)
    print(cover_letter)
    print("\n" + "=" * 80)
    
    # Add option to save both interest statement and cover letter
    save = input("\nWould you like to save these documents? (y/n): ")
    if save.lower() == 'y':
        company_name = input("Enter company name for the file names: ")
        base_name = company_name.lower().replace(' ', '_')
        
        # Save interest statement
        with open(f"interest_statement_{base_name}.txt", 'w') as f:
            f.write(interest_statement)
        
        # Save cover letter
        with open(f"cover_letter_{base_name}.txt", 'w') as f:
            f.write(cover_letter)
            
        print(f"\nFiles saved as:")
        print(f"- interest_statement_{base_name}.txt")
        print(f"- cover_letter_{base_name}.txt")

if __name__ == "__main__":
    main() 