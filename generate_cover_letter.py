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

def search_knowledge_base(index, metadata, knowledge_base, query_embedding, requirements, k=15):
    """Enhanced semantic search using FAISS index with requirements context."""
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    relevant_info = []
    processed_sources = set()
    
    # Create a comprehensive set of search terms from requirements
    search_terms = set()
    for skill in requirements['technical_skills']:
        search_terms.add(skill['skill'].lower())
        search_terms.update(term.lower() for term in skill['related_terms'])
    for tech in requirements['key_technologies']:
        search_terms.add(tech['technology'].lower())
        search_terms.update(term.lower() for term in tech['related_terms'])
    
    # Process FAISS search results
    for idx in indices[0]:
        if idx >= len(metadata):
            continue
            
        source = metadata[idx]['source']
        base_source = source.split('.')[0]  # Get the root source (e.g., "Projects[0]")
        
        if base_source in processed_sources:
            continue
        
        content = get_content_from_source(source, knowledge_base)
        if not content:
            continue
        
        # Convert content to searchable text
        if isinstance(content, dict):
            searchable_text = json.dumps(content).lower()
        else:
            searchable_text = str(content).lower()
        
        # Check if content is relevant based on search terms
        if any(term in searchable_text for term in search_terms):
            if 'Projects[' in source:
                # Get the full project data
                project_idx = int(source.split('[')[1].split(']')[0])
                project = knowledge_base['Projects'][project_idx]
                
                info = {
                    'type': 'Project',
                    'name': project.get('Name', 'Unnamed Project'),
                    'text': (
                        f"Project: {project.get('Name', 'Unnamed Project')}\n"
                        f"Overview: {project.get('Overview', '')}\n"
                        "Technical Details:\n"
                    )
                }
                
                # Add technical architecture details if available
                if 'TechnicalArchitecture' in project:
                    tech_arch = project['TechnicalArchitecture']
                    if isinstance(tech_arch, dict):
                        for key, value in tech_arch.items():
                            info['text'] += f"- {key}: {json.dumps(value, indent=2)}\n"
                
                # Add implementation details if available
                if 'Implementation' in project:
                    impl = project['Implementation']
                    if isinstance(impl, dict):
                        info['text'] += "\nImplementation:\n"
                        for key, value in impl.items():
                            info['text'] += f"- {key}: {json.dumps(value, indent=2)}\n"
                
                relevant_info.append(info)
                processed_sources.add(base_source)
                
            elif 'Education[' in source:
                education = content
                info = {
                    'type': 'Education',
                    'text': f"Education: {json.dumps(education, indent=2)}",
                    'degree': education.get('Degree', 'Not specified'),
                    'institution': education.get('Institution', 'Not specified')
                }
                relevant_info.append(info)
                processed_sources.add(base_source)
            
            elif 'WorkExperience[' in source:
                work_exp = content
                info = {
                    'type': 'Work Experience',
                    'role': work_exp.get('Role', 'Not specified'),
                    'company': work_exp.get('Company', 'Not specified'),
                    'text': (
                        f"Role: {work_exp.get('Role', 'Not specified')}\n"
                        f"Company: {work_exp.get('Company', 'Not specified')}\n"
                        f"Duration: {work_exp.get('Duration', 'Not specified')}\n"
                        "Responsibilities:\n" +
                        "\n".join(f"- {resp}" for resp in work_exp.get('Responsibilities', []))
                    )
                }
                relevant_info.append(info)
                processed_sources.add(base_source)
    
    return relevant_info

def extract_technical_requirements(job_description):
    """Extract key technical requirements from the job description using GPT."""
    prompt = f"""
    Please analyze this job description and extract the key requirements.
    Also include common synonyms and related terms for each technical skill to enable better semantic matching.
    Respond ONLY with a JSON object containing these keys:
    {{
        "technical_skills": [
            {{"skill": "main_skill", "related_terms": ["synonym1", "synonym2", "related_tech1"]}},
        ],
        "experience_years": "number or range",
        "education": "required education",
        "key_technologies": [
            {{"technology": "main_tech", "related_terms": ["synonym1", "synonym2", "related_tech1"]}}
        ]
    }}

    Job Description:
    {job_description}
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a technical recruiter expert at analyzing job descriptions and understanding technical terminology, including synonyms and related technologies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "technical_skills": [],
            "experience_years": "Not specified",
            "education": "Not specified",
            "key_technologies": []
        }

def generate_cover_letter(job_description, relevant_experience, requirements):
    """Generate a cover letter using OpenAI's GPT model with extracted requirements."""
    name = knowledge_base.get('PersonalInfo', {}).get('Name', 'Candidate')
    
    prompt = f"""
    Job Description:
    {job_description}

    Key Requirements:
    - Technical Skills Required: {', '.join(skill['skill'] for skill in requirements['technical_skills'])}
    - Experience Required: {requirements['experience_years']}
    - Key Technologies: {', '.join(tech['technology'] for tech in requirements['key_technologies'])}
    
    Relevant information about me:
    {relevant_experience}

    My name: {name}

    Please write a professional cover letter following these specific guidelines:

    1. Opening Paragraph:
    - Use my name ({name}) in the signature
    - Show genuine enthusiasm for the specific role and company
    - Mention how you learned about the position
    - Include a brief overview of your relevant technical background

    2. Body Paragraphs:
    - Address the key technical requirements
    - Connect your experience specifically to the required technologies
    - Highlight relevant projects or achievements that demonstrate the required skills
    - Address both technical skills and soft skills mentioned in the job description
    - Use specific examples and metrics where possible

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
    """Generate a personalized statement explaining interest in the role using semantic matching."""
    
    # Format experiences based on semantic relevance (already sorted by FAISS search)
    experiences = {
        'Work Experience': [info for info in relevant_experience if info['type'] == 'Work Experience'],
        'Projects': [info for info in relevant_experience if info['type'] == 'Project'],
        'Skills': [info for info in relevant_experience if info['type'] == 'Skill']
    }
    
    # Format experiences with most relevant first (using order from FAISS search)
    formatted_experience = ""
    
    # Work Experience
    if experiences['Work Experience']:
        formatted_experience += "\nRelevant Work Experience:\n" + "\n".join(
            f"- {exp['role']} at {exp['company']}: {exp.get('text', '')}" 
            for exp in experiences['Work Experience'][:2]  # Top 2 most relevant experiences
        )
    
    # Projects
    if experiences['Projects']:
        formatted_experience += "\nRelevant Projects:\n" + "\n".join(
            f"- {proj['name']}: {proj.get('text', '')}" 
            for proj in experiences['Projects'][:2]  # Top 2 most relevant projects
        )
    
    # Skills
    if experiences['Skills']:
        formatted_experience += "\nRelevant Skills:\n" + "\n".join(
            f"- {skill['text']}" for skill in experiences['Skills']
        )

    prompt = f"""
    Based on this job description and requirements:

    Job Description:
    {job_description}

    Key Requirements:
    Technical Skills: {', '.join(skill['skill'] for skill in requirements['technical_skills'])}
    Technologies: {', '.join(tech['technology'] for tech in requirements['key_technologies'])}
    
    My relevant experience and background:
    {formatted_experience}
    
    Generate a compelling and specific paragraph (150-200 words) explaining why I am interested in this role.
    The paragraph should:
    1. Show genuine enthusiasm for the specific technical challenges and technologies mentioned
    2. Connect my most relevant experiences to the role's requirements
    3. Highlight specific achievements and metrics that demonstrate expertise
    4. Show alignment with the company's technical goals
    5. End with enthusiasm for contributing to the team

    Focus on the most relevant technical experiences and specific achievements that match the job requirements.
    Keep the tone professional but enthusiastic.
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
    print("Technical Skills:", ", ".join(skill['skill'] for skill in requirements['technical_skills']))
    print("Experience:", requirements['experience_years'])
    print("Education:", requirements['education'])
    print("Key Technologies:", ", ".join(tech['technology'] for tech in requirements['key_technologies']))
    
    # Get embedding for job description
    job_embedding = get_embedding(job_description)
    
    # Search knowledge base for relevant information using FAISS
    print("\nRetrieving relevant experiences...")
    relevant_info = search_knowledge_base(index, metadata, knowledge_base, job_embedding, requirements)
    
    # Debug retrieved information
    debug_relevant_info(relevant_info)
    
    # Generate interest statement using semantic search results
    print("\nGenerating interest statement...")
    interest_statement = generate_interest_statement(job_description, requirements, relevant_info)
    
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
    # Print the interest statement
    print("\nGenerated Interest Statement:")
    print("=" * 80)
    print(interest_statement)
    print("=" * 80)
    # Format relevant information focusing on the actual content
    # Format relevant information for cover letter
    formatted_experience = "\n".join([
        f"Experience/Project: {info.get('name', info.get('role', 'Unknown'))}\n{info.get('text', '')}\n"
        for info in relevant_info
    ])
    
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