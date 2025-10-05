from flask import Flask, render_template, request, jsonify
import pandas as pd
import time
import re
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

app = Flask(__name__)

# Load and preprocess dataset better
try:
    df = pd.read_csv("data/Intern_QA_Dataset.csv")
    print(f"Dataset loaded: {len(df)} rows")
except:
    # Create comprehensive internship knowledge base
    df = pd.DataFrame({
        'Question': [
            'How to find internship opportunities?',
            'What should I include in my internship portfolio?',
            'How to prepare for technical interviews?',
            'What are common internship interview questions?',
            'How to write a good internship resume?',
            'How to negotiate an internship offer?',
            'What skills are important for internships?',
            'How to write a cover letter for internship?'
        ],
        'Answer': [
            'Use LinkedIn, company career pages, university portals, and networking events. Target specific companies and roles that match your skills.',
            'Include projects, GitHub links, technical skills, certifications, academic achievements, and any relevant work experience or volunteer work.',
            'Practice coding problems on LeetCode, review CS fundamentals, prepare for behavioral questions, and research the company thoroughly.',
            'Tell me about yourself, why this company, technical questions related to your field, situational questions, and your strengths/weaknesses.',
            'Focus on relevant projects, technical skills, education, any work experience. Use action verbs and quantify achievements where possible.',
            'Research market rates, highlight your value, be professional, consider other benefits besides salary like learning opportunities or flexibility.',
            'Technical skills specific to your field, communication, problem-solving, teamwork, adaptability, and willingness to learn new technologies.',
            'Customize for each application, highlight relevant skills, show enthusiasm for the company, and explain how you can contribute to their team.'
        ]
    })

# Enhanced knowledge base for internships
INTERNSHIP_KNOWLEDGE_BASE = {
    "finding_internships": """
    Strategies to find internship opportunities:
    1. LinkedIn - Follow companies, use job search, connect with recruiters
    2. Company Career Pages - Direct applications on official websites
    3. University Career Centers - Campus recruitment and job portals
    4. Networking Events - Career fairs, tech meetups, hackathons
    5. Referrals - Connect with alumni and professionals in your network
    6. Online Platforms - Indeed, Glassdoor, AngelList, Internshala
    7. Cold Emailing - Reach out to hiring managers directly
    """,
    
    "portfolio_creation": """
    Essential elements for internship portfolio:
    • Projects with GitHub links and live demos
    • Technical skills and proficiency levels
    • Certifications and online course completions
    • Academic projects and research work
    • Resume and contact information
    • Blog or technical writing samples
    • Recommendations or testimonials if available
    """,
    
    "interview_preparation": """
    Technical internship interview preparation:
    • Practice coding problems (arrays, strings, linked lists, trees)
    • Review data structures and algorithms
    • Study system design basics
    • Prepare for behavioral questions (STAR method)
    • Research the company and its products
    • Prepare questions to ask the interviewer
    • Mock interviews with peers or mentors
    """,
    
    "common_questions": """
    Common internship interview questions:
    Technical:
    - Explain [technology] you used in projects
    - Solve this coding problem
    - How would you optimize this code?
    
    Behavioral:
    - Tell me about yourself
    - Why do you want to work here?
    - Describe a challenging project
    - How do you handle conflicts?
    - Where do you see yourself in 5 years?
    """,
    
    "resume_tips": """
    Internship resume best practices:
    • One-page limit, clean formatting
    • Relevant projects with technologies used
    • Technical skills section (programming languages, tools, frameworks)
    • Education with GPA (if good)
    • Work experience (even if unrelated, highlight transferable skills)
    • Extracurricular activities and leadership roles
    • Certifications and online courses
    """,
    
    "negotiation_strategies": """
    Internship offer negotiation tips:
    1. Research typical stipends for similar roles
    2. Consider the entire package (learning, mentorship, future opportunities)
    3. Be professional and appreciative in communication
    4. Highlight your unique value and skills
    5. Consider asking for specific learning opportunities
    6. Get multiple offers for leverage
    7. Know your minimum acceptable offer
    """
}

# Initialize sentence transformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for better matching
knowledge_embeddings = {}
for category, content in INTERNSHIP_KNOWLEDGE_BASE.items():
    knowledge_embeddings[category] = model.encode(content, convert_to_tensor=True)

class IntelligentCareerAdvisor:
    def __init__(self):
        self.conversation_history = []
        
    def find_best_match(self, question):
        question_embedding = model.encode(question, convert_to_tensor=True)
        best_match = None
        best_score = 0
        
        for category, embedding in knowledge_embeddings.items():
            similarity = util.pytorch_cos_sim(question_embedding, embedding).item()
            if similarity > best_score:
                best_score = similarity
                best_match = category
        
        return best_match, best_score
    
    def generate_intelligent_response(self, question):
        # Clean and analyze question
        question_lower = question.lower().strip()
        
        # Enhanced intent detection
        intents = self.detect_comprehensive_intent(question_lower)
        best_category, confidence = self.find_best_match(question_lower)
        
        # Generate response based on intent and category
        response = self.construct_response(question_lower, intents, best_category, confidence)
        
        # Store conversation
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': time.time(),
            'confidence': confidence
        })
        
        return response
    
    def detect_comprehensive_intent(self, question):
        question_lower = question.lower()
        
        intent_keywords = {
            'finding_internships': ['find', 'search', 'opportunities', 'where to apply', 'get internship', 'looking for'],
            'portfolio': ['portfolio', 'projects', 'github', 'showcase', 'demonstrate skills'],
            'interview': ['interview', 'technical interview', 'prepare', 'questions', 'hr round'],
            'resume': ['resume', 'cv', 'curriculum vitae', 'application', 'apply'],
            'skills': ['skills', 'learn', 'technical skills', 'soft skills', 'required skills'],
            'negotiation': ['negotiate', 'offer', 'salary', 'stipend', 'compensation', 'accept offer'],
            'cover_letter': ['cover letter', 'application letter', 'motivation letter'],
            'networking': ['network', 'connect', 'linkedin', 'referral', 'contact']
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return detected_intents if detected_intents else ['general_advice']
    
    def construct_response(self, question, intents, category, confidence):
        # Base response from knowledge base
        if category and confidence > 0.3:
            base_response = INTERNSHIP_KNOWLEDGE_BASE.get(category, "")
        else:
            base_response = self.get_general_advice(question)
        
        # Add confidence indicator
        if confidence > 0.7:
            confidence_phrase = "🎯 I'm confident about this advice:"
        elif confidence > 0.4:
            confidence_phrase = "💡 Based on my career guidance expertise:"
        else:
            confidence_phrase = "🤔 Here's my suggestion:"
        
        # Add personalized follow-up
        follow_up = self.get_follow_up_question(intents, category)
        
        formatted_response = f"""
{confidence_phrase}

{base_response}

{follow_up}

*Confidence level: {confidence:.1%}*
        """.strip()
        
        return formatted_response
    
    def get_general_advice(self, question):
        general_responses = {
            'technical': """
            For technical roles, focus on:
            • Building practical projects to showcase your skills
            • Mastering core programming concepts
            • Contributing to open-source projects
            • Creating a strong GitHub portfolio
            • Practicing coding interview questions regularly
            """,
            'preparation': """
            General internship preparation:
            • Research companies thoroughly before applying
            • Customize your application for each position
            • Develop both technical and soft skills
            • Network with professionals in your field
            • Prepare for different types of interviews
            """,
            'default': """
            Successful internship hunting requires:
            • Starting your search early (3-6 months in advance)
            • Applying to multiple companies (20-30 applications)
            • Following up on applications
            • Preparing a strong online presence
            • Being persistent and learning from rejections
            """
        }
        
        if any(word in question for word in ['technical', 'coding', 'programming', 'developer', 'engineer']):
            return general_responses['technical']
        elif any(word in question for word in ['prepare', 'ready', 'get ready']):
            return general_responses['preparation']
        else:
            return general_responses['default']
    
    def get_follow_up_question(self, intents, category):
        follow_ups = {
            'finding_internships': "Would you like specific strategies for your target companies or locations?",
            'portfolio': "Should I help you prioritize which projects to include in your portfolio?",
            'interview': "Do you need help with technical, behavioral, or both types of interview preparation?",
            'resume': "Would you like me to review specific sections of your resume?",
            'skills': "Are you looking for technical skills, soft skills, or industry-specific skills?",
            'negotiation': "Do you need help with salary research or communication strategies?",
            'general_advice': "What specific aspect of internship hunting would you like to explore further?"
        }
        
        for intent in intents:
            if intent in follow_ups:
                return follow_ups[intent]
        
        return "What other internship-related questions can I help you with today?"

# Initialize the advisor
career_advisor = IntelligentCareerAdvisor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'})
        
        # Simulate thinking time
        time.sleep(1.5)
        
        # Generate intelligent response
        response = career_advisor.generate_intelligent_response(question)
        confidence = career_advisor.conversation_history[-1]['confidence'] if career_advisor.conversation_history else 0.5
        
        return jsonify({
            'answer': response,
            'confidence': round(confidence, 3),
            'conversation_count': len(career_advisor.conversation_history)
        })
        
    except Exception as e:
        error_response = """
        I apologize, but I'm having trouble processing your question right now. 

        Here are some topics I can definitely help with:
        • Finding internship opportunities
        • Resume and portfolio building
        • Interview preparation
        • Skill development
        • Offer negotiation

        Please try rephrasing your question about internships and career growth!
        """
        
        return jsonify({
            'answer': error_response,
            'confidence': 0.1,
            'error': str(e)
        })

@app.route("/conversation/summary")
def get_conversation_summary():
    if career_advisor.conversation_history:
        topics = [advice.detect_comprehensive_intent(entry['question'])[0] 
                 for entry in career_advisor.conversation_history[-5:] 
                 if career_advisor.conversation_history]
        common_topic = max(set(topics), key=topics.count) if topics else "career guidance"
        return jsonify({'summary': f"We've been discussing {common_topic}. I've provided {len(career_advisor.conversation_history)} pieces of career advice."})
    return jsonify({'summary': "No conversation yet. Ask me about internships!"})

@app.route("/suggestions")
def get_suggestions():
    suggestions = [
        "How can I find internship opportunities in tech companies?",
        "What should I include in my internship portfolio?",
        "How to prepare for technical internship interviews?",
        "What are the most common internship interview questions?",
        "How to write an impressive internship resume?",
        "How to negotiate an internship offer effectively?",
        "What skills are most valuable for tech internships?",
        "How to write a compelling cover letter for internships?"
    ]
    return jsonify({'suggestions': suggestions})

if __name__ == "__main__":
    print("🚀 CareerGPT AI Assistant Started!")
    print("💼 Specialized in Internship Guidance")
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True)