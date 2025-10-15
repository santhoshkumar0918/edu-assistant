#!/usr/bin/env python3
"""
TNPSC Exam Assistant - Core Module
AI-Powered Personalized Exam Preparation Assistant for Competitive Exams
"""

import os
import re
import json
import random
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
DATA_DIR = "data/TNPSC"
SYLLABUS_DIR = os.path.join(DATA_DIR, "syllabus")
QUESTIONS_DIR = os.path.join(DATA_DIR, "questions")
PDF_DIR = os.path.join(DATA_DIR, "pdf_papers")

for directory in [DATA_DIR, SYLLABUS_DIR, QUESTIONS_DIR, PDF_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Dictionary mapping unit numbers to unit names
UNITS = {
    1: "General Science",
    2: "Current Events",
    3: "Geography",
    4: "History and Culture of India",
    5: "Indian Polity",
    6: "Indian Economy",
    7: "Indian National Movement",
    8: "Mental Ability & Aptitude"
}

def create_sample_syllabus():
    """Create sample syllabus data for demonstration"""
    sample_syllabus = {
        1: """
        GENERAL SCIENCE: Physics - Universe - General Scientific laws - Scientific instruments - Inventions and discoveries - National scientific laboratories - Science glossary - Mechanics and properties of matter - Physical quantities, standards and units - Force, motion and energy - Electricity and Magnetism - Electronics & Communications - Heat, light and sound - Atomic and nuclear physics - Solid State Physics - Spectroscopy - Geophysics - Astronomy and space science.
        
        Chemistry - Elements and Compounds - Acids, bases and salts - Oxidation and reduction - Chemistry of ores and metals - Carbon, nitrogen and their compounds - Fertilizers, pesticides, insecticides - Biochemistry and biotechnology - Electrochemistry - Polymers and plastics.
        
        Botany - Main Concepts of life science - The cell - basic unit of life - Classification of living organisms - Nutrition and dietetics - Respiration - Excretion of metabolic waste - Bio-communication.
        
        Zoology - Blood and blood circulation - Endocrine system - Reproductive system - Genetics the science of heredity - Environment, ecology, health and hygiene - Human diseases - Communicable diseases and non-communicable diseases - prevention and remedies - Alcoholism and drug abuse - Animals, plants and human life.
        """,
        2: """
        CURRENT EVENTS: History - Latest diary of events - National - National symbols - Profile of States - Defence, national security and terrorism - World organizations - pacts and summits - Eminent persons & places in news - Sports & games - Books & authors - Awards & honours - Cultural panorama - Latest historical events - India and its neighbours - Latest terminology - Appointments - who is who?
        
        Political Science - India's foreign policy - Latest court verdicts - public opinion - Problems in conduct of public elections - Political parties and political system in India - Public awareness & General administration - Role of Voluntary organizations & Govt. - Welfare oriented govt. schemes, their utility.
        
        Geography - Geographical landmarks - Policy on environment and ecology.
        
        Economics - Current socio-economic problems - New economic policy & govt. sector. Science & Technology - Latest inventions on science & technology - Latest discoveries in Health Science - Mass media & communication.
        """,
        3: """
        GEOGRAPHY: Earth and Universe - Solar system - Atmosphere, hydrosphere, lithosphere - Monsoon, rainfall, weather and climate - Water resources - rivers in India - Soil, minerals & natural resources - Natural vegetation - Forest & wildlife - Agricultural pattern, livestock & fisheries - Transport including Surface transport & communication - Social geography - population-density and distribution - Natural calamities - disaster management - Bottom topography of Indian ocean, Arabian Sea and Bay of Bengal - Climate change - impact and consequences - mitigation measures - Pollution Control.
        """,
        4: """
        HISTORY AND CULTURE: Ancient India - Indus Valley Civilization - Vedic period - Mauryan Empire - Gupta period - Medieval India - Delhi Sultanate - Mughal Empire - South Indian kingdoms - Chola, Chera, Pandya dynasties - Vijayanagara Empire - Modern India - British rule - Freedom struggle - Post-independence India - Art and culture - Literature - Music and dance - Festivals - Monuments - UNESCO World Heritage sites.
        """,
        5: """
        INDIAN POLITY: Constitution of India - Fundamental Rights and Duties - Directive Principles - Union Government - Parliament - President - Prime Minister - Council of Ministers - Supreme Court - State Government - Governor - Chief Minister - State Legislature - High Courts - Local Government - Panchayati Raj - Municipal Corporations - Election Commission - Other Constitutional Bodies.
        """,
        6: """
        INDIAN ECONOMY: Economic planning - Five Year Plans - NITI Aayog - Banking system - RBI - Monetary policy - Fiscal policy - Budget - Taxation - Public finance - Agriculture - Industry - Services sector - Foreign trade - Balance of payments - Economic reforms - Liberalization - Privatization - Globalization - Economic indicators - GDP, GNP, inflation, unemployment.
        """,
        7: """
        INDIAN NATIONAL MOVEMENT: Early nationalist movement - Indian National Congress - Muslim League - Partition of Bengal - Swadeshi movement - Revolutionary activities - Mahatma Gandhi - Non-cooperation movement - Civil Disobedience - Quit India movement - Role of other leaders - Subhas Chandra Bose - Bhagat Singh - Chandrashekhar Azad - Women in freedom struggle - Partition and independence.
        """,
        8: """
        MENTAL ABILITY & APTITUDE: Logical reasoning - Analytical reasoning - Number series - Letter series - Coding and decoding - Blood relations - Direction sense - Ranking and arrangement - Puzzles - Data interpretation - Mathematical operations - Analogies - Classification - Pattern recognition - Problem solving - Decision making.
        """
    }
    
    # Save sample syllabus to files
    for unit_num, content in sample_syllabus.items():
        filepath = os.path.join(SYLLABUS_DIR, f"unit_{unit_num}_syllabus.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Sample syllabus created for {len(sample_syllabus)} units")
    return sample_syllabus

def create_sample_questions():
    """Create sample question bank for demonstration"""
    questions = []
    
    # Unit 1: General Science
    questions.extend([
        {
            "unit": 1,
            "question": "Which of the following is NOT a noble gas?",
            "option_a": "Neon",
            "option_b": "Argon", 
            "option_c": "Oxygen",
            "option_d": "Helium",
            "correct_option": "c",
            "explanation": "Oxygen is not a noble gas. The noble gases are helium, neon, argon, krypton, xenon, and radon.",
            "difficulty": "easy",
            "year": 2022
        },
        {
            "unit": 1,
            "question": "The concept of 'survival of the fittest' is associated with which scientist?",
            "option_a": "Charles Darwin",
            "option_b": "Gregor Mendel",
            "option_c": "Louis Pasteur",
            "option_d": "Alexander Fleming",
            "correct_option": "a",
            "explanation": "The concept of 'survival of the fittest' is associated with Charles Darwin's theory of evolution by natural selection.",
            "difficulty": "medium",
            "year": 2021
        },
        {
            "unit": 1,
            "question": "What is the chemical formula for water?",
            "option_a": "H2O2",
            "option_b": "H2O",
            "option_c": "HO2",
            "option_d": "H3O",
            "correct_option": "b",
            "explanation": "Water has the chemical formula H2O, consisting of two hydrogen atoms and one oxygen atom.",
            "difficulty": "easy",
            "year": 2023
        }
    ])
    
    # Unit 2: Current Events
    questions.extend([
        {
            "unit": 2,
            "question": "Who is the current Chief Minister of Tamil Nadu?",
            "option_a": "Edappadi K. Palaniswami",
            "option_b": "O. Panneerselvam",
            "option_c": "M.K. Stalin",
            "option_d": "Jayalalithaa",
            "correct_option": "c",
            "explanation": "M.K. Stalin is the current Chief Minister of Tamil Nadu since May 2021.",
            "difficulty": "easy",
            "year": 2024
        },
        {
            "unit": 2,
            "question": "Which organization was awarded the Nobel Peace Prize in 2023?",
            "option_a": "WHO",
            "option_b": "UNICEF",
            "option_c": "Narges Mohammadi",
            "option_d": "Red Cross",
            "correct_option": "c",
            "explanation": "Narges Mohammadi was awarded the Nobel Peace Prize in 2023 for her fight against the oppression of women in Iran.",
            "difficulty": "medium",
            "year": 2023
        }
    ])
    
    # Add more questions for other units
    for unit in range(3, 9):
        questions.extend([
            {
                "unit": unit,
                "question": f"Sample question for Unit {unit} - {UNITS[unit]}",
                "option_a": "Option A",
                "option_b": "Option B", 
                "option_c": "Option C",
                "option_d": "Option D",
                "correct_option": random.choice(["a", "b", "c", "d"]),
                "explanation": f"This is a sample explanation for Unit {unit}.",
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "year": random.choice([2021, 2022, 2023, 2024])
            }
        ])
    
    # Save questions to CSV
    df = pd.DataFrame(questions)
    filepath = os.path.join(QUESTIONS_DIR, "sample_questions.csv")
    df.to_csv(filepath, index=False)
    
    print(f"Sample questions created: {len(questions)} questions")
    return df

def load_syllabus():
    """Load syllabus data from files"""
    syllabus = {}
    
    # Check if syllabus files exist, if not create sample data
    if not any(os.path.exists(os.path.join(SYLLABUS_DIR, f"unit_{i}_syllabus.txt")) for i in range(1, 9)):
        print("Syllabus files not found. Creating sample syllabus...")
        return create_sample_syllabus()
    
    # Load existing syllabus files
    for unit_num in UNITS.keys():
        filepath = os.path.join(SYLLABUS_DIR, f"unit_{unit_num}_syllabus.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                syllabus[unit_num] = f.read()
        else:
            syllabus[unit_num] = f"Syllabus for Unit {unit_num} - {UNITS[unit_num]} not available."
    
    return syllabus

def load_questions():
    """Load questions from CSV files"""
    # Check if questions file exists, if not create sample data
    filepath = os.path.join(QUESTIONS_DIR, "sample_questions.csv")
    if not os.path.exists(filepath):
        print("Questions file not found. Creating sample questions...")
        return create_sample_questions()
    
    # Load existing questions
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} questions from database")
        return df
    except Exception as e:
        print(f"Error loading questions: {e}")
        print("Creating sample questions...")
        return create_sample_questions()

def extract_key_topics(text):
    """Extract key topics from syllabus text"""
    try:
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        
        # Filter out common words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'these', 'many', 'then', 'them', 'well', 'were'}
        
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_words)
        
        # Return top words
        return [word for word, _ in word_freq.most_common(15)]
    except Exception as e:
        print(f"Error extracting key topics: {e}")
        return ["topic1", "topic2", "topic3"]  # Fallback to prevent errors

class TNPSCExamAssistant:
    """Assistant class for TNPSC exam preparation"""
    
    def __init__(self):
        self.questions_df = load_questions()
        self.syllabus = load_syllabus()
        print("TNPSC Exam Assistant initialized")
    
    def get_unit_info(self, unit_num):
        """Get information about a specific unit"""
        if unit_num not in UNITS:
            return {"error": f"Unit {unit_num} not found"}
        
        # Get unit syllabus
        syllabus_text = self.syllabus[unit_num]
        
        # Get key topics
        key_topics = extract_key_topics(syllabus_text)
        
        # Get question count
        unit_questions = self.questions_df[self.questions_df['unit'] == unit_num]
        
        return {
            "unit_number": unit_num,
            "unit_name": UNITS[unit_num],
            "questions_count": len(unit_questions),
            "key_topics": key_topics,
            "syllabus_summary": syllabus_text[:200] + "..." if len(syllabus_text) > 200 else syllabus_text
        }
    
    def generate_quiz(self, unit_num=None, num_questions=10):
        """Generate a quiz with questions from specified unit or all units"""
        # Filter questions
        if unit_num is not None:
            if unit_num not in UNITS:
                return {"error": f"Unit {unit_num} not found"}
            filtered_questions = self.questions_df[self.questions_df['unit'] == unit_num]
        else:
            filtered_questions = self.questions_df
        
        # Sample questions
        if len(filtered_questions) >= num_questions:
            quiz_questions = filtered_questions.sample(num_questions).to_dict('records')
        else:
            quiz_questions = filtered_questions.to_dict('records')
            
        # Format the quiz
        for q in quiz_questions:
            q['options'] = {
                'A': q.pop('option_a'),
                'B': q.pop('option_b'),
                'C': q.pop('option_c'),
                'D': q.pop('option_d')
            }
        
        return {
            "title": f"TNPSC Practice Quiz - {UNITS[unit_num] if unit_num else 'All Units'}",
            "num_questions": len(quiz_questions),
            "questions": quiz_questions
        }
    
    def generate_study_plan(self, unit_num=None, days_remaining=30):
        """Generate a study plan based on days remaining"""
        if unit_num is not None and unit_num not in UNITS:
            return {"error": f"Unit {unit_num} not found"}
        
        # Determine units to study
        if unit_num is not None:
            units_to_study = [unit_num]
        else:
            units_to_study = list(UNITS.keys())
        
        # Create study plan
        study_plan = []
        days_per_unit = max(1, days_remaining // len(units_to_study))
        
        day_counter = 1
        for unit in units_to_study:
            unit_name = UNITS[unit]
            key_topics = extract_key_topics(self.syllabus[unit])
            
            for day in range(days_per_unit):
                # Calculate topics per day
                topics_per_day = max(1, len(key_topics) // days_per_unit)
                day_topics = key_topics[day * topics_per_day:(day + 1) * topics_per_day]
                
                study_plan.append({
                    "day": day_counter,
                    "unit": unit,
                    "unit_name": unit_name,
                    "topics": day_topics,
                    "activities": [
                        "Read syllabus and understand concepts",
                        "Practice sample questions",
                        "Review and revise"
                    ]
                })
                day_counter += 1
        
        return {
            "days_total": days_remaining,
            "units_covered": len(units_to_study),
            "plan": study_plan,
            "questions_per_day": len(self.questions_df) // days_remaining if days_remaining > 0 else 10,
            "units_to_cover": units_to_study,
            "recommended_schedule": {day["day"]: [{"subject": day["unit_name"], "questions_count": 5}] for day in study_plan}
        }

# Additional functions for compatibility
def show_statistics(assistant):
    """Show statistics about the question database"""
    df = assistant.questions_df
    print("\n" + "="*50)
    print("TNPSC QUESTION DATABASE STATISTICS")
    print("="*50)
    print(f"Total Questions: {len(df)}")
    print(f"Units Covered: {len(df['unit'].unique())}")
    print("\nQuestions per Unit:")
    for unit in sorted(df['unit'].unique()):
        count = len(df[df['unit'] == unit])
        print(f"  Unit {unit} ({UNITS[unit]}): {count}")
    
    print("\nDifficulty Distribution:")
    difficulty_counts = df['difficulty'].value_counts()
    for diff, count in difficulty_counts.items():
        print(f"  {diff.title()}: {count}")
    
    print("\nYear Distribution:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count}")
    print("="*50)

# Placeholder functions for PDF processing
def list_uploaded_pdfs():
    """List all PDFs in the PDF directory"""
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the upload directory.")
        print(f"Please upload PDF files to: {PDF_DIR}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files):
        print(f"{i+1}. {pdf}")
    
    return pdf_files

def upload_pdf_instructions():
    """Provide instructions for uploading PDF files"""
    print("\n=== PDF UPLOAD INSTRUCTIONS ===")
    print(f"To use your own PDF files, please upload them to this directory: {PDF_DIR}")
    print("Steps:")
    print("1. Create the directory if it doesn't exist")
    print(f"   mkdir -p {PDF_DIR}")
    print("2. Copy your PDF files to this directory")
    print(f"   cp /path/to/your/pdf/file.pdf {PDF_DIR}/")
    print("3. Run the list_uploaded_pdfs() function to verify your PDFs are recognized")
    print("4. Process PDFs using parse_all_pdfs() function")
    print("===============================\n")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file - placeholder"""
    print(f"PDF processing not implemented yet for: {pdf_path}")
    return None

def detect_question_pattern(text):
    """Detect question pattern - placeholder"""
    return r'(\d+)[\.\)]\\s+([^\\n]+)'

def parse_options(question_text):
    """Parse options from question text - placeholder"""
    return question_text, {}

def guess_unit_from_content(question_text):
    """Guess unit from content - placeholder"""
    return 1

def improved_parse_previous_paper(filepath, output_filename=None):
    """Parse previous paper - placeholder"""
    print(f"PDF parsing not implemented yet for: {filepath}")
    return []

def parse_all_pdfs():
    """Parse all PDFs - placeholder"""
    print("PDF parsing functionality not implemented yet")
    return []

def process_pdfs_and_save():
    """Process PDFs and save - placeholder"""
    print("PDF processing functionality not implemented yet")
    return []

if __name__ == "__main__":
    # Initialize the assistant
    assistant = TNPSCExamAssistant()
    show_statistics(assistant)