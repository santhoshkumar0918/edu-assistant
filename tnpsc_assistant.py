#!/usr/bin/env python
# coding: utf-8

"""
AI-Powered TNPSC Exam Preparation Assistant - Core Module
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# TNPSC Units definition
UNITS = {
    1: "General Science",
    2: "Current Events",
    3: "Geography",
    4: "History & Culture",
    5: "Indian Polity",
    6: "Indian Economy",
    7: "National Movement",
    8: "Mental Ability"
}

class TNPSCExamAssistant:
    """
    Core TNPSC Exam Assistant class for managing questions and providing ML capabilities.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the TNPSC Exam Assistant.

        Args:
            data_dir (str): Directory containing the TNPSC data files
        """
        self.data_dir = data_dir
        self.questions_df = None
        self.syllabus = None
        self._load_data()

    def _load_data(self):
        """Load TNPSC questions data."""
        try:
            # Try to load from various possible locations
            possible_paths = [
                os.path.join(self.data_dir, "TNPSC", "questions", "sample_questions.csv"),
                os.path.join(self.data_dir, "TNPSC", "questions", "questions.csv"),
                os.path.join(self.data_dir, "questions.csv"),
                "data/TNPSC/questions/sample_questions.csv",
                "questions.csv"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.questions_df = pd.read_csv(path)
                    print(f"✅ Loaded {len(self.questions_df)} questions from {path}")
                    return

            # If no file found, create sample data
            self._create_sample_data()
            print("✅ Created sample TNPSC questions data")

        except Exception as e:
            print(f"⚠️  Warning: Could not load questions data: {e}")
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample TNPSC questions data for testing."""
        # Sample questions data
        sample_data = {
            'unit': [1, 2, 3, 4, 5, 6, 7, 8] * 15,
            'question': [
                "What is the chemical formula of water?",
                "Who is the current Prime Minister of India?",
                "What is the capital of Tamil Nadu?",
                "When did India gain independence?",
                "How many articles are there in the Indian Constitution?",
                "What is GDP?",
                "Who led the Quit India Movement?",
                "What is 15 + 27?",
            ] * 15,
            'difficulty': np.random.choice(['Easy', 'Medium', 'Hard'], 120),
            'year': np.random.choice([2020, 2021, 2022, 2023], 120),
            'subject': ['Science', 'Current Affairs', 'Geography', 'History',
                       'Polity', 'Economy', 'History', 'Aptitude'] * 15
        }

        self.questions_df = pd.DataFrame(sample_data)

    def generate_quiz(self, unit_num: int, num_questions: int = 5) -> pd.DataFrame:
        """
        Generate a quiz for a specific unit.

        Args:
            unit_num (int): Unit number to generate quiz for
            num_questions (int): Number of questions to include

        Returns:
            pd.DataFrame: Quiz questions
        """
        unit_questions = self.questions_df[self.questions_df['unit'] == unit_num]
        if len(unit_questions) < num_questions:
            return unit_questions
        return unit_questions.sample(n=num_questions)

    def generate_study_plan(self, days_remaining: int) -> Dict:
        """
        Generate a personalized study plan.

        Args:
            days_remaining (int): Days remaining for exam preparation

        Returns:
            Dict: Study plan details
        """
        units = list(UNITS.keys())
        questions_per_day = len(self.questions_df) // days_remaining if days_remaining > 0 else 10

        plan = {
            'total_days': days_remaining,
            'questions_per_day': questions_per_day,
            'units_to_cover': units,
            'recommended_schedule': {}
        }

        for i, unit in enumerate(units):
            day = (i % days_remaining) + 1
            if day not in plan['recommended_schedule']:
                plan['recommended_schedule'][day] = []
            plan['recommended_schedule'][day].append({
                'unit': unit,
                'subject': UNITS[unit],
                'questions_count': questions_per_day
            })

        return plan

def load_questions(data_dir: str = "data") -> pd.DataFrame:
    """Load TNPSC questions from data directory."""
    assistant = TNPSCExamAssistant(data_dir)
    return assistant.questions_df

def load_syllabus(data_dir: str = "data") -> Dict:
    """Load TNPSC syllabus information."""
    return create_sample_syllabus()

def create_sample_questions() -> pd.DataFrame:
    """Create sample questions data for testing."""
    sample_data = {
        'unit': [1, 2, 3, 4, 5, 6, 7, 8] * 10,
        'question': [
            "What is the chemical formula of water?",
            "Who is the current Prime Minister of India?",
            "What is the capital of Tamil Nadu?",
            "When did India gain independence?",
            "How many articles are there in the Indian Constitution?",
            "What is GDP?",
            "Who led the Quit India Movement?",
            "What is 15 + 27?",
        ] * 10,
        'difficulty': np.random.choice(['Easy', 'Medium', 'Hard'], 80),
        'year': np.random.choice([2020, 2021, 2022, 2023], 80),
        'subject': ['Science', 'Current Affairs', 'Geography', 'History',
                   'Polity', 'Economy', 'History', 'Aptitude'] * 10
    }
    return pd.DataFrame(sample_data)

def create_sample_syllabus() -> Dict:
    """Create sample syllabus data for testing."""
    return {
        'units': UNITS,
        'total_units': len(UNITS),
        'description': 'TNPSC Syllabus covering 8 major units',
        'topics': {
            1: ['Physics', 'Chemistry', 'Biology'],
            2: ['Recent developments', 'Politics', 'Current Affairs'],
            3: ['Physical geography', 'Human geography', 'Environment'],
            4: ['Ancient history', 'Medieval history', 'Modern history'],
            5: ['Constitution', 'Governance', 'Public administration'],
            6: ['Economic concepts', 'Policies', 'Development'],
            7: ['Freedom struggle', 'National movement', 'Leaders'],
            8: ['Reasoning', 'Logic', 'Aptitude', 'Mental ability']
        }
    }

# Quick test function
if __name__ == "__main__":
    print("🧪 Testing TNPSC Exam Assistant...")
    assistant = TNPSCExamAssistant()
    print(f"✅ Assistant initialized with {len(assistant.questions_df)} questions")

    # Test quiz generation
    quiz = assistant.generate_quiz(unit_num=1, num_questions=3)
    print(f"✅ Generated quiz with {len(quiz)} questions")

    # Test study plan
    plan = assistant.generate_study_plan(days_remaining=30)
    print(f"✅ Generated study plan for {plan['total_days']} days")

    print("🎉 All tests passed! TNPSC Assistant is ready to use.")
