#!/usr/bin/env python3
"""
TNPSC AI Chat Assistant - Final Production Version
Standalone with Semantic Intent Recognition + Natural Language Understanding
"""

import os
import sys
import time
import random
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


class SimpleIntentMatcher:
    """Semantic Intent Recognition using TF-IDF + Cosine Similarity"""
    
    def __init__(self):
        """Initialize intent matcher with training data"""
        self.intents = {
            'quiz': [
                'generate quiz', 'start test', 'practice questions', 'create quiz',
                'give me quiz', 'i want to take quiz', 'quiz me', 'test me',
                'give me questions', 'show me questions', 'i need questions',
                'practice test', 'mock test', 'sample questions', 'gimme quiz',
                'can i get quiz', 'want some questions', 'need practice',
                'generate test', 'make quiz', 'quiz for', 'questions on',
                'give questions about', 'show questions for', 'test on'
            ],
            'stats': [
                'show stats', 'show unit statistics', 'tell me about',
                'show info', 'how many questions', 'statistics for',
                'info about', 'details of', 'show me stats', 'unit info',
                'what about', 'tell about', 'information on', 'stats on',
                'show data', 'unit details', 'how many in', 'count of'
            ],
            'plan': [
                'create plan', 'study plan', 'make schedule', 'i need timetable',
                'day plan', 'how should i prepare', 'preparation plan',
                'schedule for', 'make plan', 'build plan', 'plan for',
                'study schedule', 'prep plan', 'timetable', 'organize study',
                'help me plan', 'create schedule', 'make timetable'
            ],
            'search': [
                'find topic', 'search questions about', 'look for',
                'find questions', 'search for', 'look up', 'find about',
                'search topic', 'questions about', 'find on', 'search on',
                'look for questions', 'find me', 'search me', 'lookup'
            ],
            'units': [
                'show units', 'list units', 'all units', 'what units',
                'show subjects', 'list subjects', 'all subjects', 'available units',
                'which units', 'show all', 'list all', 'what subjects'
            ],
            'insights': [
                'show insights', 'overview', 'summary', 'show all data',
                'overall stats', 'complete info', 'full summary', 'insights',
                'show everything', 'all stats', 'complete overview'
            ],
            'greeting': [
                'hi', 'hello', 'hey', 'good morning', 'good evening',
                'whats up', 'wassup', 'sup', 'yo', 'greetings', 'hola',
                'namaste', 'good afternoon', 'howdy'
            ],
            'goodbye': [
                'bye', 'exit', 'quit', 'see you', 'goodbye', 'later',
                'stop', 'close', 'end', 'leave', 'done', 'finish'
            ],
            'thanks': [
                'thanks', 'thank you', 'appreciate', 'thx', 'ty',
                'thanks a lot', 'thank u', 'grateful', 'cheers'
            ],
            'motivation': [
                'motivate me', 'motivation', 'encourage', 'nervous',
                'scared', 'worried', 'anxious', 'inspire me', 'boost'
            ],
            'help': [
                'help', 'what can you do', 'show commands', 'how to use',
                'guide', 'instructions', 'commands', 'options', 'features'
            ]
        }
        
        # Build training corpus
        self.training_phrases = []
        self.training_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.training_phrases.append(phrase)
                self.training_labels.append(intent)
        
        # Train TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.training_vectors = self.vectorizer.fit_transform(self.training_phrases)
        print("   ✅ Intent Matcher trained with {} intents".format(len(self.intents)))
    
    def predict_intent(self, user_input, threshold=0.25):
        """Predict intent from user input"""
        normalized = user_input.lower().strip()
        input_vector = self.vectorizer.transform([normalized])
        similarities = cosine_similarity(input_vector, self.training_vectors)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.training_labels[best_idx]
        
        return (best_intent, best_score) if best_score >= threshold else ('unknown', best_score)


class TNPSCChatAssistant:
    """Production-Grade Interactive Chat Assistant with Semantic Understanding"""
    
    def __init__(self):
        """Initialize the chat assistant"""
        print("🚀 Initializing TNPSC AI Chat Assistant...")
        print("="*60)
        
        # Load or create sample data
        self._load_data()
        
        # Train ML models
        self._train_models()
        
        # Initialize intent matcher
        self.intent_matcher = SimpleIntentMatcher()
        
        print("✅ Chat Assistant Ready!")
        print("="*60)
    
    def _load_data(self):
        """Load TNPSC questions data"""
        try:
            # Try to load from CSV
            self.df = pd.read_csv('data/TNPSC/questions/sample_questions.csv')
            print(f"   ✅ Loaded {len(self.df)} questions from CSV")
        except:
            # Create sample data
            print("   ⚠️  Creating sample data...")
            sample_data = {
                'unit': [1, 2, 3, 4, 5, 6, 7, 8] * 15,
                'question': [
                    "What is the chemical formula of water?",
                    "Who is the current Prime Minister of India?",
                    "What is the capital of Tamil Nadu?",
                    "When did India gain independence?",
                    "How many articles are in the Indian Constitution?",
                    "What is GDP?",
                    "Who led the Quit India Movement?",
                    "What is 15 + 27?",
                ] * 15,
                'difficulty': np.random.choice(['Easy', 'Medium', 'Hard'], 120),
                'year': np.random.choice([2020, 2021, 2022, 2023], 120),
            }
            self.df = pd.DataFrame(sample_data)
            print(f"   ✅ Created {len(self.df)} sample questions")
    
    def _train_models(self):
        """Train ML models for predictions"""
        print("🔧 Training ML models...")
        
        # Prepare features
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['has_numbers'] = self.df['question'].str.contains(r'\d').astype(int)
        self.df['has_question_mark'] = self.df['question'].str.contains(r'\?').astype(int)
        self.df['avg_word_length'] = self.df['question'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Train difficulty predictor
        X = self.df[['unit', 'year', 'question_length', 'word_count', 
                     'has_numbers', 'has_question_mark', 'avg_word_length']]
        y = self.df['difficulty']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"   ✅ Difficulty Predictor trained (Accuracy: {accuracy:.2%})")
    
    def _thinking_delay(self):
        """Add natural thinking delay"""
        time.sleep(random.uniform(0.2, 0.5))
    
    def _parse_entities(self, user_input):
        """Extract entities like unit numbers, keywords, etc."""
        text = user_input.lower().strip()
        
        # Extract numbers
        numbers = [int(n) for n in re.findall(r'\d+', text)]
        
        # Written numbers
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'fifteen': 15, 'twenty': 20, 'thirty': 30
        }
        for word, num in number_words.items():
            if word in text:
                numbers.append(num)
        
        # Unit mapping
        unit_mapping = {
            'science': 1, 'physics': 1, 'chemistry': 1, 'biology': 1,
            'current': 2, 'events': 2, 'news': 2,
            'geography': 3, 'geo': 3,
            'history': 4, 'culture': 4,
            'polity': 5, 'politics': 5, 'government': 5,
            'economy': 6, 'economic': 6,
            'national': 7, 'movement': 7, 'freedom': 7,
            'mental': 8, 'ability': 8, 'aptitude': 8, 'reasoning': 8
        }
        
        detected_unit = None
        for keyword, unit_num in unit_mapping.items():
            if keyword in text:
                detected_unit = unit_num
                break
        
        if not detected_unit and numbers:
            first_num = numbers[0]
            if 1 <= first_num <= 8:
                detected_unit = first_num
        
        # Extract search keyword - better extraction
        search_keyword = text
        # Remove common command words but keep the actual topic
        remove_patterns = [
            r'\bgive\s+me\b', r'\bshow\s+me\b', r'\bfind\b', r'\bsearch\b',
            r'\blook\s+for\b', r'\bquestions?\b', r'\babout\b', r'\bon\b',
            r'\bfor\b', r'\bme\b', r'\bshow\b', r'\bgive\b', r'\bcan\s+we\b',
            r'\bstart\b', r'\bget\b', r'\bwant\b', r'\bneed\b', r'\bi\b'
        ]
        for pattern in remove_patterns:
            search_keyword = re.sub(pattern, '', search_keyword, flags=re.IGNORECASE)
        search_keyword = re.sub(r'\s+', ' ', search_keyword).strip()
        
        return {
            'numbers': numbers,
            'unit': detected_unit,
            'text': text,
            'search_keyword': search_keyword
        }
    
    def chat(self):
        """Main chat interface"""
        print("\n" + "="*60)
        print("🎓 TNPSC AI EXAM PREPARATION ASSISTANT")
        print("="*60)
        print("\nHey! 👋 I'm your AI buddy for TNPSC exam prep!")
        print("\n✨ I understand natural language! Just talk to me.")
        print("\nTry saying:")
        print("  💬 'give me 5 questions on water'")
        print("  💬 'show me stats for history'")
        print("  💬 'I need a 30 day study plan'")
        print("\nType 'help' anytime!")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Predict intent
                intent, confidence = self.intent_matcher.predict_intent(user_input)
                entities = self._parse_entities(user_input)
                
                self._thinking_delay()
                
                # Route based on intent
                if intent == 'goodbye':
                    farewells = [
                        "🎉 Good luck with your TNPSC prep! You got this!",
                        "👋 See you later! Keep studying hard!",
                        "✨ Goodbye! Go ace that exam!"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(farewells)}")
                    break
                
                elif intent == 'greeting':
                    greetings = [
                        "Hey there! 👋 Ready to crush that TNPSC exam?",
                        "Hello! 😊 Let's get you prepared!",
                        "Hi! 🎯 What can I help you with today?"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(greetings)}")
                
                elif intent == 'thanks':
                    thanks_responses = ["You're welcome! 💪", "Anytime! 🎯", "Happy to help! 🚀"]
                    print(f"\n🤖 Assistant: {random.choice(thanks_responses)}")
                
                elif intent == 'motivation':
                    motivations = [
                        "You've got this! 💪 Every question you practice gets you closer!",
                        "Believe in yourself! 🌟 You're doing great!",
                        "Stay focused! 🎯 Success is just around the corner!"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(motivations)}")
                
                elif intent == 'help':
                    self._show_help()
                
                elif intent == 'quiz' or 'question' in entities['text']:
                    quiz_responses = ["Sure! Let's do a quiz 🔥", "Got it! Generating questions 📝"]
                    print(f"\n🤖 Assistant: {random.choice(quiz_responses)}")
                    self._handle_quiz(entities)
                
                elif intent == 'stats':
                    stats_responses = ["Let me pull up those stats 📊", "Here's the info 📈"]
                    print(f"\n🤖 Assistant: {random.choice(stats_responses)}")
                    self._handle_stats(entities)
                
                elif intent == 'plan':
                    plan_responses = ["Cool! Let's build your plan 📅", "Creating your schedule 🗓️"]
                    print(f"\n🤖 Assistant: {random.choice(plan_responses)}")
                    self._handle_plan(entities)
                
                elif intent == 'search':
                    search_responses = ["Searching for you 🔍", "Let me find that 🔎"]
                    print(f"\n🤖 Assistant: {random.choice(search_responses)}")
                    self._handle_search(entities)
                
                elif intent == 'units':
                    print(f"\n🤖 Assistant: Here are all the units 📚")
                    self._show_units()
                
                elif intent == 'insights':
                    print(f"\n🤖 Assistant: Here's the overview 📊")
                    self._handle_insights()
                
                else:
                    print(f"\n🤖 Assistant: Hmm, not sure I got that. 🤔")
                    print("   Try: 'give me a quiz' or 'show units' or 'help'")
                    
            except KeyboardInterrupt:
                print("\n\n🎉 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Oops! Error: {e}")
    
    def _handle_quiz(self, entities):
        """Handle quiz generation - FIXED: No duplicate questions"""
        unit = entities['unit']
        numbers = entities['numbers']
        keyword = entities['search_keyword']
        
        # Determine number of questions
        num_questions = 5  # default
        for num in numbers:
            if 1 <= num <= 20:  # reasonable quiz size
                num_questions = num
                break
        
        # Filter questions - prioritize unit over keyword if both exist
        if unit:
            # Filter by unit
            filtered = self.df[self.df['unit'] == unit]
        elif keyword and len(keyword) > 2:
            # Search by keyword
            filtered = self.df[self.df['question'].str.contains(keyword, case=False, na=False)]
            if len(filtered) == 0:
                print(f"\n❌ No questions found about '{keyword}'")
                print(f"💡 Try: 'quiz on science' or 'show all units'")
                return
        else:
            # Random from all
            filtered = self.df
        
        # Sample WITHOUT replacement to avoid duplicates
        if len(filtered) < num_questions:
            quiz = filtered
            print(f"\n⚠️  Only {len(filtered)} questions available")
        else:
            quiz = filtered.sample(n=num_questions, replace=False)  # replace=False prevents duplicates
        
        # Display quiz
        if unit:
            print(f"\n📝 QUIZ: {UNITS[unit]}")
        elif keyword:
            print(f"\n📝 QUIZ: Questions about '{keyword}'")
        else:
            print(f"\n📝 QUIZ: Mixed Topics")
        
        print("=" * 60)
        
        for idx, (_, row) in enumerate(quiz.iterrows(), 1):
            print(f"\nQ{idx}. {row['question']}")
            print(f"    Difficulty: {row['difficulty']} | Year: {row['year']}")
        
        print("=" * 60)
        print(f"✅ Generated {len(quiz)} unique questions! Good luck! 🍀")
    
    def _handle_stats(self, entities):
        """Handle stats display"""
        unit = entities['unit']
        
        if not unit:
            print("\n💡 Which unit? Try: 'stats for science'")
            return
        
        unit_data = self.df[self.df['unit'] == unit]
        
        print(f"\n📊 STATISTICS: {UNITS[unit]}")
        print("-" * 60)
        print(f"  Total Questions: {len(unit_data)}")
        print(f"  Year Range: {unit_data['year'].min()} - {unit_data['year'].max()}")
        print(f"\n  Difficulty Breakdown:")
        for diff, count in unit_data['difficulty'].value_counts().items():
            pct = (count / len(unit_data)) * 100
            print(f"    {diff}: {count} ({pct:.1f}%)")
        print("-" * 60)
    
    def _handle_plan(self, entities):
        """Handle study plan creation"""
        numbers = entities['numbers']
        days = numbers[0] if numbers else 30
        
        questions_per_day = len(self.df) // days if days > 0 else 10
        
        print(f"\n📅 STUDY PLAN FOR {days} DAYS")
        print("=" * 60)
        print(f"  Questions per day: {questions_per_day}")
        print(f"  Total units: {len(UNITS)}")
        print(f"\n  Sample Schedule (First 5 Days):")
        
        for day in range(1, min(6, days + 1)):
            unit = ((day - 1) % 8) + 1
            print(f"\n  Day {day}: {UNITS[unit]} ({questions_per_day} questions)")
        
        if days > 5:
            print(f"\n  ... (continues for {days} days)")
        
        print("=" * 60)
        print("✅ Plan created! Stay consistent! 💪")
    
    def _handle_search(self, entities):
        """Handle question search"""
        keyword = entities['search_keyword']
        
        if not keyword or len(keyword) < 2:
            print("\n💡 What topic? Try: 'search water'")
            return
        
        results = self.df[self.df['question'].str.contains(keyword, case=False, na=False)]
        
        print(f"\n🔍 SEARCH: '{keyword}'")
        print("=" * 60)
        
        if len(results) == 0:
            print("  No matches found. Try different keywords! 🔎")
        else:
            print(f"  Found {len(results)} questions:\n")
            for idx, (_, row) in enumerate(results.head(5).iterrows(), 1):
                print(f"{idx}. {row['question']}")
                print(f"   Unit: {UNITS[row['unit']]} | {row['difficulty']}\n")
            
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more!")
        
        print("=" * 60)
    
    def _show_units(self):
        """Show all units"""
        print("\n📖 TNPSC EXAM UNITS:")
        print("-" * 60)
        for unit_num, unit_name in UNITS.items():
            count = len(self.df[self.df['unit'] == unit_num])
            print(f"  Unit {unit_num}: {unit_name} ({count} questions)")
        print("-" * 60)
    
    def _handle_insights(self):
        """Show overall insights"""
        print("\n📈 OVERALL INSIGHTS")
        print("=" * 60)
        print(f"  Total Questions: {len(self.df)}")
        print(f"  Units Covered: {len(self.df['unit'].unique())}")
        print(f"\n  Difficulty Breakdown:")
        for diff, count in self.df['difficulty'].value_counts().items():
            pct = (count / len(self.df)) * 100
            print(f"    {diff}: {count} ({pct:.1f}%)")
        print("=" * 60)
    
    def _show_help(self):
        """Show help"""
        print("\n📚 I CAN HELP YOU WITH:")
        print("-" * 60)
        print("  🎯 Generate quizzes - 'give me 5 questions on water'")
        print("  📊 Show statistics - 'show me stats for history'")
        print("  📅 Create study plans - 'I need a 30 day plan'")
        print("  🔍 Search questions - 'find questions about economy'")
        print("  📖 List units - 'show all units'")
        print("  📈 Show insights - 'give me an overview'")
        print("-" * 60)
        print("Just talk naturally - I'll understand! 😊")


def main():
    """Main entry point"""
    try:
        assistant = TNPSCChatAssistant()
        assistant.chat()
    except KeyboardInterrupt:
        print("\n\n🎉 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
