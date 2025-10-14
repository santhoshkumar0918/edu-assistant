#!/usr/bin/env python3
"""
TNPSC AI Chat Assistant V2 - Production Grade
Semantic Intent Recognition + Natural Language Understanding
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

# Import our core assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS


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
            'predict': [
                'predict difficulty', 'how hard', 'difficulty of',
                'check difficulty', 'predict', 'difficulty level'
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
    
    def normalize_text(self, text):
        """Normalize text for better matching"""
        text = text.lower().strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def predict_intent(self, user_input, threshold=0.25):
        """Predict intent from user input"""
        normalized = self.normalize_text(user_input)
        
        # Vectorize input
        input_vector = self.vectorizer.transform([normalized])
        
        # Calculate cosine similarity with all training phrases
        similarities = cosine_similarity(input_vector, self.training_vectors)[0]
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.training_labels[best_idx]
        
        # Return intent if above threshold
        if best_score >= threshold:
            return best_intent, best_score
        else:
            return 'unknown', best_score


class TNPSCChatAssistantV2:
    """Production-Grade Interactive Chat Assistant with Semantic Understanding"""
    
    def __init__(self):
        """Initialize the chat assistant with ML models and intent recognition"""
        print("🚀 Initializing TNPSC AI Chat Assistant V2...")
        print("="*60)
        
        # Load core assistant
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # Train ML models
        self._train_models()
        
        # Initialize intent matcher
        self.intent_matcher = SimpleIntentMatcher()
        
        # Conversation memory
        self.last_intent = None
        self.last_unit = None
        self.conversation_context = {}
        
        print("✅ Chat Assistant V2 Ready!")
        print("="*60)
    
    def _train_models(self):
        """Train ML models for predictions with enhanced features"""
        print("🔧 Training ML models...")
        
        # Prepare enhanced features
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
        
        self.difficulty_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.difficulty_model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.difficulty_model.predict(X_test))
        print(f"   ✅ Difficulty Predictor trained (Accuracy: {accuracy:.2%})")
    
    def _thinking_delay(self):
        """Add natural thinking delay"""
        time.sleep(random.uniform(0.2, 0.5))
    
    def _parse_entities(self, user_input):
        """Extract entities like unit numbers, keywords, etc."""
        import re
        
        text = user_input.lower().strip()
        
        # Extract numbers
        numbers = [int(n) for n in re.findall(r'\d+', text)]
        
        # Written numbers
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }
        for word, num in number_words.items():
            if word in text:
                numbers.append(num)
        
        # Unit mapping
        unit_mapping = {
            'science': 1, 'general science': 1, 'sci': 1, 'physics': 1, 
            'chemistry': 1, 'biology': 1, 'bio': 1, 'chem': 1,
            'current': 2, 'current events': 2, 'events': 2, 'news': 2,
            'current affairs': 2, 'affairs': 2, 'recent': 2,
            'geography': 3, 'geo': 3, 'places': 3, 'location': 3,
            'history': 4, 'culture': 4, 'historical': 4,
            'polity': 5, 'politics': 5, 'government': 5, 'constitution': 5,
            'economy': 6, 'economic': 6, 'economics': 6, 'gdp': 6,
            'national': 7, 'movement': 7, 'freedom': 7, 'independence': 7,
            'mental': 8, 'ability': 8, 'aptitude': 8, 'reasoning': 8, 'logic': 8
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
        
        return {
            'numbers': numbers,
            'unit': detected_unit,
            'text': text
        }
    
    def chat(self):
        """Main chat interface with semantic understanding"""
        print("\n" + "="*60)
        print("🎓 TNPSC AI EXAM PREPARATION ASSISTANT V2")
        print("="*60)
        print("\nHey! 👋 I'm your AI buddy for TNPSC exam prep!")
        print("\n✨ I understand natural language! Just talk to me like a friend.")
        print("\nTry saying:")
        print("  💬 'give me a quiz on science'")
        print("  💬 'show me stats for history'")
        print("  💬 'I need a 30 day study plan'")
        print("  💬 'find questions about water'")
        print("\nType 'help' anytime if you're stuck!")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Predict intent using semantic matcher
                intent, confidence = self.intent_matcher.predict_intent(user_input)
                
                # Parse entities
                entities = self._parse_entities(user_input)
                
                # Add thinking delay for natural feel
                self._thinking_delay()
                
                # Route based on intent
                if intent == 'goodbye':
                    farewells = [
                        "🎉 Good luck with your TNPSC prep! You got this!",
                        "👋 See you later! Keep studying hard!",
                        "✨ Goodbye! Go ace that exam!",
                        "🚀 Catch you later! Stay focused!"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(farewells)}")
                    break
                
                elif intent == 'greeting':
                    greetings = [
                        "Hey there! 👋 Ready to crush that TNPSC exam? What do you need?",
                        "Hello! 😊 Let's get you prepared! Want a quiz or study plan?",
                        "Hi! 🎯 What can I help you with today?",
                        "Yo! 🔥 Let's make today productive! What's up?"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(greetings)}")
                
                elif intent == 'thanks':
                    thanks_responses = [
                        "You're welcome! Keep crushing it! 💪",
                        "No problem! You got this! 🎯",
                        "Anytime! Keep up the awesome work! ⭐",
                        "Happy to help! Go ace that exam! 🚀",
                        "My pleasure! Stay consistent! 🌟"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(thanks_responses)}")
                
                elif intent == 'motivation':
                    motivations = [
                        "You've got this! 💪 Every question you practice gets you closer to success!",
                        "Believe in yourself! 🌟 You're putting in the work, and it will pay off!",
                        "Stay focused and consistent! 🎯 Success is just around the corner!",
                        "Don't worry! 😊 With practice and dedication, you'll ace this exam!",
                        "You're doing great! 🚀 Keep pushing forward, one day at a time!"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(motivations)}")
                
                elif intent == 'help':
                    self._show_help()
                
                elif intent == 'quiz':
                    confirmations = ["Sure! Let's do a quick quiz 🔥", "Alright! Quiz time 📝", 
                                   "Got it! Generating your quiz 🎯", "Cool! Let's test your knowledge 💡"]
                    print(f"\n🤖 Assistant: {random.choice(confirmations)}")
                    self._handle_quiz(entities)
                    self.last_intent = 'quiz'
                
                elif intent == 'stats':
                    confirmations = ["Let me pull up those stats 📊", "Sure thing! Here's the info 📈",
                                   "Got it! Checking the data 🔍", "Alright! Let me show you 📋"]
                    print(f"\n🤖 Assistant: {random.choice(confirmations)}")
                    self._handle_stats(entities)
                    self.last_intent = 'stats'
                
                elif intent == 'plan':
                    confirmations = ["Cool! Let's build your study plan 📅", "Awesome! Planning time 🗓️",
                                   "Got it! Creating your schedule 📆", "Perfect! Let's organize your prep 📋"]
                    print(f"\n🤖 Assistant: {random.choice(confirmations)}")
                    self._handle_plan(entities)
                    self.last_intent = 'plan'
                
                elif intent == 'search':
                    confirmations = ["Alright! Searching for you 🔍", "Let me find that 🔎",
                                   "Got it! Looking it up 💡", "Sure! Searching now 🔦"]
                    print(f"\n🤖 Assistant: {random.choice(confirmations)}")
                    self._handle_search(entities, user_input)
                    self.last_intent = 'search'
                
                elif intent == 'units':
                    print(f"\n🤖 Assistant: Here are all the TNPSC units 📚")
                    self._show_units()
                
                elif intent == 'insights':
                    print(f"\n🤖 Assistant: Here's the complete overview 📊")
                    self._handle_insights()
                
                elif intent == 'predict':
                    print(f"\n🤖 Assistant: Let's predict the difficulty 🔮")
                    self._handle_predict()
                
                else:
                    # Unknown intent - provide helpful suggestions
                    suggestions = [
                        "Hmm, I'm not sure I got that. 🤔",
                        "I didn't quite catch that. 😅",
                        "Not sure what you mean. 🧐"
                    ]
                    print(f"\n🤖 Assistant: {random.choice(suggestions)}")
                    print("   Try saying:")
                    print("   • 'give me a quiz on science'")
                    print("   • 'show me history stats'")
                    print("   • 'I need a 30 day plan'")
                    print("   • 'find questions about water'")
                    print(f"\n   (Confidence: {confidence:.2f} - I'm still learning! 😊)")
                    
            except KeyboardInterrupt:
                print("\n\n🎉 Good luck with your TNPSC preparation! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Oops! Something went wrong: {e}")
                print("   Let's try that again! 😊")
    
    def _handle_quiz(self, entities):
        """Handle quiz generation"""
        unit = entities['unit']
        numbers = entities['numbers']
        
        if not unit:
            print("\n💡 Which unit would you like? Try:")
            print("   'quiz on science' or 'quiz for unit 1'")
            return
        
        num_questions = numbers[1] if len(numbers) > 1 else (numbers[0] if numbers and numbers[0] <= 20 else 5)
        
        quiz = self.df[self.df['unit'] == unit].sample(n=min(num_questions, len(self.df[self.df['unit'] == unit])))
        
        print(f"\n📝 QUIZ: {UNITS[unit]}")
        print("=" * 60)
        
        for idx, (_, row) in enumerate(quiz.iterrows(), 1):
            print(f"\nQ{idx}. {row['question']}")
            print(f"    Difficulty: {row['difficulty']} | Year: {row['year']}")
        
        print("=" * 60)
        print(f"✅ Generated {len(quiz)} questions! Good luck! 🍀")
    
    def _handle_stats(self, entities):
        """Handle stats display"""
        unit = entities['unit']
        
        if not unit:
            print("\n💡 Which unit? Try: 'stats for science' or 'stats 1'")
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
        
        if not numbers:
            print("\n💡 How many days? Try: 'create 30 day plan'")
            return
        
        days = numbers[0]
        plan = self.assistant.generate_study_plan(days)
        
        print(f"\n📅 STUDY PLAN FOR {days} DAYS")
        print("=" * 60)
        print(f"  Questions per day: {plan['questions_per_day']}")
        print(f"\n  First 7 Days Schedule:")
        
        for day in sorted(plan['recommended_schedule'].keys())[:7]:
            print(f"\n  Day {day}:")
            for item in plan['recommended_schedule'][day]:
                print(f"    • {item['subject']} ({item['questions_count']} questions)")
        
        if days > 7:
            print(f"\n  ... (continues for {days} days)")
        
        print("=" * 60)
        print("✅ Plan created! Stay consistent! 💪")
    
    def _handle_search(self, entities, user_input):
        """Handle question search"""
        # Extract keyword
        text = entities['text']
        for word in ['search', 'find', 'look', 'for', 'about', 'on', 'questions']:
            text = text.replace(word, '')
        keyword = text.strip()
        
        if not keyword:
            print("\n💡 What topic? Try: 'search water' or 'find economy questions'")
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
    
    def _handle_predict(self):
        """Handle difficulty prediction"""
        print("\n🔮 DIFFICULTY PREDICTION")
        print("-" * 60)
        try:
            unit = int(input("  Unit (1-8): "))
            year = int(input("  Year: "))
            question = input("  Question: ")
            
            difficulty = self.predict_difficulty(unit, year, question)
            print(f"\n  ✅ Predicted: {difficulty}")
            print(f"  📚 Unit: {UNITS[unit]}")
        except:
            print("  ❌ Invalid input")
        print("-" * 60)
    
    def predict_difficulty(self, unit, year, question_text):
        """Predict difficulty"""
        question_length = len(question_text)
        word_count = len(question_text.split())
        has_numbers = 1 if any(c.isdigit() for c in question_text) else 0
        has_question_mark = 1 if '?' in question_text else 0
        avg_word_length = np.mean([len(w) for w in question_text.split()]) if word_count > 0 else 0
        
        features = [[unit, year, question_length, word_count, has_numbers, has_question_mark, avg_word_length]]
        return self.difficulty_model.predict(features)[0]
    
    def _show_help(self):
        """Show help"""
        print("\n📚 I CAN HELP YOU WITH:")
        print("-" * 60)
        print("  🎯 Generate quizzes - 'give me a quiz on science'")
        print("  📊 Show statistics - 'show me stats for history'")
        print("  📅 Create study plans - 'I need a 30 day plan'")
        print("  🔍 Search questions - 'find questions about water'")
        print("  📖 List units - 'show all units'")
        print("  📈 Show insights - 'give me an overview'")
        print("  🔮 Predict difficulty - 'predict difficulty'")
        print("-" * 60)
        print("Just talk naturally - I'll understand! 😊")


def main():
    """Main entry point"""
    try:
        assistant = TNPSCChatAssistantV2()
        assistant.chat()
    except KeyboardInterrupt:
        print("\n\n🎉 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
