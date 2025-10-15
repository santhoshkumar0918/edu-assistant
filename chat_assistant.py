#!/usr/bin/env python3
"""
TNPSC AI Study Buddy - Production-Grade Conversational Assistant
================================================================================
Enhanced with:
- Semantic NLU (sentence-transformers + TF-IDF fallback)
- Context retention & conversational memory
- Natural, human-like responses
- Error resilience & debug logging
- Offline-first architecture
- Personalized user experience

Author: ML Engineering Team
Version: 2.0 (Production)
================================================================================
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

from tnpsc_assistant import TNPSCExamAssistant, UNITS


class HybridIntentMatcher:
    """
    Hybrid Intent Detection System
    - Primary: Sentence Transformers (semantic embeddings)
    - Fallback: TF-IDF + Cosine Similarity
    """
    
    def __init__(self, use_transformers=True):
        """Initialize with hybrid approach"""
        self.use_transformers = use_transformers
        self.transformer_model = None
        
        # Intent training data
        self.intents = {
            'quiz': [
                'generate quiz', 'quiz me', 'give me questions', 'start practice',
                'I want to practice', 'test me', 'practice questions', 'create quiz',
                'give me quiz', 'show questions', 'quiz for', 'questions on',
                'I need questions', 'practice test', 'mock test', 'sample questions'
            ],
            'stats': [
                'show stats', 'show unit statistics', 'tell me about', 'unit info',
                'how many questions', 'statistics for', 'info about', 'show me stats',
                'unit details', 'what about', 'show data', 'details of'
            ],
            'plan': [
                'create study plan', 'make schedule', '30 day plan', 'I need a timetable',
                'help me prepare', 'study plan', 'preparation plan', 'make plan',
                'schedule for', 'organize study', 'plan for exam', 'study schedule'
            ],
            'search': [
                'search topic', 'find question', 'look for', 'find questions about',
                'search for', 'lookup', 'find me', 'search questions', 'look up'
            ],
            'greeting': [
                'hi', 'hello', 'hey', 'good morning', 'what\'s up', 'yo',
                'wassup', 'greetings', 'good evening', 'good afternoon', 'howdy'
            ],
            'goodbye': [
                'bye', 'exit', 'quit', 'see you', 'stop', 'goodbye',
                'later', 'close', 'end', 'leave', 'done'
            ],
            'motivation': [
                'I\'m bored', 'nervous', 'demotivated', 'motivate me', 'I feel lazy',
                'tired', 'stressed', 'anxious', 'worried', 'inspire me', 'encourage'
            ],
            'help': [
                'help', 'what can you do', 'show commands', 'I\'m lost',
                'guide me', 'instructions', 'how to use', 'features'
            ],
            'thanks': [
                'thanks', 'thank you', 'appreciate', 'thx', 'ty', 'grateful'
            ],
            'affirmative': [
                'yes', 'okay', 'ok', 'sure', 'continue', 'go on', '1',
                'yeah', 'yep', 'yup', 'alright', 'fine'
            ]
        }
        
        # Try to load sentence transformers
        if self.use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                print("   🧠 Loading semantic model (all-MiniLM-L6-v2)...")
                self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._precompute_embeddings()
                print("   ✅ Semantic NLU ready!")
            except ImportError:
                print("   ⚠️  sentence-transformers not found, using TF-IDF fallback")
                self.use_transformers = False
                self._init_tfidf()
        else:
            self._init_tfidf()
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all intent examples"""
        self.intent_embeddings = {}
        for intent, examples in self.intents.items():
            self.intent_embeddings[intent] = self.transformer_model.encode(examples)
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback"""
        print("   📊 Initializing TF-IDF intent matcher...")
        self.training_phrases = []
        self.training_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.training_phrases.append(phrase)
                self.training_labels.append(intent)
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=500)
        self.training_vectors = self.vectorizer.fit_transform(self.training_phrases)
        print("   ✅ TF-IDF NLU ready!")
    
    def clean_input(self, text):
        """Clean and normalize input text"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text
    
    def detect_intent(self, user_input):
        """
        Detect intent using hybrid approach
        Returns: (intent, confidence_score)
        """
        cleaned = self.clean_input(user_input)
        
        if self.use_transformers:
            return self._detect_with_transformers(cleaned)
        else:
            return self._detect_with_tfidf(cleaned)
    
    def _detect_with_transformers(self, text):
        """Semantic intent detection using sentence transformers"""
        user_embedding = self.transformer_model.encode([text])[0]
        
        best_intent = None
        best_score = 0.0
        
        for intent, embeddings in self.intent_embeddings.items():
            # Calculate cosine similarity with all examples
            similarities = cosine_similarity([user_embedding], embeddings)[0]
            max_sim = np.max(similarities)
            
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent
        
        threshold = 0.35
        if best_score >= threshold:
            return best_intent, best_score
        else:
            return 'unknown', best_score
    
    def _detect_with_tfidf(self, text):
        """TF-IDF based intent detection"""
        input_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(input_vector, self.training_vectors)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.training_labels[best_idx]
        
        threshold = 0.25
        if best_score >= threshold:
            return best_intent, best_score
        else:
            return 'unknown', best_score


class TNPSCStudyBuddy:
    """
    Production-Grade Conversational AI Study Assistant
    Context-aware, natural, and intelligent
    """
    
    def __init__(self, debug_mode=False):
        """Initialize the study buddy"""
        self.debug_mode = debug_mode
        
        print("🚀 Initializing TNPSC AI Study Buddy...")
        print("="*70)
        
        # Load data
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # Train ML models
        self._train_models()
        
        # Initialize intent matcher
        self.intent_matcher = HybridIntentMatcher(use_transformers=True)
        
        # Context & memory
        self.username = None
        self.last_intent = None
        self.last_unit = None
        self.last_entities = {}
        self.conversation_history = []
        
        print("✅ Study Buddy Ready!")
        print("="*70)
    
    def _train_models(self):
        """Train ML models for difficulty prediction"""
        if self.debug_mode:
            print("🔧 Training ML models...")
        
        # Feature engineering
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['has_numbers'] = self.df['question'].str.contains(r'\d').astype(int)
        self.df['has_question_mark'] = self.df['question'].str.contains(r'\?').astype(int)
        self.df['avg_word_length'] = self.df['question'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Train model
        X = self.df[['unit', 'year', 'question_length', 'word_count', 
                     'has_numbers', 'has_question_mark', 'avg_word_length']]
        y = self.df['difficulty']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        
        if self.debug_mode:
            print(f"   ✅ ML Model Accuracy: {accuracy:.2%}")
    
    def _typing_delay(self, min_delay=0.3, max_delay=0.8):
        """Simulate natural typing delay"""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def _parse_entities(self, user_input):
        """Extract entities: units, numbers, keywords"""
        text = user_input.lower().strip()
        
        # Extract numbers
        numbers = [int(n) for n in re.findall(r'\d+', text)]
        
        # Number words
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
            'science': 1, 'physics': 1, 'chemistry': 1, 'biology': 1,
            'current': 2, 'events': 2, 'news': 2, 'affairs': 2,
            'geography': 3, 'geo': 3, 'places': 3,
            'history': 4, 'culture': 4, 'historical': 4,
            'polity': 5, 'politics': 5, 'government': 5, 'constitution': 5,
            'economy': 6, 'economic': 6, 'gdp': 6, 'finance': 6,
            'national': 7, 'movement': 7, 'freedom': 7,
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
        
        # Extract keyword
        search_keyword = text
        remove_patterns = [
            r'\bgive\s+me\b', r'\bshow\s+me\b', r'\bfind\b', r'\bsearch\b',
            r'\blook\s+for\b', r'\bquestions?\b', r'\babout\b', r'\bon\b',
            r'\bfor\b', r'\bme\b', r'\bshow\b', r'\bgive\b'
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
    
    def _get_natural_response(self, intent):
        """Get natural, randomized responses"""
        responses = {
            'quiz': [
                "Let's warm up your brain 🧠",
                "Time to test your knowledge 🔥",
                "Alright, quiz mode activated 📝",
                "Let's see what you've got 💪"
            ],
            'stats': [
                "Here's what I found 📊",
                "Let me pull up those numbers 📈",
                "Checking the data for you 🔍",
                "Here are your stats 📋"
            ],
            'plan': [
                "Let's design your prep roadmap 🚀",
                "Time to plan your success 📅",
                "Building your study schedule 🗓️",
                "Let's map out your journey 🎯"
            ],
            'search': [
                "Searching for you 🔎",
                "Let me find that 🔍",
                "Looking it up now 💡",
                "On it! Searching... 🔦"
            ],
            'motivation': [
                "You've got this 💪 every minute counts!",
                "Believe in yourself 🌟 you're doing great!",
                "Stay focused 🎯 success is near!",
                "Keep pushing 🚀 you're stronger than you think!"
            ],
            'greeting': [
                "Hey there! Ready to ace your exam? 👋",
                "Hello! Let's make today productive 😊",
                "Hi! What can I help you with? 🎯",
                "Yo! Let's get studying 🔥"
            ],
            'thanks': [
                "You're welcome! Keep crushing it 💪",
                "Anytime! You got this 🎯",
                "Happy to help! Stay awesome ⭐",
                "My pleasure! Keep going 🚀"
            ]
        }
        
        return random.choice(responses.get(intent, ["Got it!"]))
    
    def chat(self):
        """Main conversational interface"""
        # ASCII Banner
        print("\n" + "="*70)
        print("🎓 TNPSC AI STUDY BUDDY - Your Personal Exam Coach")
        print("="*70)
        
        # Get username
        self.username = input("\n👤 What's your name? ").strip() or "Friend"
        print(f"\nHi {self.username}! I'll track your progress as we go 👀")
        
        print("\n✨ Just talk naturally - I understand you!")
        print("Try: 'give me 5 questions on economy' or 'I need a study plan'")
        print("\nType 'help' anytime or 'bye' to exit")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n💬 {self.username}: ").strip()
                
                if not user_input:
                    continue
                
                # Detect intent
                intent, confidence = self.intent_matcher.detect_intent(user_input)
                entities = self._parse_entities(user_input)
                
                # Debug logging
                if self.debug_mode:
                    print(f"[DEBUG] Intent: {intent}, Confidence: {confidence:.3f}")
                    print(f"[DEBUG] Entities: {entities}")
                
                # Store in history
                self.conversation_history.append({
                    'input': user_input,
                    'intent': intent,
                    'confidence': confidence
                })
                
                # Typing delay for natural feel
                self._typing_delay()
                
                # Handle affirmative (follow-up)
                if intent == 'affirmative' and self.last_intent:
                    print(f"\n🤖 Buddy: Sure! Continuing with {self.last_intent}...")
                    intent = self.last_intent
                    entities = self.last_entities
                
                # Route to handlers
                if intent == 'goodbye':
                    farewells = [
                        f"Goodbye {self.username}! You're going to ace this 🎉",
                        f"See you later {self.username}! Keep studying 👋",
                        f"Catch you later! Stay focused {self.username} ✨"
                    ]
                    print(f"\n🤖 Buddy: {random.choice(farewells)}")
                    break
                
                elif intent == 'greeting':
                    print(f"\n🤖 Buddy: {self._get_natural_response('greeting')}")
                
                elif intent == 'thanks':
                    print(f"\n🤖 Buddy: {self._get_natural_response('thanks')}")
                
                elif intent == 'motivation':
                    print(f"\n🤖 Buddy: {self._get_natural_response('motivation')}")
                
                elif intent == 'help':
                    self._show_help()
                
                elif intent == 'quiz':
                    print(f"\n🤖 Buddy: {self._get_natural_response('quiz')}")
                    self._handle_quiz(entities)
                    self.last_intent = 'quiz'
                    self.last_entities = entities
                
                elif intent == 'stats':
                    print(f"\n🤖 Buddy: {self._get_natural_response('stats')}")
                    self._handle_stats(entities)
                    self.last_intent = 'stats'
                    self.last_entities = entities
                
                elif intent == 'plan':
                    print(f"\n🤖 Buddy: {self._get_natural_response('plan')}")
                    self._handle_plan(entities)
                    self.last_intent = 'plan'
                    self.last_entities = entities
                
                elif intent == 'search':
                    print(f"\n🤖 Buddy: {self._get_natural_response('search')}")
                    self._handle_search(entities)
                    self.last_intent = 'search'
                    self.last_entities = entities
                
                else:
                    # Conversational recovery
                    recovery_msgs = [
                        f"Hmm, that didn't click 🤔 — want me to quiz you or show your stats?",
                        f"Not quite sure what you mean {self.username} 😅 Try 'quiz me' or 'show stats'",
                        f"I'm still learning! 🧠 Try: 'give me questions' or 'create study plan'"
                    ]
                    print(f"\n🤖 Buddy: {random.choice(recovery_msgs)}")
                    print(f"   💡 Suggestions: 'quiz on science' | 'show stats' | 'help'")
                    
            except KeyboardInterrupt:
                print(f"\n\n🎉 Goodbye {self.username}!")
                break
            except Exception as e:
                print(f"\n❌ Oops! Something went wrong: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
    
    def _handle_quiz(self, entities):
        """Generate quiz"""
        unit = entities['unit'] or self.last_unit
        numbers = entities['numbers']
        keyword = entities['search_keyword']
        
        num_questions = 5
        for num in numbers:
            if 1 <= num <= 20:
                num_questions = num
                break
        
        # Filter
        if unit:
            filtered = self.df[self.df['unit'] == unit]
            self.last_unit = unit
        elif keyword and len(keyword) > 2:
            filtered = self.df[self.df['question'].str.contains(keyword, case=False, na=False)]
            if len(filtered) == 0:
                print(f"\n❌ No questions found about '{keyword}'")
                print(f"💡 Try: 'quiz on {UNITS[1]}' or 'show all units'")
                return
        else:
            filtered = self.df
        
        # Sample
        if len(filtered) < num_questions:
            quiz = filtered
        else:
            quiz = filtered.sample(n=num_questions, replace=False)
        
        # Display
        if unit:
            print(f"\n📝 QUIZ: {UNITS[unit]}")
        elif keyword:
            print(f"\n📝 QUIZ: '{keyword}'")
        else:
            print(f"\n📝 QUIZ: Mixed Topics")
        
        print("="*70)
        for idx, (_, row) in enumerate(quiz.iterrows(), 1):
            print(f"\nQ{idx}. {row['question']}")
            print(f"    Difficulty: {row['difficulty']} | Year: {row['year']}")
        print("="*70)
        print(f"✅ {len(quiz)} unique questions! Good luck {self.username}! 🍀")
    
    def _handle_stats(self, entities):
        """Show statistics"""
        unit = entities['unit'] or self.last_unit
        
        if not unit:
            print(f"\n💡 Which unit {self.username}? Try: 'stats for science'")
            return
        
        self.last_unit = unit
        unit_data = self.df[self.df['unit'] == unit]
        
        print(f"\n📊 STATISTICS: {UNITS[unit]}")
        print("-"*70)
        print(f"  Total Questions: {len(unit_data)}")
        print(f"  Year Range: {unit_data['year'].min()} - {unit_data['year'].max()}")
        print(f"\n  Difficulty Breakdown:")
        for diff, count in unit_data['difficulty'].value_counts().items():
            pct = (count / len(unit_data)) * 100
            print(f"    {diff}: {count} ({pct:.1f}%)")
        print("-"*70)
    
    def _handle_plan(self, entities):
        """Create study plan"""
        numbers = entities['numbers']
        days = numbers[0] if numbers else 30
        
        questions_per_day = len(self.df) // days if days > 0 else 10
        
        print(f"\n📅 STUDY PLAN FOR {days} DAYS")
        print("="*70)
        print(f"  Questions per day: {questions_per_day}")
        print(f"  Total units: {len(UNITS)}")
        print(f"\n  Sample Schedule (First 5 Days):")
        
        for day in range(1, min(6, days + 1)):
            unit = ((day - 1) % 8) + 1
            print(f"\n  Day {day}: {UNITS[unit]} ({questions_per_day} questions)")
        
        if days > 5:
            print(f"\n  ... (continues for {days} days)")
        
        print("="*70)
        print(f"✅ Plan ready {self.username}! Stay consistent 💪")
    
    def _handle_search(self, entities):
        """Search questions"""
        keyword = entities['search_keyword']
        
        if not keyword or len(keyword) < 2:
            print(f"\n💡 What topic {self.username}? Try: 'search water'")
            return
        
        results = self.df[self.df['question'].str.contains(keyword, case=False, na=False)]
        
        print(f"\n🔍 SEARCH: '{keyword}'")
        print("="*70)
        
        if len(results) == 0:
            print("  No matches found. Try different keywords! 🔎")
        else:
            print(f"  Found {len(results)} questions:\n")
            for idx, (_, row) in enumerate(results.head(5).iterrows(), 1):
                print(f"{idx}. {row['question']}")
                print(f"   Unit: {UNITS[row['unit']]} | {row['difficulty']}\n")
            
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more!")
        
        print("="*70)
    
    def _show_help(self):
        """Show help"""
        print(f"\n📚 HEY {self.username.upper()}, HERE'S WHAT I CAN DO:")
        print("-"*70)
        print("  🎯 Quiz - 'give me 5 questions on economy'")
        print("  📊 Stats - 'show me stats for history'")
        print("  📅 Plan - 'I need a 30 day study plan'")
        print("  🔍 Search - 'find questions about water'")
        print("  💪 Motivation - 'I'm feeling lazy'")
        print("  👋 Greeting - 'hi' or 'hello'")
        print("  🚪 Exit - 'bye' or 'quit'")
        print("-"*70)
        print("Just talk naturally - I'll understand! 😊")


def main():
    """Entry point"""
    try:
        # Set debug_mode=True to see confidence scores
        buddy = TNPSCStudyBuddy(debug_mode=False)
        buddy.chat()
    except KeyboardInterrupt:
        print("\n\n🎉 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
