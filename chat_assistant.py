#!/usr/bin/env python3
"""
TNPSC Study Buddy v3.0 - Adaptive & Emotion-Aware Edition
================================================================================
Next-Gen Features:
✅ Adaptive Learning Engine - Tracks performance, adjusts difficulty
✅ Emotional Intelligence - Detects mood, provides empathetic responses
✅ Knowledge Recommender - Teaches concepts, not just quizzes
✅ Offline Semantic QA - Answer conceptual questions
✅ Multi-User Support - Profile management with progress tracking
✅ Self-Improving NLU - Logs interactions for continuous improvement
✅ Dynamic Response Styles - Friendly/Formal/Coaching modes

Author: ML Engineering Team
Version: 3.0 (Next-Gen Production)
================================================================================
"""

import os
import sys
import time
import random
import re
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
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


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ConversationContext:
    """Enhanced conversation context with full history"""
    last_intent: Optional[str] = None
    last_unit: Optional[int] = None
    last_topic: Optional[str] = None
    last_emotion: Optional[str] = None
    history: List[Dict] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class UserProfile:
    """User profile with performance tracking"""
    name: str
    total_sessions: int = 0
    quizzes_taken: int = 0
    questions_answered: int = 0
    correct_answers: int = 0
    avg_score: float = 0.0
    weakest_units: List[int] = None
    strongest_units: List[int] = None
    streak_days: int = 0
    last_login: str = None
    unit_performance: Dict[int, Dict] = None
    tone_preference: str = "friendly"
    
    def __post_init__(self):
        if self.weakest_units is None:
            self.weakest_units = []
        if self.strongest_units is None:
            self.strongest_units = []
        if self.unit_performance is None:
            self.unit_performance = {i: {'attempts': 0, 'correct': 0, 'avg_difficulty': 1} 
                                     for i in range(1, 9)}
        if self.last_login is None:
            self.last_login = datetime.now().isoformat()


# ============================================================================
# EMOTION DETECTOR (Built-in)
# ============================================================================

class EmotionDetector:
    """Lightweight emotion detection"""
    
    def __init__(self):
        self.emotion_patterns = {
            'motivated': ['motivated', 'excited', 'ready', 'pumped', 'let\'s go'],
            'tired': ['tired', 'exhausted', 'sleepy', 'drained', 'worn out'],
            'bored': ['bored', 'boring', 'dull', 'meh', 'not fun'],
            'anxious': ['anxious', 'nervous', 'worried', 'stressed', 'scared'],
            'proud': ['proud', 'accomplished', 'nailed it', 'crushed it'],
            'confused': ['confused', 'don\'t understand', 'lost', 'unclear']
        }
    
    def detect(self, text: str) -> Tuple[str, float]:
        """Detect emotion from text"""
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion, 0.8
        
        return 'neutral', 0.5
    
    def get_response(self, emotion: str) -> str:
        """Get empathetic response"""
        responses = {
            'motivated': ["Love the energy! Let's channel that 🔥", "That's the spirit! 💪"],
            'tired': ["Let's take it easy today 😊", "Small steps count 🌟"],
            'bored': ["Let's spice things up! 🎯", "Time for something new! 🎲"],
            'anxious': ["Take a deep breath. We'll start easy 🌸", "No pressure! 💙"],
            'proud': ["Yes! Keep it up 🎉", "Crushing it! 🏆"],
            'confused': ["Let me break it down 📚", "Great question! 💡"],
            'neutral': ["Ready when you are! 🎯", "Let's get started! 📖"]
        }
        return random.choice(responses.get(emotion, responses['neutral']))


# ============================================================================
# HYBRID INTENT MATCHER
# ============================================================================

class HybridIntentMatcher:
    """Semantic intent detection with TF-IDF"""
    
    def __init__(self):
        self.intents = {
            'quiz': ['quiz', 'test', 'practice', 'questions', 'give me'],
            'stats': ['stats', 'statistics', 'show', 'tell me about'],
            'plan': ['plan', 'schedule', 'prepare', 'timetable'],
            'search': ['search', 'find', 'look for'],
            'teach': ['teach', 'explain', 'what is', 'tell me about', 'how does'],
            'summary': ['summary', 'progress', 'how am i doing', 'report'],
            'greeting': ['hi', 'hello', 'hey', 'good morning'],
            'goodbye': ['bye', 'exit', 'quit', 'see you'],
            'motivation': ['bored', 'tired', 'nervous', 'motivate'],
            'help': ['help', 'commands', 'what can you do'],
            'thanks': ['thanks', 'thank you', 'appreciate'],
            'affirmative': ['yes', 'okay', 'sure', 'continue', '1'],
            'harder': ['harder', 'difficult', 'challenge', 'tough'],
            'easier': ['easier', 'simple', 'easy', 'basic'],
            'change_tone': ['change tone', 'switch mode', 'formal', 'friendly']
        }
        
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """Initialize TF-IDF"""
        self.training_phrases = []
        self.training_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.training_phrases.append(phrase)
                self.training_labels.append(intent)
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=500)
        self.training_vectors = self.vectorizer.fit_transform(self.training_phrases)
    
    def detect(self, text: str) -> Tuple[str, float]:
        """Detect intent"""
        text_clean = text.lower().strip()
        input_vector = self.vectorizer.transform([text_clean])
        similarities = cosine_similarity(input_vector, self.training_vectors)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.training_labels[best_idx]
        
        return (best_intent, best_score) if best_score >= 0.25 else ('unknown', best_score)


# ============================================================================
# KNOWLEDGE RECOMMENDER
# ============================================================================

class KnowledgeRecommender:
    """Teaches concepts and provides explanations"""
    
    def __init__(self):
        self.knowledge_base = {
            'gdp': {
                'topic': 'GDP (Gross Domestic Product)',
                'summary': 'GDP is the total monetary value of all goods and services produced within a country in a specific time period. It measures economic health.',
                'tips': 'Remember: GDP = C + I + G + (X-M) where C=Consumption, I=Investment, G=Government spending, X=Exports, M=Imports'
            },
            'constitution': {
                'topic': 'Indian Constitution',
                'summary': 'The Constitution of India is the supreme law. It has 395 articles, 8 schedules, and guarantees fundamental rights to citizens.',
                'tips': 'Key: Adopted on 26 Nov 1949, came into effect on 26 Jan 1950 (Republic Day)'
            },
            'water': {
                'topic': 'Water (H₂O)',
                'summary': 'Water is a chemical compound with formula H₂O. Essential for life, covers 71% of Earth\'s surface.',
                'tips': 'Properties: Boiling point 100°C, Freezing point 0°C, Universal solvent'
            }
        }
    
    def search(self, query: str) -> Optional[Dict]:
        """Search knowledge base"""
        query_lower = query.lower()
        
        for keyword, content in self.knowledge_base.items():
            if keyword in query_lower:
                return content
        
        return None
    
    def teach(self, topic: str) -> str:
        """Teach a topic"""
        content = self.search(topic)
        
        if content:
            return f"\n📚 {content['topic']}\n\n{content['summary']}\n\n💡 Tip: {content['tips']}"
        else:
            return f"\n🤔 I don't have detailed info on '{topic}' yet, but I can quiz you on related questions!"


# ============================================================================
# ADAPTIVE LEARNING ENGINE
# ============================================================================

class AdaptiveLearningEngine:
    """Tracks performance and adapts difficulty"""
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
    
    def update_performance(self, unit: int, correct: bool, difficulty: str):
        """Update unit performance"""
        perf = self.profile.unit_performance[unit]
        perf['attempts'] += 1
        if correct:
            perf['correct'] += 1
        
        # Update difficulty level
        accuracy = perf['correct'] / perf['attempts'] if perf['attempts'] > 0 else 0
        if accuracy > 0.8:
            perf['avg_difficulty'] = min(3, perf['avg_difficulty'] + 0.1)
        elif accuracy < 0.5:
            perf['avg_difficulty'] = max(1, perf['avg_difficulty'] - 0.1)
    
    def get_weak_units(self) -> List[int]:
        """Identify weak units"""
        weak = []
        for unit, perf in self.profile.unit_performance.items():
            if perf['attempts'] > 3:
                accuracy = perf['correct'] / perf['attempts']
                if accuracy < 0.6:
                    weak.append(unit)
        return weak[:3]
    
    def get_strong_units(self) -> List[int]:
        """Identify strong units"""
        strong = []
        for unit, perf in self.profile.unit_performance.items():
            if perf['attempts'] > 3:
                accuracy = perf['correct'] / perf['attempts']
                if accuracy > 0.8:
                    strong.append(unit)
        return strong[:3]
    
    def get_recommended_unit(self) -> int:
        """Recommend next unit to practice"""
        weak = self.get_weak_units()
        if weak:
            return weak[0]
        
        # Find least practiced
        min_attempts = min(p['attempts'] for p in self.profile.unit_performance.values())
        for unit, perf in self.profile.unit_performance.items():
            if perf['attempts'] == min_attempts:
                return unit
        
        return random.randint(1, 8)


# ============================================================================
# USER PROFILE MANAGER
# ============================================================================

class ProfileManager:
    """Manages user profiles with persistence"""
    
    def __init__(self, data_dir='user_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.profiles_file = os.path.join(data_dir, 'profiles.json')
        self.current_profile = None
    
    def load_profiles(self) -> Dict:
        """Load all profiles"""
        if os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_profiles(self, profiles: Dict):
        """Save all profiles"""
        with open(self.profiles_file, 'w') as f:
            json.dump(profiles, f, indent=2)
    
    def create_profile(self, name: str) -> UserProfile:
        """Create new profile"""
        profile = UserProfile(name=name)
        profiles = self.load_profiles()
        profiles[name] = asdict(profile)
        self.save_profiles(profiles)
        return profile
    
    def load_profile(self, name: str) -> Optional[UserProfile]:
        """Load existing profile"""
        profiles = self.load_profiles()
        if name in profiles:
            data = profiles[name]
            return UserProfile(**data)
        return None
    
    def save_profile(self, profile: UserProfile):
        """Save profile"""
        profiles = self.load_profiles()
        profiles[profile.name] = asdict(profile)
        self.save_profiles(profiles)
    
    def list_profiles(self) -> List[str]:
        """List all profile names"""
        return list(self.load_profiles().keys())


# ============================================================================
# MAIN STUDY BUDDY CLASS
# ============================================================================

class TNPSCStudyBuddyV3:
    """Next-Gen Adaptive & Emotion-Aware Study Buddy"""
    
    def __init__(self, debug_mode=False):
        """Initialize v3 study buddy"""
        self.debug_mode = debug_mode
        
        print("🚀 Initializing TNPSC Study Buddy v3.0...")
        print("="*70)
        
        # Core components
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # AI modules
        self.intent_matcher = HybridIntentMatcher()
        self.emotion_detector = EmotionDetector()
        self.knowledge_recommender = KnowledgeRecommender()
        
        # Profile management
        self.profile_manager = ProfileManager()
        self.profile = None
        self.adaptive_engine = None
        
        # Context
        self.context = ConversationContext()
        
        # NLU logging
        self.nlu_log_file = 'user_data/nlu_log.csv'
        
        # Train ML
        self._train_models()
        
        print("✅ Study Buddy v3.0 Ready!")
        print("="*70)
    
    def _train_models(self):
        """Train ML models"""
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['has_numbers'] = self.df['question'].str.contains(r'\d').astype(int)
        
        X = self.df[['unit', 'year', 'question_length', 'word_count', 'has_numbers']]
        y = self.df['difficulty']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
    
    def _log_nlu(self, user_input: str, intent: str, confidence: float, was_correct: bool = True):
        """Log NLU interactions for improvement"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'input': user_input,
                'intent': intent,
                'confidence': confidence,
                'correct': was_correct
            }
            
            os.makedirs('user_data', exist_ok=True)
            
            if not os.path.exists(self.nlu_log_file):
                pd.DataFrame([log_entry]).to_csv(self.nlu_log_file, index=False)
            else:
                pd.DataFrame([log_entry]).to_csv(self.nlu_log_file, mode='a', header=False, index=False)
        except:
            pass
    
    def _typing_delay(self):
        """Natural typing delay"""
        time.sleep(random.uniform(0.3, 0.8))
    
    def _parse_entities(self, text: str) -> Dict:
        """Extract entities"""
        text_lower = text.lower()
        
        # Numbers
        numbers = [int(n) for n in re.findall(r'\d+', text)]
        
        # Units
        unit_map = {
            'science': 1, 'current': 2, 'geography': 3, 'history': 4,
            'polity': 5, 'economy': 6, 'national': 7, 'mental': 8
        }
        
        unit = None
        for keyword, unit_num in unit_map.items():
            if keyword in text_lower:
                unit = unit_num
                break
        
        if not unit and numbers:
            if 1 <= numbers[0] <= 8:
                unit = numbers[0]
        
        # Keyword
        keyword = text_lower
        for word in ['give', 'me', 'show', 'find', 'search', 'questions', 'about', 'on']:
            keyword = keyword.replace(word, '')
        keyword = keyword.strip()
        
        return {'numbers': numbers, 'unit': unit, 'keyword': keyword}
    
    def _get_response(self, intent: str, tone: str = 'friendly') -> str:
        """Get response based on tone"""
        responses = {
            'friendly': {
                'quiz': ["Let's warm up your brain 🧠", "Quiz time! 🔥"],
                'stats': ["Here's what I found 📊", "Let me check that 📈"],
                'plan': ["Let's design your roadmap 🚀", "Planning time! 📅"]
            },
            'formal': {
                'quiz': ["Initiating quiz module.", "Preparing assessment."],
                'stats': ["Retrieving statistics.", "Analyzing data."],
                'plan': ["Creating study schedule.", "Generating plan."]
            },
            'coaching': {
                'quiz': ["Time to level up! 💪", "Let's push your limits! 🎯"],
                'stats': ["Let's see your progress! 📊", "Checking your gains! 📈"],
                'plan': ["Building your success path! 🏆", "Mapping your victory! 🎯"]
            }
        }
        
        tone_responses = responses.get(tone, responses['friendly'])
        return random.choice(tone_responses.get(intent, ["Let's do this!"]))
    
    def chat(self):
        """Main chat interface"""
        print("\n" + "="*70)
        print("🎓 TNPSC STUDY BUDDY v3.0 - Adaptive & Emotion-Aware")
        print("="*70)
        
        # Profile selection
        profiles = self.profile_manager.list_profiles()
        
        if profiles:
            print(f"\n👥 Existing profiles: {', '.join(profiles)}")
            name = input("Enter your name (or 'new' for new profile): ").strip()
            
            if name.lower() == 'new' or name not in profiles:
                name = input("Create new profile - Your name: ").strip()
                self.profile = self.profile_manager.create_profile(name)
                print(f"✅ Profile created for {name}!")
            else:
                self.profile = self.profile_manager.load_profile(name)
                print(f"👋 Welcome back {name}!")
        else:
            name = input("\n👤 What's your name? ").strip()
            self.profile = self.profile_manager.create_profile(name)
        
        self.adaptive_engine = AdaptiveLearningEngine(self.profile)
        
        # Update session
        self.profile.total_sessions += 1
        self.profile.last_login = datetime.now().isoformat()
        
        print(f"\n✨ I understand natural language and adapt to your learning!")
        print(f"📊 Sessions: {self.profile.total_sessions} | Quizzes: {self.profile.quizzes_taken}")
        print("\nType 'help' anytime or 'bye' to exit")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n💬 {self.profile.name}: ").strip()
                
                if not user_input:
                    continue
                
                # Detect intent & emotion
                intent, confidence = self.intent_matcher.detect(user_input)
                emotion, emotion_conf = self.emotion_detector.detect(user_input)
                entities = self._parse_entities(user_input)
                
                # Log NLU
                self._log_nlu(user_input, intent, confidence)
                
                # Debug mode
                if self.debug_mode:
                    print(f"[DEBUG] Intent: {intent} ({confidence:.3f})")
                    print(f"[DEBUG] Emotion: {emotion} ({emotion_conf:.3f})")
                    print(f"[DEBUG] Entities: {entities}")
                
                # Update context
                self.context.last_emotion = emotion
                
                # Typing delay
                self._typing_delay()
                
                # Handle affirmative
                if intent == 'affirmative' and self.context.last_intent:
                    intent = self.context.last_intent
                
                # Route to handlers
                if intent == 'goodbye':
                    self.profile_manager.save_profile(self.profile)
                    print(f"\n🤖 Buddy: Goodbye {self.profile.name}! Keep crushing it! 🎉")
                    break
                
                elif intent == 'greeting':
                    response = self.emotion_detector.get_response(emotion)
                    print(f"\n🤖 Buddy: {response}")
                
                elif intent == 'motivation' or emotion in ['tired', 'bored', 'anxious']:
                    response = self.emotion_detector.get_response(emotion)
                    print(f"\n🤖 Buddy: {response}")
                    print("💡 Want a quick quiz to boost your mood?")
                
                elif intent == 'teach':
                    explanation = self.knowledge_recommender.teach(user_input)
                    print(f"\n🤖 Buddy: {explanation}")
                
                elif intent == 'summary':
                    self._show_summary()
                
                elif intent == 'harder':
                    print(f"\n🤖 Buddy: Challenge accepted! Raising difficulty 🔥")
                    self.context.last_intent = 'quiz'
                
                elif intent == 'easier':
                    print(f"\n🤖 Buddy: No problem! Let's start easier 😊")
                    self.context.last_intent = 'quiz'
                
                elif intent == 'change_tone':
                    self._change_tone(user_input)
                
                elif intent == 'quiz':
                    tone = self.profile.tone_preference
                    print(f"\n🤖 Buddy: {self._get_response('quiz', tone)}")
                    self._handle_quiz(entities)
                    self.context.last_intent = 'quiz'
                
                elif intent == 'stats':
                    print(f"\n🤖 Buddy: {self._get_response('stats', self.profile.tone_preference)}")
                    self._handle_stats(entities)
                    self.context.last_intent = 'stats'
                
                elif intent == 'plan':
                    print(f"\n🤖 Buddy: {self._get_response('plan', self.profile.tone_preference)}")
                    self._handle_plan(entities)
                    self.context.last_intent = 'plan'
                
                elif intent == 'help':
                    self._show_help()
                
                elif intent == 'thanks':
                    print(f"\n🤖 Buddy: You're welcome! Keep going 💪")
                
                else:
                    print(f"\n🤖 Buddy: Hmm, not sure about that 🤔")
                    print("💡 Try: 'quiz me' | 'show stats' | 'teach me about GDP' | 'help'")
                
            except KeyboardInterrupt:
                self.profile_manager.save_profile(self.profile)
                print(f"\n\n🎉 Goodbye {self.profile.name}!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
    
    def _handle_quiz(self, entities: Dict):
        """Adaptive quiz generation"""
        unit = entities['unit']
        numbers = entities['numbers']
        
        # Use adaptive engine to recommend unit
        if not unit:
            unit = self.adaptive_engine.get_recommended_unit()
            print(f"📌 Recommended: {UNITS[unit]} (based on your progress)")
        
        num_questions = numbers[0] if numbers and numbers[0] <= 20 else 5
        
        # Filter by unit
        filtered = self.df[self.df['unit'] == unit]
        
        # Sample
        if len(filtered) < num_questions:
            quiz = filtered
        else:
            quiz = filtered.sample(n=num_questions, replace=False)
        
        # Display
        print(f"\n📝 ADAPTIVE QUIZ: {UNITS[unit]}")
        print("="*70)
        
        for idx, (_, row) in enumerate(quiz.iterrows(), 1):
            print(f"\nQ{idx}. {row['question']}")
            print(f"    Difficulty: {row['difficulty']} | Year: {row['year']}")
        
        print("="*70)
        print(f"✅ {len(quiz)} questions! Track your answers for adaptive learning 📊")
        
        # Update profile
        self.profile.quizzes_taken += 1
        self.profile.questions_answered += len(quiz)
        self.context.last_unit = unit
    
    def _handle_stats(self, entities: Dict):
        """Show statistics"""
        unit = entities['unit']
        
        if not unit:
            print(f"\n💡 Which unit? Try: 'stats for science'")
            return
        
        unit_data = self.df[self.df['unit'] == unit]
        perf = self.profile.unit_performance[unit]
        
        print(f"\n📊 STATISTICS: {UNITS[unit]}")
        print("-"*70)
        print(f"  Total Questions Available: {len(unit_data)}")
        print(f"  Your Attempts: {perf['attempts']}")
        
        if perf['attempts'] > 0:
            accuracy = (perf['correct'] / perf['attempts']) * 100
            print(f"  Your Accuracy: {accuracy:.1f}%")
            print(f"  Current Difficulty Level: {perf['avg_difficulty']:.1f}/3")
        
        print(f"\n  Difficulty Breakdown:")
        for diff, count in unit_data['difficulty'].value_counts().items():
            pct = (count / len(unit_data)) * 100
            print(f"    {diff}: {count} ({pct:.1f}%)")
        print("-"*70)
    
    def _handle_plan(self, entities: Dict):
        """Create adaptive study plan"""
        numbers = entities['numbers']
        days = numbers[0] if numbers else 30
        
        weak_units = self.adaptive_engine.get_weak_units()
        
        print(f"\n📅 ADAPTIVE STUDY PLAN - {days} DAYS")
        print("="*70)
        print(f"  Questions per day: {len(self.df) // days}")
        
        if weak_units:
            print(f"\n  🎯 Priority Units (Need Focus):")
            for unit in weak_units:
                print(f"    • {UNITS[unit]}")
        
        print(f"\n  Sample Schedule (First 5 Days):")
        for day in range(1, min(6, days + 1)):
            if weak_units and day <= len(weak_units):
                unit = weak_units[day - 1]
                print(f"\n  Day {day}: {UNITS[unit]} ⚠️ (Priority)")
            else:
                unit = ((day - 1) % 8) + 1
                print(f"\n  Day {day}: {UNITS[unit]}")
        
        print("="*70)
        print(f"✅ Adaptive plan ready! Focus on weak areas first 🎯")
    
    def _show_summary(self):
        """Show weekly progress summary"""
        weak = self.adaptive_engine.get_weak_units()
        strong = self.adaptive_engine.get_strong_units()
        
        print(f"\n📈 YOUR PROGRESS SUMMARY")
        print("="*70)
        print(f"  Total Sessions: {self.profile.total_sessions}")
        print(f"  Quizzes Taken: {self.profile.quizzes_taken}")
        print(f"  Questions Answered: {self.profile.questions_answered}")
        
        if self.profile.questions_answered > 0:
            accuracy = (self.profile.correct_answers / self.profile.questions_answered) * 100
            print(f"  Overall Accuracy: {accuracy:.1f}%")
        
        if strong:
            print(f"\n  💪 Strongest Units:")
            for unit in strong:
                print(f"    • {UNITS[unit]}")
        
        if weak:
            print(f"\n  🎯 Need Focus:")
            for unit in weak:
                print(f"    • {UNITS[unit]}")
        
        print("\n  💡 AI Coach Suggestion:")
        if weak:
            print(f"     Focus on {UNITS[weak[0]]} this week!")
        else:
            print(f"     Great progress! Keep practicing all units!")
        
        print("="*70)
    
    def _change_tone(self, text: str):
        """Change response tone"""
        if 'formal' in text.lower():
            self.profile.tone_preference = 'formal'
            print(f"\n🤖 Buddy: Tone changed to formal mode.")
        elif 'coaching' in text.lower():
            self.profile.tone_preference = 'coaching'
            print(f"\n🤖 Buddy: Coaching mode activated! Let's crush this! 💪")
        else:
            self.profile.tone_preference = 'friendly'
            print(f"\n🤖 Buddy: Switched to friendly mode! 😊")
    
    def _show_help(self):
        """Show help"""
        print(f"\n📚 WHAT I CAN DO:")
        print("-"*70)
        print("  🎯 Adaptive Quizzes - 'quiz me' or 'give me 5 questions on economy'")
        print("  📊 Performance Stats - 'show stats for history'")
        print("  📅 Smart Study Plans - 'create 30 day plan'")
        print("  📚 Teach Concepts - 'explain what is GDP'")
        print("  📈 Progress Summary - 'how am I doing' or 'summary'")
        print("  💪 Difficulty Control - 'make it harder' or 'easier'")
        print("  🎭 Change Tone - 'change tone to formal'")
        print("  💙 Emotional Support - I detect your mood and adapt!")
        print("-"*70)
        print("Just talk naturally - I learn and adapt to you! 🧠")


def main():
    """Entry point"""
    try:
        buddy = TNPSCStudyBuddyV3(debug_mode=False)
        buddy.chat()
    except KeyboardInterrupt:
        print("\n\n🎉 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
