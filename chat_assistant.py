#!/usr/bin/env python3
"""
TNPSC AI Chat Assistant - Main Application
Interactive chat-based exam preparation assistant
"""

import os
import sys
import pandas as pd
import numpy as np
import datetime
import time
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our core assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS

class TNPSCChatAssistant:
    """Interactive Chat Assistant for TNPSC Exam Preparation"""
    
    def __init__(self):
        """Initialize the chat assistant with ML models"""
        self._show_loading_animation("🚀 Initializing TNPSC AI Chat Assistant")
        
        # Load core assistant
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # Initialize user session data for personalization
        self.user_session = {
            'quizzes_taken': 0,
            'total_questions_answered': 0,
            'correct_answers': 0,
            'favorite_units': [],
            'last_activity': None,
            'encouragement_level': 'normal',  # normal, high, motivational
            'conversation_context': [],  # Remember recent conversation
            'user_name': None,
            'mood': 'neutral',  # happy, sad, frustrated, excited, neutral
            'learning_style': 'unknown',  # visual, practice-heavy, theory-first
            'current_topic': None,
            'just_finished_quiz': False,
            'needs_encouragement': False
        }
        
        # Train ML models
        self._train_models()
        
        self._clear_screen()
        self._show_welcome_banner()
        print("✅ Chat Assistant Ready!")
        print("🤗 I'm excited to be your study companion!")
        print("="*60)
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_loading_animation(self, message):
        """Show a loading animation"""
        print(f"\n{message}")
        animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for i in range(20):  # Show animation for 2 seconds
            print(f"\r{animation[i % len(animation)]} Loading...", end="", flush=True)
            time.sleep(0.1)
        print("\r✅ Ready!                    ")
    
    def _show_welcome_banner(self):
        """Show an animated welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🎓 TNPSC STUDY BUDDY 🎓                   ║
║                                                              ║
║        Your AI-Powered Personal Exam Preparation Assistant   ║
║                                                              ║
║  🚀 Interactive Quizzes  📊 Progress Tracking  💪 Motivation ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        time.sleep(0.5)
    
    def _typing_effect(self, text, delay=0.03):
        """Simulate typing effect for more natural conversation"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def _show_progress_bar(self, current, total, width=30):
        """Show a progress bar"""
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total) * 100
        return f"[{bar}] {percentage:.1f}% ({current}/{total})"
    
    def _get_input_with_suggestions(self, prompt, suggestions=None):
        """Get user input with live suggestions"""
        print(f"\n{prompt}")
        if suggestions:
            print("💡 Suggestions:", end="")
            for i, suggestion in enumerate(suggestions[:3]):
                print(f" '{suggestion}'", end="")
                if i < len(suggestions[:3]) - 1:
                    print(",", end="")
            print()
        
        return input("💬 You: ").strip()
    
    def _show_thinking_animation(self, duration=1):
        """Show thinking animation"""
        thinking = ["🤔", "💭", "🧠", "💡"]
        for i in range(int(duration * 4)):
            print(f"\r{thinking[i % len(thinking)]} Thinking...", end="", flush=True)
            time.sleep(0.25)
        print("\r" + " " * 20 + "\r", end="")  # Clear the line
    
    def _show_interactive_intro(self):
        """Show an interactive introduction"""
        print("\n" + "="*60)
        self._typing_effect("🎓 Welcome to your TNPSC Study Journey! 🎓", 0.05)
        print("="*60)
        
        time.sleep(0.5)
        self._typing_effect("\nHey there, future TNPSC champion! 👋", 0.03)
        self._typing_effect("I'm your personal AI study buddy, and I'm SUPER excited to help you succeed! 🚀", 0.03)
        
        print("\n🎯 Here's what makes me special:")
        features = [
            "📝 Interactive quizzes that adapt to your pace",
            "📅 Smart study plans based on your goals", 
            "📊 Real-time progress tracking with celebrations",
            "🔍 Instant topic search and explanations",
            "💪 Motivational support when you need it most",
            "🤗 Natural conversation - just like a real friend!"
        ]
        
        for feature in features:
            time.sleep(0.3)
            print(f"  • {feature}")
        
        print("\n💬 The best part? Just talk to me naturally!")
        example_phrases = [
            "'quiz me on geography'",
            "'I'm struggling with history'", 
            "'create a study plan'",
            "'I'm feeling nervous'",
            "'surprise me!'",
            "'just chat with me'"
        ]
        
        print("  Try saying things like:", end="")
        for i, phrase in enumerate(example_phrases):
            time.sleep(0.4)
            if i % 2 == 0:
                print(f"\n    🗣️ {phrase}", end="")
            else:
                print(f" or {phrase}", end="")
        
        print("\n\n🌟 I'm not just a bot - I'm your dedicated study companion!")
        print("="*60)
    
    def _get_user_name(self):
        """Get user's name for personalization"""
        print("\n🤗 Before we start, I'd love to know what to call you!")
        name_input = input("💬 What's your name? (or just press Enter to skip): ").strip()
        
        if name_input:
            self.user_session['user_name'] = name_input
            responses = [
                f"Nice to meet you, {name_input}! 😊 That's a great name!",
                f"Hey {name_input}! 🌟 I'm so excited to help you succeed!",
                f"Awesome, {name_input}! 🎯 Let's make your TNPSC prep amazing!",
                f"Perfect, {name_input}! 💪 Ready to crush this exam together?"
            ]
            self._typing_effect(f"\n🤖 Assistant: {np.random.choice(responses)}")
        else:
            self.user_session['user_name'] = "Champion"
            self._typing_effect("\n🤖 Assistant: No worries! I'll just call you Champion! 🏆")
        
        time.sleep(1)
    
    def _get_contextual_suggestions(self):
        """Get contextual suggestions based on user's history and current state"""
        suggestions = []
        
        # Based on quiz history
        if self.user_session['quizzes_taken'] == 0:
            suggestions.extend(["quiz me", "start with basics", "show me units"])
        elif self.user_session['just_finished_quiz']:
            suggestions.extend(["another quiz", "check my progress", "different topic"])
        else:
            suggestions.extend(["practice questions", "my stats", "study plan"])
        
        # Based on mood/needs
        if self.user_session['needs_encouragement']:
            suggestions.extend(["motivate me", "I need help", "feeling stuck"])
        
        # Based on time of day
        hour = datetime.datetime.now().hour
        if 6 <= hour < 12:
            suggestions.extend(["morning quiz", "plan my day"])
        elif 18 <= hour < 22:
            suggestions.extend(["evening review", "quick practice"])
        
        # Always include some fun options
        suggestions.extend(["surprise me", "something fun", "just chat"])
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _train_models(self):
        """Train ML models for predictions with better accuracy"""
        print("🔧 Training ML models...")
        
        # Prepare enhanced features
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['has_numbers'] = self.df['question'].str.contains(r'\d').astype(int)
        self.df['has_question_mark'] = self.df['question'].str.contains(r'\?').astype(int)
        self.df['avg_word_length'] = self.df['question'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Train difficulty predictor with more features
        X = self.df[['unit', 'year', 'question_length', 'word_count', 
                     'has_numbers', 'has_question_mark', 'avg_word_length']]
        y = self.df['difficulty']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use better hyperparameters
        self.difficulty_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.difficulty_model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.difficulty_model.predict(X_test))
        print(f"   ✅ Difficulty Predictor trained (Accuracy: {accuracy:.2%})")
    
    def predict_difficulty(self, unit, year, question_text):
        """Predict difficulty of a question with enhanced features"""
        question_length = len(question_text)
        word_count = len(question_text.split())
        has_numbers = 1 if any(char.isdigit() for char in question_text) else 0
        has_question_mark = 1 if '?' in question_text else 0
        avg_word_length = np.mean([len(word) for word in question_text.split()]) if word_count > 0 else 0
        
        features = [[unit, year, question_length, word_count, has_numbers, has_question_mark, avg_word_length]]
        prediction = self.difficulty_model.predict(features)[0]
        
        return prediction
    
    def get_unit_stats(self, unit_num):
        """Get statistics for a specific unit"""
        unit_data = self.df[self.df['unit'] == unit_num]
        
        if len(unit_data) == 0:
            return None
        
        stats = {
            'unit_name': UNITS[unit_num],
            'total_questions': len(unit_data),
            'difficulty_distribution': unit_data['difficulty'].value_counts().to_dict(),
            'year_range': f"{unit_data['year'].min()} - {unit_data['year'].max()}",
            'avg_question_length': unit_data['question'].str.len().mean()
        }
        
        return stats
    
    def generate_quiz(self, unit_num, difficulty=None, num_questions=5):
        """Generate a quiz for practice"""
        unit_data = self.df[self.df['unit'] == unit_num]
        
        if difficulty:
            unit_data = unit_data[unit_data['difficulty'] == difficulty]
        
        if len(unit_data) < num_questions:
            quiz = unit_data
        else:
            quiz = unit_data.sample(n=num_questions)
        
        return quiz
    
    def get_study_plan(self, days_remaining, weak_units=None):
        """Generate personalized study plan"""
        plan = self.assistant.generate_study_plan(days_remaining)
        
        if weak_units:
            # Prioritize weak units
            plan['priority_units'] = weak_units
            plan['recommendation'] = f"Focus extra time on units: {', '.join([UNITS[u] for u in weak_units])}"
        
        return plan
    
    def search_questions(self, keyword, unit=None):
        """Search questions by keyword"""
        results = self.df[self.df['question'].str.contains(keyword, case=False, na=False)]
        
        if unit:
            results = results[results['unit'] == unit]
        
        return results
    
    def get_performance_insights(self):
        """Get overall performance insights"""
        insights = {
            'total_questions': len(self.df),
            'units_covered': len(self.df['unit'].unique()),
            'difficulty_breakdown': self.df['difficulty'].value_counts().to_dict(),
            'questions_per_unit': self.df['unit'].value_counts().sort_index().to_dict(),
            'year_coverage': sorted(self.df['year'].unique())
        }
        
        return insights
    
    def _parse_natural_input(self, user_input):
        """Parse natural language input to extract intent and parameters - Super casual!"""
        import re
        
        text = user_input.lower().strip()
        
        # Extract numbers from text (including written numbers)
        numbers = re.findall(r'\d+', text)
        
        # Convert written numbers to digits
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }
        for word, num in number_words.items():
            if word in text:
                numbers.append(str(num))
        
        # Super comprehensive unit mapping with casual variations
        unit_mapping = {
            # Unit 1 - Science
            'science': 1, 'general science': 1, 'sci': 1, 'physics': 1, 
            'chemistry': 1, 'biology': 1, 'bio': 1, 'chem': 1,
            
            # Unit 2 - Current Events
            'current': 2, 'current events': 2, 'events': 2, 'news': 2,
            'current affairs': 2, 'affairs': 2, 'recent': 2,
            
            # Unit 3 - Geography
            'geography': 3, 'geo': 3, 'places': 3, 'location': 3,
            'earth': 3, 'map': 3, 'world': 3,
            
            # Unit 4 - History & Culture
            'history': 4, 'culture': 4, 'historical': 4, 'ancient': 4,
            'medieval': 4, 'modern': 4, 'past': 4,
            
            # Unit 5 - Polity
            'polity': 5, 'indian polity': 5, 'politics': 5, 'government': 5,
            'constitution': 5, 'governance': 5, 'parliament': 5,
            
            # Unit 6 - Economy
            'economy': 6, 'indian economy': 6, 'economic': 6, 'economics': 6,
            'gdp': 6, 'budget': 6, 'finance': 6, 'money': 6,
            
            # Unit 7 - National Movement
            'national': 7, 'movement': 7, 'national movement': 7, 'freedom': 7,
            'independence': 7, 'struggle': 7,
            
            # Unit 8 - Mental Ability
            'mental': 8, 'ability': 8, 'mental ability': 8, 'aptitude': 8,
            'reasoning': 8, 'logic': 8, 'maths': 8, 'math': 8
        }
        
        detected_unit = None
        for keyword, unit_num in unit_mapping.items():
            if keyword in text:
                detected_unit = unit_num
                break
        
        # If no keyword match, check for "unit X" pattern
        if not detected_unit and numbers:
            first_num = int(numbers[0])
            if 1 <= first_num <= 8:
                detected_unit = first_num
        
        return {
            'text': text,
            'numbers': numbers,
            'unit': detected_unit
        }
    
    def chat(self):
        """Main chat interface with natural language understanding"""
        self._show_interactive_intro()
        
        # Get user's name for personalization
        if not self.user_session['user_name']:
            self._get_user_name()
        
        # Main conversation loop with enhanced interactivity
        conversation_starters = [
            f"So {self.user_session['user_name']}, what's on your mind today?",
            f"How can I help you succeed today, {self.user_session['user_name']}?",
            f"Ready to tackle some learning, {self.user_session['user_name']}?",
            f"What would you like to explore, {self.user_session['user_name']}?"
        ]
        
        print(f"\n🤖 Assistant: {np.random.choice(conversation_starters)}")
        
        while True:
            try:
                # Dynamic suggestions based on context
                suggestions = self._get_contextual_suggestions()
                user_input = self._get_input_with_suggestions("", suggestions).strip()
                
                if not user_input:
                    gentle_prompts = [
                        "🤔 Take your time! What's on your mind?",
                        "😊 No rush! Just say whatever you're thinking!",
                        "💭 Anything at all - I'm here to help!",
                        "🌟 Even a simple 'hi' works! What would you like to do?"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(gentle_prompts)}")
                    continue
                
                # Show thinking animation for more natural feel
                self._show_thinking_animation(0.5)
                
                # Parse natural language input
                parsed = self._parse_natural_input(user_input)
                cmd = parsed['text']
                
                # Add to conversation context
                self.user_session['conversation_context'].append(user_input)
                if len(self.user_session['conversation_context']) > 5:
                    self.user_session['conversation_context'].pop(0)  # Keep last 5 exchanges
                
                # Exit commands - more natural
                if any(word in cmd for word in ['quit', 'exit', 'bye', 'goodbye', 'stop', 'see you', 'gotta go']):
                    farewell_messages = [
                        "Aww, leaving already? 😊 It was awesome studying with you! Come back soon!",
                        "Take care! 🤗 Remember, I'm always here when you need a study buddy!",
                        "Bye for now! 👋 You're going to absolutely crush that TNPSC exam!",
                        "See you later! 🌟 Keep up that amazing dedication - you've got this!",
                        "Until next time! 💪 I believe in you 100%!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(farewell_messages)}")
                    
                    # Add personalized goodbye based on session
                    if self.user_session['quizzes_taken'] > 0:
                        print(f"🎯 You practiced {self.user_session['total_questions_answered']} questions today - that's dedication!")
                    
                    print("🎉 Good luck with your TNPSC preparation!")
                    break
                
                # Help command - contextual based on what they just did
                elif any(word in cmd for word in ['help', 'commands', 'what can you do', 'confused', 'lost']):
                    if self.user_session['just_finished_quiz']:
                        print("\n🤖 Assistant: Just finished a quiz? Nice! 🎯 Want to try another one, or maybe check your progress?")
                        print("   • 'another quiz' - Keep practicing!")
                        print("   • 'my progress' - See how you're doing")
                        print("   • 'different topic' - Switch subjects")
                        self.user_session['just_finished_quiz'] = False
                    elif self.user_session['last_activity'] == 'stats':
                        print("\n🤖 Assistant: Saw your stats? Cool! 📊 Ready to practice those topics or need a study plan?")
                        print("   • 'quiz me' - Jump into practice")
                        print("   • 'study plan' - Get organized")
                        print("   • 'focus on weak areas' - Target improvement")
                    else:
                        self._show_help()
                
                # Unit stats
                elif any(word in cmd for word in ['stats', 'statistics', 'show me', 'tell me about']):
                    if parsed['unit']:
                        self._handle_stats_smart(parsed['unit'])
                    elif parsed['numbers']:
                        self._handle_stats_smart(int(parsed['numbers'][0]))
                    else:
                        print("\n💡 Which unit would you like stats for? (1-8)")
                        print("   Try: 'stats for science' or 'stats 1'")
                
                # Generate quiz - super casual variations
                elif any(word in cmd for word in ['quiz', 'test', 'practice', 'questions', 'generate', 
                                                   'give me', 'show me', 'need', 'want', 'get me',
                                                   'gimme', 'can i get', 'can you give']):
                    if parsed['unit']:
                        self._handle_quiz_smart(parsed['unit'], parsed['numbers'])
                    elif parsed['numbers']:
                        self._handle_quiz_smart(int(parsed['numbers'][0]), parsed['numbers'][1:])
                    else:
                        print("\n💡 Which unit would you like a quiz for? (1-8)")
                        print("   Try: 'quiz for science' or 'give me some questions on history'")
                
                # Study plan
                elif any(word in cmd for word in ['plan', 'schedule', 'study', 'prepare']):
                    if parsed['numbers']:
                        self._handle_plan_smart(int(parsed['numbers'][0]))
                    else:
                        print("\n💡 How many days do you have for preparation?")
                        print("   Try: 'create 30 day plan' or 'plan for 15 days'")
                
                # Search questions
                elif any(word in cmd for word in ['search', 'find', 'look for']):
                    # Extract search keyword (remove command words)
                    keywords = cmd
                    for word in ['search', 'find', 'look for', 'questions about', 'about']:
                        keywords = keywords.replace(word, '')
                    keywords = keywords.strip()
                    
                    if keywords:
                        self._handle_search_smart(keywords)
                    else:
                        print("\n💡 What topic would you like to search for?")
                        print("   Try: 'search water' or 'find questions about economy'")
                
                # Predict difficulty
                elif any(word in cmd for word in ['predict', 'difficulty', 'how hard']):
                    self._handle_predict(user_input)
                
                # Show insights
                elif any(word in cmd for word in ['insights', 'overview', 'summary', 'show all']):
                    self._handle_insights()
                
                # List units
                elif any(word in cmd for word in ['units', 'list', 'show units', 'all units', 'subjects']):
                    self._show_units()
                
                # Greetings - natural and contextual
                elif any(word in cmd for word in ['hi', 'hello', 'hey', 'greetings', 'sup', 'yo', 'wassup']):
                    # Check if returning user
                    if self.user_session['quizzes_taken'] > 0:
                        returning_greetings = [
                            f"Hey, welcome back! 😊 Last time you were working on some questions - ready to continue the momentum?",
                            f"Oh hey! 👋 Good to see you again! You've answered {self.user_session['total_questions_answered']} questions so far - impressive!",
                            f"Hi there! 🌟 Back for more practice? I love your consistency!",
                            f"Hey! 🎯 Ready to pick up where we left off? You're building such a great study habit!"
                        ]
                        print(f"\n🤖 Assistant: {np.random.choice(returning_greetings)}")
                    else:
                        first_time_greetings = [
                            "Hey there! 👋 I'm so excited to meet you! I'm like your personal study buddy for TNPSC prep!",
                            "Hi! 😊 Welcome! I'm here to make studying actually fun and way more effective!",
                            "Hello! 🎯 Ready to turn your TNPSC prep into an adventure? Let's do this together!",
                            "Hey! 🚀 I'm your new study companion! Think of me as that friend who actually enjoys helping with homework!"
                        ]
                        print(f"\n🤖 Assistant: {np.random.choice(first_time_greetings)}")
                    
                    # Natural follow-up questions
                    follow_ups = [
                        "So, what's on your mind today? Feeling ready to tackle some questions?",
                        "How are you feeling about your prep? Confident, nervous, or somewhere in between?",
                        "What would help you most right now - practice, planning, or just a confidence boost?",
                        "Tell me, what's your biggest challenge with TNPSC prep lately?"
                    ]
                    print(f"🤔 {np.random.choice(follow_ups)}")
                    
                    # Time-based suggestions
                    hour = datetime.datetime.now().hour
                    if 5 <= hour < 12:
                        print("🌅 Morning energy is perfect for learning new concepts!")
                    elif 12 <= hour < 17:
                        print("☀️ Afternoon focus is great for practice questions!")
                    elif 17 <= hour < 21:
                        print("🌆 Evening review sessions really help retention!")
                    else:
                        print("🌙 Late night study? Let's make it count!")
                
                # Thanks - warm and encouraging responses
                elif any(word in cmd for word in ['thanks', 'thank you', 'appreciate', 'thx', 'ty']):
                    gratitude_responses = [
                        "Aww, you're so welcome! 🤗 It makes me happy to help you succeed! Keep crushing it! 💪",
                        "No problem at all! 😊 Seeing you learn and grow is the best reward! You got this! 🎯",
                        "Anytime, my friend! ⭐ Your dedication inspires me! Keep up the fantastic work!",
                        "Happy to help! 🚀 You're going to absolutely ace that exam - I can feel it!",
                        "You're very welcome! 💫 Remember, we're a team in this journey to success!",
                        "My pleasure! 🌟 Your gratitude means the world! Now go show that exam who's boss!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(gratitude_responses)}")
                    
                    # Add a little extra encouragement
                    bonus_encouragement = [
                        "💡 Pro tip: Gratitude is a sign of a growth mindset - you're already thinking like a winner!",
                        "🎯 Fun fact: People who say thank you tend to be more successful. You're on the right track!",
                        "🌈 Your positive attitude is contagious! Keep spreading those good vibes!",
                        "⚡ That thankful spirit will carry you far in life, not just in exams!"
                    ]
                    print(f"{np.random.choice(bonus_encouragement)}")
                
                # Add surprise/random features
                elif any(word in cmd for word in ['surprise', 'random', 'fun', 'something different', 'bored']):
                    surprises = [
                        "🎲 SURPRISE CHALLENGE! Let me pick a random topic for you!",
                        "🎪 FUN TIME! How about a lightning round of mixed questions?",
                        "🎭 PLOT TWIST! Let's do something unexpected!",
                        "🎨 CREATIVE MODE! Time for a different kind of learning adventure!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(surprises)}")
                    
                    # Generate a random surprise activity
                    surprise_activities = [
                        ("🔥 Rapid Fire Round", "Let's do 3 quick questions from random units!"),
                        ("🧠 Brain Teaser", "Here's a tricky question to challenge you!"),
                        ("📚 Random Fact", "Did you know? Let me share something cool!"),
                        ("🎯 Skill Check", "Let's test your knowledge in your strongest area!"),
                        ("🌟 Confidence Booster", "Time for some easy wins to pump you up!")
                    ]
                    
                    activity_name, activity_desc = np.random.choice(surprise_activities)
                    print(f"\n{activity_name}")
                    print(f"📋 {activity_desc}")
                    
                    if "rapid fire" in activity_name.lower():
                        print("\n⚡ Starting Rapid Fire Round...")
                        # Trigger a quick mixed quiz
                        random_unit = np.random.choice(list(UNITS.keys()))
                        self._handle_quiz_smart(random_unit, ['3'])
                    elif "fact" in activity_name.lower():
                        facts = [
                            "🌍 Tamil Nadu has the second-largest economy among Indian states!",
                            "📚 The TNPSC was established in 1956 and has conducted thousands of exams since!",
                            "🏛️ Tamil Nadu has more UNESCO World Heritage Sites than most other Indian states!",
                            "🎓 Consistent daily study for just 30 minutes is more effective than cramming for hours!",
                            "🧠 Your brain forms stronger memories when you're in a positive mood while learning!"
                        ]
                        print(f"\n💫 {np.random.choice(facts)}")
                        print("🤔 Want to test your knowledge on this topic? Just ask!")
                
                # Add mood-based responses
                elif any(word in cmd for word in ['happy', 'excited', 'pumped', 'ready', 'energetic']):
                    energy_responses = [
                        "YES! 🔥 I LOVE that energy! Let's channel it into some serious learning!",
                        "That's the spirit! 🚀 When you're excited to learn, magic happens!",
                        "AMAZING! 💫 Your enthusiasm is absolutely contagious! Let's ride this wave!",
                        "Perfect mindset! 🌟 Excited learners are successful learners! What shall we tackle?"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(energy_responses)}")
                    print("\n🎯 Let's use that energy:")
                    print("   ⚡ 'challenge me' - Bring on the tough questions!")
                    print("   🏆 'quiz marathon' - Multiple rounds of practice!")
                    print("   🎪 'surprise me' - Random fun activities!")
                
                elif any(word in cmd for word in ['sad', 'down', 'low', 'disappointed', 'upset']):
                    comfort_responses = [
                        "Hey, it's okay to feel down sometimes. 🤗 Learning has ups and downs, and that's totally normal!",
                        "I'm here for you! 💙 Bad days don't last, but resilient people like you do!",
                        "Feeling low? 🌈 Let's turn this around together. Sometimes a small win is all we need!",
                        "It's alright to have tough moments. 🌟 What matters is that you're still here, still trying!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(comfort_responses)}")
                    print("\n💝 Let's lift your spirits:")
                    print("   🌱 'easy quiz' - Start with some confidence builders")
                    print("   🎵 'motivate me' - Get some positive energy")
                    print("   📈 'my progress' - See how far you've come")
                    print("   🤗 Just chat with me - I'm here to listen!")
                
                # Motivation requests - Enhanced with personalized encouragement
                elif any(word in cmd for word in ['motivate', 'motivation', 'encourage', 'nervous', 'scared', 'worried', 'anxious']):
                    motivational_messages = [
                        "Listen, you're AMAZING! 💪 Every single question you practice is building your confidence brick by brick!",
                        "Hey, I believe in you 100%! 🌟 You wouldn't be here preparing if you didn't have what it takes!",
                        "You know what? 🎯 The fact that you're studying shows you're already ahead of the game!",
                        "Feeling nervous? That's totally normal! 😊 It means you care, and that's your superpower!",
                        "Remember: 🚀 Every expert was once a beginner, and every pro was once an amateur!",
                        "You're not just studying - you're investing in your future! 💎 That's incredibly brave!",
                        "Scared? Channel that energy! 🔥 Fear can be fuel when you use it right!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(motivational_messages)}")
                    
                    # Add specific encouragement based on their progress
                    stats = self.get_performance_insights()
                    if stats['total_questions'] > 0:
                        print(f"\n📈 Look at your progress: You've already engaged with {stats['total_questions']} questions!")
                        print("🏆 That's dedication right there!")
                    
                    print("\n💪 Here's your action plan to feel more confident:")
                    print("   🎯 Start small: 'easy quiz' to build momentum")
                    print("   📚 Review basics: 'stats for unit 1' to see what you know")
                    print("   🗓️ Get organized: 'plan for 60 days' for a stress-free schedule")
                    print("   🎉 Celebrate wins: Every correct answer is progress!")
                    
                    daily_affirmations = [
                        "🌟 Daily reminder: You're capable of incredible things!",
                        "💫 Today's mantra: Progress, not perfection!",
                        "🎯 Remember: Consistency beats intensity every time!",
                        "🚀 Your future self will thank you for today's effort!"
                    ]
                    print(f"\n{np.random.choice(daily_affirmations)}")
                
                # Handle uncertainty - more conversational and supportive
                elif any(word in cmd for word in ['i dont know', "i don't know", 'no idea', 'not sure', 'confused', 'help me', 'lost']):
                    self.user_session['mood'] = 'confused'
                    self.user_session['needs_encouragement'] = True
                    
                    # Check what they were just doing for context
                    if self.user_session['just_finished_quiz']:
                        responses = [
                            "Hey, no worries about that last quiz! 😊 Sometimes questions can be tricky. Want to try a different topic or review that one?",
                            "Totally get it! 🤗 Those questions can be challenging. How about we break it down step by step?",
                            "That's completely normal! 💪 Even experts get stumped sometimes. Want to tackle it from a different angle?"
                        ]
                    elif self.user_session['current_topic']:
                        responses = [
                            f"No stress about {self.user_session['current_topic']}! 😊 It can be a tough topic. Want to start with the basics?",
                            f"I hear you on {self.user_session['current_topic']} - it trips up lots of people! 🤗 Let's make it clearer together.",
                            f"Yeah, {self.user_session['current_topic']} can be confusing at first! 💡 How about we approach it differently?"
                        ]
                    else:
                        responses = [
                            "Hey, that's totally okay! 😊 Feeling lost is just part of learning. What's got you puzzled?",
                            "No worries at all! 🤗 Everyone feels confused sometimes. Tell me what's on your mind?",
                            "That's perfectly normal! 💪 Confusion means you're pushing your boundaries. What can I clarify?"
                        ]
                    
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                    
                    # Offer specific, contextual help
                    if self.user_session['quizzes_taken'] == 0:
                        print("🌱 Since you're just starting, how about we begin with something easy to build confidence?")
                        print("   • 'easy science questions' - Start gentle")
                        print("   • 'show me the units' - Get oriented")
                        print("   • 'what should I study first?' - Get guidance")
                    else:
                        print("🎯 Based on what we've done together, here's what might help:")
                        print("   • 'review my weak areas' - Focus on improvement")
                        print("   • 'easier questions' - Build back confidence")
                        print("   • 'explain [topic]' - Get clarification")
                        print("   • 'just chat' - Sometimes talking helps!")
                
                # Handle expressions of difficulty or frustration
                elif any(word in cmd for word in ['difficult', 'hard', 'tough', 'struggling', 'frustrated', 'stuck']):
                    responses = [
                        "I hear you! 💙 Learning can be challenging, but you're not alone in this journey!",
                        "It's tough sometimes, I get it! 🤗 But remember, every expert was once a beginner!",
                        "Feeling stuck is part of learning! 💪 Let's break it down into smaller, manageable pieces!",
                        "I understand it feels overwhelming! 🌟 But you've got this - let's tackle it step by step!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                    print("\n🎯 Let's make it easier! Try:")
                    print("   📚 Start with easy questions: 'easy quiz'")
                    print("   📖 Review a specific unit: 'stats for unit 1'")
                    print("   🗓️ Create a gentle study plan: 'plan for 45 days'")
                
                # Handle positive expressions
                elif any(word in cmd for word in ['good', 'great', 'awesome', 'cool', 'nice', 'love it', 'amazing']):
                    responses = [
                        "That's the spirit! 🎉 I love your enthusiasm! What shall we tackle next?",
                        "Awesome attitude! 🚀 You're going to do great! What would you like to explore?",
                        "Yes! 🌟 That positive energy will take you far! Ready for more?",
                        "I'm so glad you're enjoying this! 😊 Let's keep the momentum going!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                
                # Handle expressions of boredom or disinterest
                elif any(word in cmd for word in ['boring', 'bored', 'tired', 'sleepy', 'not interested']):
                    responses = [
                        "I get it! 😅 Sometimes studying feels like a drag. Let's make it more fun!",
                        "Totally understand! 🎮 How about we gamify your learning? Want to try a quick challenge?",
                        "Feeling bored? 🎪 Let's spice things up! How about some rapid-fire questions?",
                        "I hear you! 🎭 Learning should be engaging. Let's find what excites you!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                    print("\n🎯 Let's make it interesting:")
                    print("   ⚡ Quick 3-question challenge: 'quick quiz'")
                    print("   🎲 Random topic surprise: 'surprise me'")
                    print("   🏆 Check your achievements: 'my progress'")
                
                # Handle study-related queries
                elif any(word in cmd for word in ['study', 'learn', 'prepare', 'exam', 'test']):
                    responses = [
                        "Great mindset! 📚 Preparation is key to success. Let's create a solid plan!",
                        "Love the dedication! 🎯 Smart preparation leads to confident performance!",
                        "That's the winning attitude! 💪 Let's make your study time super effective!",
                        "Perfect! 🌟 Focused preparation is your secret weapon for TNPSC success!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                    print("\n📋 Study options:")
                    print("   📅 'plan for 30 days' - Complete study schedule")
                    print("   📊 'show my stats' - Track your progress")
                    print("   🎯 'quiz on [topic]' - Practice specific areas")
                    print("   🔍 'find [keyword]' - Search specific topics")
                
                # Natural conversation - like talking to a real study buddy
                else:
                    # Try to understand what they might mean based on context
                    context_responses = []
                    
                    # Check for partial matches or similar intent
                    if any(word in cmd for word in ['question', 'ask', 'test']):
                        context_responses = [
                            "Sounds like you want some practice questions! 🎯 Which topic are you thinking about?",
                            "Want me to quiz you? 😊 Just tell me what subject and I'll get some questions ready!",
                            "Ready for some questions? 🚀 Pick a topic and let's go!"
                        ]
                    elif any(word in cmd for word in ['study', 'learn', 'practice']):
                        context_responses = [
                            "Love the study motivation! 📚 What would you like to focus on today?",
                            "Great attitude! 💪 Which area needs some attention - or should I suggest something?",
                            "Perfect mindset for learning! 🌟 Tell me what's on your study agenda!"
                        ]
                    elif any(word in cmd for word in ['hard', 'difficult', 'tough']):
                        context_responses = [
                            "Yeah, some topics can be really challenging! 🤗 Which one's giving you trouble?",
                            "I totally get that! 💙 What's feeling difficult right now? Let's tackle it together!",
                            "Tough stuff, huh? 💪 That's where the real learning happens! What's the challenge?"
                        ]
                    elif len(cmd.split()) == 1 and cmd in ['science', 'history', 'geography', 'polity', 'economy']:
                        # Single word topic mention
                        context_responses = [
                            f"Ah, {cmd}! 🎯 Want to practice some {cmd} questions or learn more about it?",
                            f"Interested in {cmd}? 😊 I can quiz you, show stats, or just chat about it!",
                            f"{cmd.title()} - good choice! 🌟 What would you like to do with this topic?"
                        ]
                    
                    if context_responses:
                        print(f"\n🤖 Assistant: {np.random.choice(context_responses)}")
                    else:
                        # Genuinely didn't understand - be more conversational
                        natural_responses = [
                            "Hmm, I'm not quite following! 🤔 Could you tell me a bit more about what you're looking for?",
                            "I think I missed something there! 😅 Mind rephrasing that? I want to help but need to understand better!",
                            "Ooh, I'm not sure I caught that right! 🤗 What's on your mind? Just say it however feels natural!",
                            "I'm a bit lost on that one! 😊 No worries though - just tell me what you're thinking about!"
                        ]
                        print(f"\n🤖 Assistant: {np.random.choice(natural_responses)}")
                        
                        # Offer contextual suggestions based on their history
                        if self.user_session['quizzes_taken'] > 0:
                            print("🎯 Maybe you want to continue where we left off? Or try something completely different?")
                        else:
                            print("💡 If you're not sure where to start, I can suggest something based on what most people find helpful!")
                        
                        print("\n🗣️ You can literally just say things like:")
                        print("   • 'I want to practice' or 'test me on something'")
                        print("   • 'I'm struggling with history' or 'science is hard'")
                        print("   • 'What should I do?' or 'I'm bored'")
                        print("   • Or even just 'help!' - I'll figure it out! 😊")
                    
            except KeyboardInterrupt:
                print("\n\n🎉 Good luck with your TNPSC preparation! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 AVAILABLE COMMANDS:")
        print("-" * 60)
        print("  units              - List all TNPSC units")
        print("  stats <unit>       - Show statistics for a unit (e.g., 'stats 1')")
        print("  quiz <unit>        - Generate quiz for a unit (e.g., 'quiz 1')")
        print("  plan <days>        - Create study plan (e.g., 'plan 30')")
        print("  search <keyword>   - Search questions (e.g., 'search water')")
        print("  predict            - Predict question difficulty")
        print("  insights           - Show overall performance insights")
        print("  help               - Show this help message")
        print("  quit               - Exit the assistant")
        print("-" * 60)
    
    def _show_units(self):
        """Show all units"""
        print("\n📖 TNPSC EXAM UNITS:")
        print("-" * 60)
        for unit_num, unit_name in UNITS.items():
            count = len(self.df[self.df['unit'] == unit_num])
            print(f"  Unit {unit_num}: {unit_name} ({count} questions)")
        print("-" * 60)
    
    def _handle_stats_smart(self, unit_num):
        """Handle stats command with smart parsing"""
        try:
            if unit_num not in UNITS:
                print(f"\n❌ Invalid unit number. Please choose 1-{len(UNITS)}")
                self._show_units()
                return
            
            stats = self.get_unit_stats(unit_num)
            
            if stats:
                print(f"\n📊 STATISTICS FOR UNIT {unit_num}: {stats['unit_name']}")
                print("-" * 60)
                print(f"  Total Questions: {stats['total_questions']}")
                print(f"  Year Range: {stats['year_range']}")
                print(f"  Avg Question Length: {stats['avg_question_length']:.0f} characters")
                print(f"\n  Difficulty Distribution:")
                for diff, count in stats['difficulty_distribution'].items():
                    percentage = (count / stats['total_questions']) * 100
                    print(f"    {diff}: {count} ({percentage:.1f}%)")
                print("-" * 60)
            else:
                print(f"\n❌ No data available for Unit {unit_num}")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def _handle_quiz_smart(self, unit_num, extra_numbers):
        """Handle quiz command with smart parsing - Enhanced Interactive Version"""
        try:
            if unit_num not in UNITS:
                print(f"\n❌ Invalid unit number. Please choose 1-{len(UNITS)}")
                self._show_units()
                return
            
            # Check for number of questions in extra_numbers
            num_questions = int(extra_numbers[0]) if extra_numbers else 3  # Reduced for better interaction
            
            quiz = self.generate_quiz(unit_num, None, num_questions)
            
            self._clear_screen()
            print(f"\n🎯 INTERACTIVE QUIZ: {UNITS[unit_num]}")
            print("=" * 60)
            
            # Animated intro
            self._typing_effect("💡 Alright! Let's make this fun and interactive!", 0.04)
            self._typing_effect("   I'll ask questions one by one - just pick a, b, c, or d!", 0.03)
            self._typing_effect("   Feel free to say 'skip', 'hint', 'explain', or even 'pause' anytime!", 0.03)
            
            print("\n🎮 QUIZ CONTROLS:")
            print("   ⚡ Type your answer: a, b, c, or d")
            print("   💡 Need help? Say 'hint'")
            print("   📚 Want explanation? Say 'explain'") 
            print("   ⏭️ Skip question? Say 'skip'")
            print("   ⏸️ Take a break? Say 'pause'")
            print("=" * 60)
            
            input("\n🚀 Press Enter when you're ready to start! ")
            self._clear_screen()
            
            score = 0
            total_questions = len(quiz)
            
            # Update session tracking
            self.user_session['quizzes_taken'] += 1
            self.user_session['last_activity'] = 'quiz'
            
            for idx, (_, row) in enumerate(quiz.iterrows(), 1):
                # Animated question display
                print(f"\n� Qu[estion {idx}/{total_questions}")
                print("─" * 50)
                
                # Progress bar
                progress = self._show_progress_bar(idx-1, total_questions, 25)
                print(f"� Progrfess: {progress}")
                print()
                
                # Question with typing effect for engagement
                self._typing_effect(f"📚 {row['question']}", 0.02)
                
                print("\n📝 Your options:")
                options = [
                    f"   A) {row['option_a']}",
                    f"   B) {row['option_b']}",
                    f"   C) {row['option_c']}",
                    f"   D) {row['option_d']}"
                ]
                
                for option in options:
                    time.sleep(0.2)
                    print(option)
                
                # Difficulty indicator with emoji
                difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                diff_emoji = difficulty_emoji.get(row['difficulty'], "⚪")
                print(f"\n💭 {diff_emoji} Difficulty: {row['difficulty'].title()} | 📅 Year: {row['year']}")
                
                # Interactive answer collection with natural conversation
                while True:
                    # Make the prompt more conversational and context-aware
                    if idx == 1:
                        prompts = [
                            f"💬 What do you think, {self.user_session['user_name']}? (a/b/c/d): ",
                            f"🤔 Your first guess? (a/b/c/d): ",
                            f"🎯 Let's start! Pick one (a/b/c/d): "
                        ]
                    elif idx == total_questions:
                        prompts = [
                            f"🏁 Final question! What's your answer? (a/b/c/d): ",
                            f"🎯 Last one - you've got this! (a/b/c/d): ",
                            f"💪 Finish strong! Your choice? (a/b/c/d): "
                        ]
                    else:
                        prompts = [
                            f"💭 What feels right to you? (a/b/c/d): ",
                            f"🤔 Your instinct says? (a/b/c/d): ",
                            f"🎯 Go with your gut! (a/b/c/d): ",
                            f"💡 Which one, {self.user_session['user_name']}? (a/b/c/d): "
                        ]
                    
                    user_answer = input(f"\n{np.random.choice(prompts)}").strip().lower()
                    
                    # Handle natural language responses
                    if user_answer in ['a', 'b', 'c', 'd']:
                        pass  # Valid answer, continue
                    elif user_answer in ['option a', 'option b', 'option c', 'option d']:
                        user_answer = user_answer[-1]  # Extract the letter
                        print(f"📝 Got it! Going with option {user_answer.upper()}!")
                    elif any(word in user_answer for word in ['first', 'top', '1st']):
                        user_answer = 'a'
                        print("📝 Got it! Going with option A!")
                    elif any(word in user_answer for word in ['second', '2nd']):
                        user_answer = 'b'
                        print("📝 Got it! Going with option B!")
                    elif any(word in user_answer for word in ['third', '3rd']):
                        user_answer = 'c'
                        print("📝 Got it! Going with option C!")
                    elif any(word in user_answer for word in ['fourth', 'last', '4th']):
                        user_answer = 'd'
                        print("📝 Got it! Going with option D!")
                    
                    if user_answer in ['a', 'b', 'c', 'd']:
                        correct_answer = row['correct_option'].lower()
                        
                        # Show thinking animation
                        print(f"\n🤔 Let me check your answer...")
                        self._show_thinking_animation(1)
                        
                        # Update session tracking
                        self.user_session['total_questions_answered'] += 1
                        
                        if user_answer == correct_answer:
                            score += 1
                            self.user_session['correct_answers'] += 1
                            
                            # Animated celebration
                            celebration_emojis = ["🎉", "✨", "🚀", "💪", "⭐", "🏆", "🌟"]
                            for emoji in celebration_emojis[:3]:
                                print(f"\r{emoji} CORRECT! {emoji}", end="", flush=True)
                                time.sleep(0.3)
                            print()
                            
                            # Personalized encouragements based on performance
                            if self.user_session['correct_answers'] >= 5:
                                encouragements = [
                                    f"🎉 WOW {self.user_session['user_name']}! You're absolutely crushing it! That's another correct answer!",
                                    f"✨ INCREDIBLE! You're becoming a TNPSC expert right before my eyes!",
                                    f"🚀 PHENOMENAL! Your knowledge is really shining through!",
                                    f"💪 OUTSTANDING! You're on an amazing streak!",
                                    f"⭐ BRILLIANT! I'm so proud of your progress!"
                                ]
                            else:
                                encouragements = [
                                    f"🎉 Excellent, {self.user_session['user_name']}! You got it right!",
                                    "✨ Perfect! You're on fire!",
                                    "� Amazing!' Keep it up!",
                                    "💪 Brilliant! You know your stuff!",
                                    "⭐ Outstanding! Well done!"
                                ]
                            self._typing_effect(f"{np.random.choice(encouragements)}", 0.04)
                        else:
                            # Show wrong answer animation
                            print(f"\r❌ Not quite... The answer is {correct_answer.upper()}", end="", flush=True)
                            time.sleep(1)
                            print()
                            
                            # Adaptive motivational responses
                            accuracy = (self.user_session['correct_answers'] / max(1, self.user_session['total_questions_answered'])) * 100
                            
                            if accuracy < 30:
                                motivations = [
                                    f"�x Hey {self.user_session['user_name']}, it's {correct_answer.upper()}! Every master was once a disaster. You're learning!",
                                    f"🌱 The answer is {correct_answer.upper()}. Growth happens outside your comfort zone - you're growing!",
                                    f"🤗 It's {correct_answer.upper()}! Don't be hard on yourself. Progress, not perfection!",
                                    f"💪 {correct_answer.upper()} is correct! Every wrong answer teaches you something valuable!"
                                ]
                            else:
                                motivations = [
                                    f"💔 Oops! The correct answer is {correct_answer.upper()}. Don't worry, you'll get the next one!",
                                    f"🤔 Not quite! It's {correct_answer.upper()}. Learning from mistakes makes us stronger!",
                                    f"📚 Close! The answer is {correct_answer.upper()}. You're getting better with each question!",
                                    f"💡 Good try! It's {correct_answer.upper()}. Every expert was once a beginner!"
                                ]
                            self._typing_effect(f"{np.random.choice(motivations)}", 0.04)
                        
                        # Show explanation with animation
                        if 'explanation' in row and pd.notna(row['explanation']):
                            print(f"\n📖 Quick explanation:")
                            self._typing_effect(f"   {row['explanation']}", 0.03)
                        
                        # Pause before next question
                        if idx < total_questions:
                            input(f"\n⏭️ Press Enter for question {idx + 1}...")
                            self._clear_screen()
                        
                        break
                    
                    elif user_answer in ['skip', 's']:
                        print(f"\n⏭️ Skipped! The answer was {row['correct_option'].upper()}.")
                        if 'explanation' in row and pd.notna(row['explanation']):
                            print(f"📖 Explanation: {row['explanation']}")
                        break
                    
                    elif user_answer in ['hint', 'h']:
                        # Give a hint by eliminating one wrong option
                        options = ['a', 'b', 'c', 'd']
                        correct = row['correct_option'].lower()
                        options.remove(correct)
                        wrong_option = np.random.choice(options)
                        print(f"💡 Hint: Option {wrong_option.upper()} is definitely wrong!")
                        continue
                    
                    elif user_answer in ['explain', 'e']:
                        if 'explanation' in row and pd.notna(row['explanation']):
                            print(f"📖 Explanation: {row['explanation']}")
                        else:
                            print("📖 No explanation available for this question.")
                        continue
                    
                    elif user_answer in ['quit', 'exit', 'stop']:
                        print("\n🛑 Quiz stopped. Come back anytime to continue learning!")
                        return
                    
                    else:
                        print("🤔 Please enter a, b, c, d, or say 'skip', 'hint', 'explain'")
                        continue
            
            # Animated final results
            self._clear_screen()
            print("\n🏁 QUIZ COMPLETE!")
            print("=" * 60)
            
            # Animated score reveal
            print("📊 Calculating your results...")
            self._show_thinking_animation(2)
            
            percentage = (score / total_questions) * 100
            overall_accuracy = (self.user_session['correct_answers'] / max(1, self.user_session['total_questions_answered'])) * 100
            
            # Dramatic score reveal
            print(f"\n🎯 {self.user_session['user_name']}'s Performance:")
            print("─" * 40)
            
            # Animated progress bars
            print(f"📊 This Quiz: ", end="")
            time.sleep(0.5)
            quiz_progress = self._show_progress_bar(score, total_questions, 20)
            print(f"{quiz_progress}")
            
            print(f"📈 Overall: ", end="")
            time.sleep(0.5)
            overall_progress = self._show_progress_bar(self.user_session['correct_answers'], 
                                                    self.user_session['total_questions_answered'], 20)
            print(f"{overall_progress}")
            
            print(f"🎯 Quizzes Completed: {self.user_session['quizzes_taken']}")
            print("─" * 40)
            
            # Personalized feedback based on overall performance
            if percentage >= 80:
                if overall_accuracy >= 70:
                    print("🌟 PHENOMENAL! You're consistently performing at an expert level! 🏆")
                else:
                    print("🌟 EXCELLENT! You're absolutely crushing it! Keep up the fantastic work!")
            elif percentage >= 60:
                if overall_accuracy >= 60:
                    print("👍 SOLID PERFORMANCE! You're building strong, consistent knowledge! 💪")
                else:
                    print("👍 GOOD JOB! You're doing well! A bit more practice and you'll be perfect!")
            elif percentage >= 40:
                if self.user_session['quizzes_taken'] >= 3:
                    print("📚 STEADY PROGRESS! I can see you're improving with each quiz! 📈")
                else:
                    print("📚 KEEP GOING! You're learning! Practice makes perfect - don't give up!")
            else:
                if self.user_session['quizzes_taken'] == 1:
                    print("💪 GREAT START! Everyone begins somewhere! You're brave for taking the first step! 🌟")
                else:
                    print("💪 PERSISTENCE PAYS OFF! You're not giving up, and that's what winners do! 🔥")
            
            # Personalized suggestions based on performance and history
            if overall_accuracy >= 70:
                suggestions = [
                    f"🔥 You're ready for harder challenges! Try 'difficult quiz {unit_num}'!",
                    "🏆 Consider taking a full practice test across all units!",
                    "🎯 You could help others - your knowledge is solid!",
                    "📚 Time to explore advanced topics in this unit!"
                ]
            elif overall_accuracy >= 50:
                suggestions = [
                    f"💡 Want to strengthen this area? Try 'quiz {unit_num}' again!",
                    "📈 Check 'my progress' to see your improvement trends!",
                    "📅 A structured 'plan for 30 days' might help consolidate your knowledge!",
                    "🔍 Search for specific weak topics with 'find [topic]'!"
                ]
            else:
                suggestions = [
                    f"🌱 Let's build confidence! Try 'easy questions {unit_num}'!",
                    "📚 'Study plan' can help you approach this systematically!",
                    "💪 'Motivate me' when you need encouragement!",
                    "🤗 Remember: I'm here to support you every step of the way!"
                ]
            
            print(f"\n{np.random.choice(suggestions)}")
            
            # Interactive achievement system
            achievements = self._check_achievements()
            if achievements:
                print("\n🏆 ACHIEVEMENTS UNLOCKED!")
                print("─" * 30)
                for achievement in achievements:
                    self._typing_effect(f"🎉 {achievement}", 0.04)
                    time.sleep(0.5)
            
            # Interactive next steps
            print(f"\n🚀 What's next, {self.user_session['user_name']}?")
            next_suggestions = self._get_post_quiz_suggestions(percentage, overall_accuracy)
            
            for i, suggestion in enumerate(next_suggestions, 1):
                print(f"   {i}. {suggestion}")
            
            print("=" * 60)
            
            # Mark quiz as finished for context
            self.user_session['just_finished_quiz'] = True
            self.user_session['current_topic'] = UNITS[unit_num]
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Don't worry! Try again or ask for help! 😊")
    
    def _check_achievements(self):
        """Check for new achievements and return them"""
        achievements = []
        
        # Quiz milestones
        if self.user_session['quizzes_taken'] == 1:
            achievements.append("🌟 First Quiz Complete - You've started your journey!")
        elif self.user_session['quizzes_taken'] == 5:
            achievements.append("🏅 Quiz Warrior - 5 quizzes completed!")
        elif self.user_session['quizzes_taken'] == 10:
            achievements.append("🏆 Quiz Master - 10 quizzes conquered!")
        
        # Question milestones
        if self.user_session['total_questions_answered'] == 10:
            achievements.append("📚 Knowledge Seeker - 10 questions answered!")
        elif self.user_session['total_questions_answered'] == 25:
            achievements.append("⭐ Dedicated Learner - 25 questions tackled!")
        elif self.user_session['total_questions_answered'] == 50:
            achievements.append("🎯 Study Champion - 50 questions mastered!")
        
        # Accuracy achievements
        accuracy = (self.user_session['correct_answers'] / max(1, self.user_session['total_questions_answered'])) * 100
        if accuracy >= 80 and self.user_session['total_questions_answered'] >= 10:
            achievements.append("🌟 Accuracy Expert - 80%+ success rate!")
        elif accuracy >= 90 and self.user_session['total_questions_answered'] >= 20:
            achievements.append("💎 Precision Master - 90%+ accuracy!")
        
        return achievements
    
    def _get_post_quiz_suggestions(self, quiz_percentage, overall_accuracy):
        """Get personalized suggestions after quiz completion"""
        suggestions = []
        
        if quiz_percentage >= 80:
            suggestions.extend([
                "🔥 Try a harder topic - you're ready for the challenge!",
                "🎯 Take a full practice test across all units",
                "📚 Explore advanced concepts in this subject"
            ])
        elif quiz_percentage >= 60:
            suggestions.extend([
                f"💪 Practice more {self.user_session['current_topic']} questions",
                "📊 Check your progress across all topics",
                "📅 Create a focused study plan"
            ])
        else:
            suggestions.extend([
                "🌱 Review the basics in this topic",
                "💡 Try easier questions to build confidence",
                "🤗 Get some motivation and encouragement"
            ])
        
        # Always include some general options
        suggestions.extend([
            "🎲 Surprise me with something different!",
            "💬 Just chat - I'm here to listen"
        ])
        
        return suggestions[:4]
    
    def _handle_plan_smart(self, days):
        """Handle study plan command with smart parsing - Enhanced Interactive Version"""
        try:
            if days <= 0:
                print("\n❌ Please provide a positive number of days")
                return
            
            plan = self.get_study_plan(days)
            
            print(f"\n�️ PEDRSONALIZED STUDY PLAN FOR {days} DAYS")
            print("=" * 60)
            
            # Add motivational intro based on timeline
            if days <= 15:
                print("🔥 INTENSIVE PREP MODE! Short timeline means focused action!")
            elif days <= 30:
                print("⚡ BALANCED APPROACH! Perfect timeline for thorough preparation!")
            elif days <= 60:
                print("🌟 COMPREHENSIVE PLAN! Plenty of time for deep learning!")
            else:
                print("🎯 MARATHON PREPARATION! Steady pace for lasting knowledge!")
            
            print(f"\n📊 Your Plan Overview:")
            print(f"   📚 Questions per day: {plan['questions_per_day']}")
            print(f"   🎯 Units to master: {len(plan['units_to_cover'])}")
            print(f"   ⏰ Daily study time: {2 if days <= 30 else 1.5} hours recommended")
            
            # Show detailed first week
            print(f"\n📋 YOUR FIRST WEEK BREAKDOWN:")
            print("-" * 40)
            
            for day in sorted(plan['recommended_schedule'].keys())[:7]:
                day_info = plan['recommended_schedule'][day]
                if day_info:
                    subject = day_info[0]['subject']
                    questions = day_info[0]['questions_count']
                    
                    # Add day-specific motivation
                    day_emoji = ["🌅", "💪", "🔥", "⚡", "🌟", "🎯", "🏆"][day-1] if day <= 7 else "📚"
                    
                    print(f"\n  {day_emoji} Day {day}: {subject}")
                    print(f"     • Practice {questions} questions")
                    print(f"     • Review key concepts")
                    print(f"     • Take notes on weak areas")
            
            if days > 7:
                print(f"\n  📈 ... (detailed schedule continues for all {days} days)")
                print("     💡 Each week builds on the previous one!")
            
            # Add success tips
            print(f"\n🎯 SUCCESS TIPS FOR YOUR {days}-DAY JOURNEY:")
            print("-" * 40)
            
            tips = [
                "🌅 Study at the same time daily to build a habit",
                "📝 Keep a learning journal to track progress",
                "🎉 Celebrate small wins - they add up!",
                "💧 Stay hydrated and take regular breaks",
                "🤝 Join study groups or find a study buddy",
                "📱 Use this assistant daily for practice!"
            ]
            
            for tip in tips[:4]:  # Show 4 tips
                print(f"   {tip}")
            
            # Personalized encouragement
            print(f"\n💪 PERSONAL MESSAGE:")
            if days <= 15:
                message = "You've got this! Intensive preparation can be incredibly effective. Stay focused, trust the process, and give it your all!"
            elif days <= 30:
                message = "Perfect timeline! You have enough time to cover everything thoroughly. Consistency is your secret weapon!"
            else:
                message = "Amazing foresight! With this much time, you can become truly confident in every topic. Steady progress wins the race!"
            
            print(f"   🌟 {message}")
            
            # Interactive follow-up
            print(f"\n🤔 WHAT'S NEXT?")
            print("   📚 'quiz me' - Start practicing right now!")
            print("   📊 'my stats' - See your current progress")
            print(f"   🎯 'unit 1 quiz' - Begin with your first topic")
            print("   💪 'motivate me' - Get pumped for the journey!")
            
            print("=" * 60)
            print("🚀 Your success journey starts NOW! I believe in you! 🌟")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Don't worry! Let's try creating your plan again! 😊")
    
    def _handle_search_smart(self, keyword):
        """Handle search command with smart parsing"""
        try:
            results = self.search_questions(keyword)
            
            print(f"\n� SEARCaH RESULTS FOR '{keyword}'")
            print("=" * 60)
            
            if len(results) == 0:
                print("  No questions found matching your search.")
                print("  Try different keywords or check spelling.")
            else:
                print(f"  Found {len(results)} questions:\n")
                
                for idx, (_, row) in enumerate(results.head(5).iterrows(), 1):
                    print(f"{idx}. {row['question']}")
                    print(f"   Unit: {UNITS[row['unit']]} | Difficulty: {row['difficulty']}")
                    print()
                
                if len(results) > 5:
                    print(f"  ... and {len(results) - 5} more results")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def _handle_predict(self, user_input):
        """Handle predict difficulty command"""
        print("\n🔮 DIFFICULTY PREDICTION")
        print("-" * 60)
        
        try:
            unit = int(input("  Enter unit number (1-8): "))
            if unit not in UNITS:
                print(f"  ❌ Invalid unit number")
                return
            
            year = int(input("  Enter year (e.g., 2023): "))
            question = input("  Enter question text: ")
            
            if not question:
                print("  ❌ Question text cannot be empty")
                return
            
            difficulty = self.predict_difficulty(unit, year, question)
            
            print(f"\n  ✅ Predicted Difficulty: {difficulty}")
            print(f"  📚 Unit: {UNITS[unit]}")
            print("-" * 60)
            
        except ValueError:
            print("  ❌ Please provide valid inputs")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    def _handle_insights(self):
        """Handle insights command"""
        insights = self.get_performance_insights()
        
        print("\n📈 OVERALL INSIGHTS")
        print("=" * 60)
        print(f"  Total Questions: {insights['total_questions']}")
        print(f"  Units Covered: {insights['units_covered']}")
        print(f"  Year Coverage: {insights['year_coverage']}")
        print(f"\n  Difficulty Breakdown:")
        for diff, count in insights['difficulty_breakdown'].items():
            percentage = (count / insights['total_questions']) * 100
            print(f"    {diff}: {count} ({percentage:.1f}%)")
        print(f"\n  Questions per Unit:")
        for unit, count in insights['questions_per_unit'].items():
            print(f"    Unit {unit} ({UNITS[unit]}): {count}")
        print("=" * 60)
    
    def _handle_general(self, user_input):
        """Handle general conversation"""
        responses = {
            'hi': "Hello! How can I help you with your TNPSC preparation today?",
            'hello': "Hi there! Ready to prepare for TNPSC? Type 'help' to see what I can do.",
            'thanks': "You're welcome! Keep up the great work!",
            'thank you': "Happy to help! Good luck with your studies!",
        }
        
        cmd = user_input.lower()
        
        if cmd in responses:
            print(f"\n🤖 Assistant: {responses[cmd]}")
        else:
            print(f"\n🤖 Assistant: I'm not sure how to help with that.")
            print("   Type 'help' to see available commands.")

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
