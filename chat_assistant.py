#!/usr/bin/env python3
"""
TNPSC AI Chat Assistant - Main Application
Interactive chat-based exam preparation assistant
"""

import os
import sys
import pandas as pd
import numpy as np
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
        print("🚀 Initializing TNPSC AI Chat Assistant...")
        print("="*60)
        
        # Load core assistant
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # Train ML models
        self._train_models()
        
        print("✅ Chat Assistant Ready!")
        print("="*60)
    
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
        print("\n" + "="*60)
        print("🎓 TNPSC AI EXAM PREPARATION ASSISTANT")
        print("="*60)
        print("\nHey! 👋 I'm your AI buddy for TNPSC exam prep!")
        print("\nI can help you with:")
        print("  • Generate practice quizzes 📝")
        print("  • Create study plans 📅")
        print("  • Show unit statistics 📊")
        print("  • Search questions by topic 🔍")
        print("  • Predict question difficulty 🎯")
        print("\nJust talk naturally! Try saying:")
        print("  'give me a quiz on science'")
        print("  'show me stats for history'")
        print("  'I need a 30 day study plan'")
        print("  'find questions about water'")
        print("\nType 'help' anytime if you're stuck!")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Parse natural language input
                parsed = self._parse_natural_input(user_input)
                cmd = parsed['text']
                
                # Exit commands
                if any(word in cmd for word in ['quit', 'exit', 'bye', 'goodbye', 'stop']):
                    print("\n🎉 Good luck with your TNPSC preparation! Goodbye!")
                    break
                
                # Help command
                elif any(word in cmd for word in ['help', 'commands', 'what can you do']):
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
                
                # Greetings - super casual
                elif any(word in cmd for word in ['hi', 'hello', 'hey', 'greetings', 'sup', 'yo', 'wassup']):
                    responses = [
                        "Hey there! 👋 Ready to crush that TNPSC exam? What do you need?",
                        "Hello! 😊 Let's get you prepared! Want a quiz or study plan?",
                        "Hi! 🎯 What can I help you with today? Quiz, stats, or study plan?"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                
                # Thanks - casual responses
                elif any(word in cmd for word in ['thanks', 'thank you', 'appreciate', 'thx', 'ty']):
                    responses = [
                        "You're welcome! Keep crushing it! 💪",
                        "No problem! You got this! 🎯",
                        "Anytime! Keep up the awesome work! ⭐",
                        "Happy to help! Go ace that exam! 🚀"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                
                # Motivation requests
                elif any(word in cmd for word in ['motivate', 'motivation', 'encourage', 'nervous', 'scared']):
                    responses = [
                        "You've got this! 💪 Every question you practice gets you closer to success!",
                        "Believe in yourself! 🌟 You're putting in the work, and it will pay off!",
                        "Stay focused and consistent! 🎯 Success is just around the corner!",
                        "Don't worry! 😊 With practice and dedication, you'll ace this exam!"
                    ]
                    print(f"\n🤖 Assistant: {np.random.choice(responses)}")
                
                # General conversation - more helpful
                else:
                    print(f"\n🤖 Assistant: Hmm, I'm not sure I got that. Let me help you out!")
                    print("   You can say things like:")
                    print("   • 'give me a quiz on science'")
                    print("   • 'show me history stats'")
                    print("   • 'I need a 30 day plan'")
                    print("   • 'find questions about water'")
                    print("   • 'show all units'")
                    print("\n   Just talk naturally - I'll understand! 😊")
                    
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
        """Handle quiz command with smart parsing"""
        try:
            if unit_num not in UNITS:
                print(f"\n❌ Invalid unit number. Please choose 1-{len(UNITS)}")
                self._show_units()
                return
            
            # Check for number of questions in extra_numbers
            num_questions = int(extra_numbers[0]) if extra_numbers else 5
            
            quiz = self.generate_quiz(unit_num, None, num_questions)
            
            print(f"\n📝 QUIZ: {UNITS[unit_num]}")
            print("=" * 60)
            
            for idx, (_, row) in enumerate(quiz.iterrows(), 1):
                print(f"\nQ{idx}. {row['question']}")
                print(f"    Difficulty: {row['difficulty']} | Year: {row['year']}")
            
            print("=" * 60)
            print(f"✅ Generated {len(quiz)} questions for {UNITS[unit_num]}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def _handle_plan_smart(self, days):
        """Handle study plan command with smart parsing"""
        try:
            if days <= 0:
                print("\n❌ Please provide a positive number of days")
                return
            
            plan = self.get_study_plan(days)
            
            print(f"\n📅 STUDY PLAN FOR {days} DAYS")
            print("=" * 60)
            print(f"  Questions per day: {plan['questions_per_day']}")
            print(f"  Total units to cover: {len(plan['units_to_cover'])}")
            print(f"\n  Recommended Schedule:")
            
            for day in sorted(plan['recommended_schedule'].keys())[:7]:  # Show first 7 days
                print(f"\n  Day {day}:")
                for item in plan['recommended_schedule'][day]:
                    print(f"    • {item['subject']} ({item['questions_count']} questions)")
            
            if days > 7:
                print(f"\n  ... (schedule continues for {days} days)")
            
            print("=" * 60)
            print(f"✅ Study plan created! Stay consistent and you'll do great! 💪")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
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
