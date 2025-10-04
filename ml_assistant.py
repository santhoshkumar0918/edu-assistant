import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # For saving/loading the model

# Import the existing TNPSCExamAssistant class and other necessary functions
# We'll need to ensure tnpsc_assistant.py is in the same directory or properly imported
from tnpsc_assistant import TNPSCExamAssistant, UNITS, create_sample_syllabus, create_sample_questions, load_syllabus, load_questions, extract_key_topics, improved_parse_previous_paper, parse_all_pdfs, list_uploaded_pdfs, upload_pdf_instructions, extract_text_from_pdf, detect_question_pattern, parse_options, guess_unit_from_content, process_pdfs_and_save

class MLAssistant:
    def __init__(self):
        self.tnpsc_assistant = TNPSCExamAssistant()
        self.intents = {
            "list_units": [
                "list units", "show units", "what are the units", "units",
                "tell me all units", "display units", "available units"
            ],
            "get_syllabus": [
                "syllabus for unit", "show syllabus for unit", "unit syllabus",
                "what is in unit", "tell me about unit", "syllabus unit"
            ],
            "generate_quiz": [
                "quiz me", "give me a quiz", "start a quiz", "generate quiz",
                "quiz on unit", "questions for unit", "practice questions"
            ],
            "generate_study_plan": [
                "study plan", "create a study plan", "plan my study",
                "generate a study plan for", "study schedule"
            ],
            "manage_pdf": [
                "pdf", "manage pdfs", "process pdfs", "upload pdf",
                "handle pdfs", "pdf options"
            ],
            "add_question": [
                "add question", "add a question", "new question",
                "create question", "add custom question"
            ],
            "show_statistics": [
                "stats", "show statistics", "display stats",
                "question statistics", "how many questions"
            ],
            "show_help": [
                "help", "commands", "what can I do", "assist me",
                "show commands", "help me"
            ],
            "exit_program": [
                "exit", "quit", "bye", "goodbye", "stop"
            ]
        }
        self.vectorizer = TfidfVectorizer()
        self.intent_classifier = LogisticRegression(max_iter=1000)
        self._train_intent_classifier()

    def _train_intent_classifier(self):
        corpus = []
        labels = []
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                corpus.append(phrase)
                labels.append(intent)

        X = self.vectorizer.fit_transform(corpus)
        y = labels

        # For a small dataset, we'll train on all data. For larger datasets, use train_test_split.
        self.intent_classifier.fit(X, y)
        print("Intent classifier trained.")

    def predict_intent(self, text):
        text_vector = self.vectorizer.transform([text])
        intent = self.intent_classifier.predict(text_vector)[0]
        return intent

    def extract_entities(self, text, intent):
        entities = {}
        text_lower = text.lower()

        if intent in ["get_syllabus", "generate_quiz", "generate_study_plan", "analyze_performance"]:
            # Extract unit number
            unit_match = re.search(r'(unit|for unit|on unit|unit number)\s*(\d+)', text_lower)
            if unit_match:
                entities['unit_num'] = int(unit_match.group(2))
            else:
                # Try to extract unit name and map to number
                for u_num, u_name in UNITS.items():
                    if u_name.lower() in text_lower:
                        entities['unit_num'] = u_num
                        break

        if intent == "generate_quiz":
            # Extract count
            count_match = re.search(r'(\d+)\s*(questions|quiz)', text_lower)
            if count_match:
                entities['count'] = int(count_match.group(1))

        if intent == "generate_study_plan":
            # Extract days
            days_match = re.search(r'(\d+)\s*(days|day)', text_lower)
            if days_match:
                entities['days'] = int(days_match.group(1))

        return entities

    def process_command(self, user_input):
        intent = self.predict_intent(user_input)
        entities = self.extract_entities(user_input, intent)

        response = ""
        if intent == "list_units":
            response += "\nAVAILABLE UNITS:\n"
            for num, name in UNITS.items():
                response += f"{num}. {name}\n"
        elif intent == "get_syllabus":
            unit_num = entities.get('unit_num')
            if unit_num:
                if unit_num in UNITS:
                    response += f"\nSYLLABUS FOR UNIT {unit_num}: {UNITS[unit_num]}\n"
                    response += "-" * 50 + "\n"
                    response += self.tnpsc_assistant.syllabus[unit_num] + "\n"
                else:
                    response = f"Unit {unit_num} not found. Available units: {list(UNITS.keys())}\n"
            else:
                response = "Please specify a unit number (e.g., 'syllabus for unit 1')\n"
        elif intent == "generate_quiz":
            unit_num = entities.get('unit_num')
            count = entities.get('count', 5) # Default to 5 questions
            quiz = self.tnpsc_assistant.generate_quiz(unit_num, count)
            if 'error' in quiz:
                response = f"Error: {quiz['error']}\n"
            else:
                response += f"\n{quiz['title']} ({quiz['num_questions']} questions)\n"
                response += "=" * 50 + "\n"
                for i, q in enumerate(quiz['questions']):
                    response += f"\nQ{i+1}. {q['question']}\n"
                    for opt, text in q['options'].items():
                        response += f"  {opt}) {text}\n"
                    response += f"Correct answer: {q['correct_option'].upper()}) {q['options'][q['correct_option'].upper()]}\n"
                    if 'explanation' in q:
                        response += f"Explanation: {q['explanation']}\n"
                    response += "-" * 50 + "\n"
        elif intent == "generate_study_plan":
            days = entities.get('days', 30) # Default to 30 days
            unit_num = entities.get('unit_num') # Optional unit for study plan
            plan = self.tnpsc_assistant.generate_study_plan(unit_num=unit_num, days_remaining=days)
            if 'error' in plan:
                response = f"Error: {plan['error']}\n"
            else:
                response += f"\nSTUDY PLAN FOR {plan['days_total']} DAYS\n"
                response += f"Covering {plan['units_covered']} units\n"
                response += "=" * 50 + "\n"
                for day_plan in plan['plan']:
                    response += f"\nDay {day_plan['day']}: Unit {day_plan['unit']} - {day_plan['unit_name']}\n"
                    response += f"Focus topics: {', '.join(day_plan['topics'])}\n"
                    response += "Activities:\n"
                    for i, activity in enumerate(day_plan['activities']):
                        response += f"  {i+1}. {activity}\n"
                    response += "-" * 50 + "\n"
        elif intent == "manage_pdf":
            # This intent will lead to a sub-menu or specific PDF actions
            response = "Entering PDF management mode. You can 'list pdfs', 'upload instructions', or 'process all pdfs'. Type 'back' to return to main menu.\n"
            # For now, we'll just return this message. A full implementation would involve a sub-loop.
        elif intent == "add_question":
            response = "To add a question, please provide details like unit, question text, options, and correct answer.\n"
            # This would require a multi-turn interaction, which we can implement later.
        elif intent == "show_statistics":
            # Call the show_statistics function from tnpsc_assistant.py
            # For now, we'll just print to console directly from the function
            # In a more advanced chat, we'd capture output or return structured data
            response = "Displaying statistics...\n"
            self.tnpsc_assistant.show_statistics(self.tnpsc_assistant) # Pass the assistant instance
        elif intent == "show_help":
            response += "\nAVAILABLE COMMANDS:\n"
            response += "=" * 50 + "\n"
            for cmd, phrases in self.intents.items():
                response += f"- {cmd.replace('_', ' ')}: {phrases[0]}\n" # Show first example phrase
            response += "=" * 50 + "\n"
        elif intent == "exit_program":
            response = "Thank you for using the TNPSC Exam Assistant! Goodbye!\n"
        else:
            response = "I'm not sure how to help with that. Please try rephrasing or type 'help' for available commands.\n"
        return response

def main_ml_assistant():
    print("Initializing ML-driven TNPSC Exam Assistant...")
    assistant = MLAssistant()
    print("ML Assistant ready. Type 'help' for commands or 'exit' to quit.")

    while True:
        user_input = input("\nEnter your command: ").strip()
        if user_input.lower() == 'exit':
            print(assistant.process_command(user_input))
            break
        elif user_input.lower() == 'back': # Special command for PDF menu
            print("Returning to main menu.")
            continue # Skip processing as it's a navigation command

        response = assistant.process_command(user_input)
        print(response)

if __name__ == "__main__":
    main_ml_assistant()
