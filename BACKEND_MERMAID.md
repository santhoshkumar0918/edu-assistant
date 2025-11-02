# 🏗️ TNPSC Study Buddy Backend - Mermaid Architecture

## **Backend System Architecture for chat_assistant.py**

---

## **1️⃣ BACKEND CLASS ARCHITECTURE**

```mermaid
classDiagram
    class TNPSCStudyBuddyV3 {
        -assistant: TNPSCExamAssistant
        -df: DataFrame
        -intent_matcher: HybridIntentMatcher
        -emotion_detector: EmotionDetector
        -knowledge_recommender: KnowledgeRecommender
        -profile_manager: ProfileManager
        -adaptive_engine: AdaptiveLearningEngine
        -model: RandomForestClassifier
        -context: ConversationContext
        -debug_mode: bool
        +__init__(debug_mode)
        +chat()
        +_train_models()
        +_handle_quiz(entities)
        +_handle_stats(entities)
        +_handle_plan(entities)
        +_parse_entities(text)
        +_get_response(intent, tone)
    }
    
    class HybridIntentMatcher {
        -intents: Dict[str, List[str]]
        -vectorizer: TfidfVectorizer
        -training_vectors: ndarray
        +__init__()
        +detect(text): Tuple[str, float]
        +_init_vectorizer()
    }
    
    class EmotionDetector {
        -emotion_patterns: Dict[str, List[str]]
        +__init__()
        +detect(text): Tuple[str, float]
        +get_response(emotion): str
    }
    
    class AdaptiveLearningEngine {
        -profile: UserProfile
        +__init__(profile)
        +update_performance(unit, correct, difficulty)
        +get_weak_units(): List[int]
        +get_strong_units(): List[int]
        +get_recommended_unit(): int
    }
    
    class KnowledgeRecommender {
        -knowledge_base: Dict[str, Dict]
        +__init__()
        +search(query): Optional[Dict]
        +teach(topic): str
    }
    
    class ProfileManager {
        -data_dir: str
        -profiles_file: str
        -current_profile: UserProfile
        +__init__(data_dir)
        +load_profiles(): Dict
        +save_profiles(profiles)
        +create_profile(name): UserProfile
        +load_profile(name): UserProfile
        +list_profiles(): List[str]
    }
    
    class UserProfile {
        +name: str
        +total_sessions: int
        +quizzes_taken: int
        +questions_answered: int
        +correct_answers: int
        +avg_score: float
        +weakest_units: List[int]
        +strongest_units: List[int]
        +unit_performance: Dict[int, Dict]
        +tone_preference: str
        +last_login: str
    }
    
    class ConversationContext {
        +last_intent: str
        +last_unit: int
        +last_topic: str
        +last_emotion: str
        +history: List[Dict]
    }
    
    TNPSCStudyBuddyV3 --> HybridIntentMatcher
    TNPSCStudyBuddyV3 --> EmotionDetector
    TNPSCStudyBuddyV3 --> AdaptiveLearningEngine
    TNPSCStudyBuddyV3 --> KnowledgeRecommender
    TNPSCStudyBuddyV3 --> ProfileManager
    TNPSCStudyBuddyV3 --> ConversationContext
    ProfileManager --> UserProfile
    AdaptiveLearningEngine --> UserProfile
```

---

## **2️⃣ BACKEND DATA FLOW**

```mermaid
flowchart TB
    Start([User Input]) --> Parse[_parse_entities]
    Parse --> Intent[HybridIntentMatcher.detect]
    Parse --> Emotion[EmotionDetector.detect]
    
    Intent --> Router{Intent Router}
    Emotion --> Router
    
    Router --> |quiz| HandleQuiz[_handle_quiz]
    Router --> |stats| HandleStats[_handle_stats]
    Router --> |plan| HandlePlan[_handle_plan]
    Router --> |search| HandleSearch[_handle_search]
    Router --> |help| ShowHelp[_show_help]
    
    HandleQuiz --> Adaptive[AdaptiveLearningEngine]
    HandleStats --> Profile[ProfileManager]
    HandlePlan --> Profile
    
    Adaptive --> Recommend[get_recommended_unit]
    Adaptive --> Update[update_performance]
    
    Profile --> Load[load_profile]
    Profile --> Save[save_profile]
    
    Load --> Storage[(JSON Storage)]
    Save --> Storage
    
    Recommend --> Response[Generate Response]
    Update --> Response
    ShowHelp --> Response
    
    Response --> Context[Update ConversationContext]
    Context --> Output([User Output])
    
    style Start fill:#e1f5e1
    style Router fill:#ffd700
    style Storage fill:#dda0dd
    style Output fill:#ffe1e1
```

---

## **3️⃣ ML MODEL BACKEND PIPELINE**

```mermaid
flowchart LR
    Input[Raw Question Data] --> Feature[Feature Engineering]
    
    Feature --> F1[question_length]
    Feature --> F2[word_count]
    Feature --> F3[has_numbers]
    Feature --> F4[has_question_mark]
    Feature --> F5[avg_word_length]
    Feature --> F6[unit]
    Feature --> F7[year]
    
    F1 --> Vector[7D Feature Vector]
    F2 --> Vector
    F3 --> Vector
    F4 --> Vector
    F5 --> Vector
    F6 --> Vector
    F7 --> Vector
    
    Vector --> Split[Train/Test Split]
    Split --> Train[RandomForestClassifier.fit]
    
    Train --> Model[Trained Model]
    Model --> Predict[predict_proba]
    
    Predict --> Easy[Easy: 0.X]
    Predict --> Medium[Medium: 0.Y]
    Predict --> Hard[Hard: 0.Z]
    
    Easy --> Decision[Difficulty Decision]
    Medium --> Decision
    Hard --> Decision
    
    style Vector fill:#ffd700
    style Model fill:#ff6b6b
    style Decision fill:#4ecdc4
```

---

## **4️⃣ BACKEND STORAGE ARCHITECTURE**

```mermaid
erDiagram
    USER_PROFILES {
        string name PK
        int total_sessions
        int quizzes_taken
        int questions_answered
        int correct_answers
        float avg_score
        json weakest_units
        json strongest_units
        json unit_performance
        string tone_preference
        string last_login
    }
    
    UNIT_PERFORMANCE {
        int unit_id PK
        string user_name FK
        int attempts
        int correct
        float avg_difficulty
    }
    
    NLU_LOGS {
        string timestamp PK
        string user_input
        string predicted_intent
        float confidence
        boolean was_correct
    }
    
    CONVERSATION_HISTORY {
        string session_id PK
        string user_name FK
        json messages
        string last_intent
        int last_unit
        string last_emotion
    }
    
    USER_PROFILES ||--o{ UNIT_PERFORMANCE : has
    USER_PROFILES ||--o{ CONVERSATION_HISTORY : creates
```

---

## **5️⃣ BACKEND API ENDPOINTS (Conceptual)**

```mermaid
flowchart TB
    subgraph "Chat Interface"
        Chat[chat_assistant.py]
    end
    
    subgraph "Core Services"
        Intent[Intent Detection Service]
        Emotion[Emotion Detection Service]
        ML[ML Prediction Service]
        Adaptive[Adaptive Learning Service]
        Knowledge[Knowledge Service]
    end
    
    subgraph "Data Layer"
        Profile[Profile Manager]
        Storage[(JSON Files)]
        Logs[(CSV Logs)]
    end
    
    Chat --> Intent
    Chat --> Emotion
    Chat --> ML
    Chat --> Adaptive
    Chat --> Knowledge
    
    Intent --> Profile
    Adaptive --> Profile
    
    Profile --> Storage
    Intent --> Logs
    
    style Chat fill:#e1f5e1
    style ML fill:#ffd700
    style Storage fill:#dda0dd
```

---

## **6️⃣ BACKEND PROCESSING SEQUENCE**

```mermaid
sequenceDiagram
    participant User
    participant Main as TNPSCStudyBuddyV3
    participant Intent as HybridIntentMatcher
    participant Emotion as EmotionDetector
    participant ML as RandomForestClassifier
    participant Adaptive as AdaptiveLearningEngine
    participant Profile as ProfileManager
    participant Storage as JSON Storage
    
    User->>Main: "give me 5 questions on economy"
    
    Main->>Intent: detect(text)
    Intent-->>Main: ("quiz", 0.85)
    
    Main->>Emotion: detect(text)
    Emotion-->>Main: ("neutral", 0.5)
    
    Main->>Profile: load_profile(user)
    Profile->>Storage: read JSON
    Storage-->>Profile: user_data
    Profile-->>Main: UserProfile
    
    Main->>Adaptive: get_recommended_unit()
    Adaptive-->>Main: unit=6 (Economy)
    
    Main->>ML: predict_difficulty(questions)
    ML-->>Main: [Easy: 0.3, Medium: 0.5, Hard: 0.2]
    
    Main->>Adaptive: update_performance(results)
    Adaptive->>Profile: save_profile(updated)
    Profile->>Storage: write JSON
    
    Main-->>User: Display 5 Economy questions
```

---

## **7️⃣ BACKEND ERROR HANDLING**

```mermaid
flowchart TB
    Input[User Input] --> Validate{Input Validation}
    
    Validate --> |Valid| Process[Process Request]
    Validate --> |Invalid| Error1[Input Error Handler]
    
    Process --> Intent[Intent Detection]
    Intent --> |Success| Route[Route to Handler]
    Intent --> |Fail| Error2[Intent Error Handler]
    
    Route --> Handler[Execute Handler]
    Handler --> |Success| Response[Generate Response]
    Handler --> |Fail| Error3[Handler Error Handler]
    
    Error1 --> Fallback[Fallback Response]
    Error2 --> Fallback
    Error3 --> Fallback
    
    Response --> Log[Log Interaction]
    Fallback --> Log
    
    Log --> Output[Return to User]
    
    style Error1 fill:#ff6b6b
    style Error2 fill:#ff6b6b
    style Error3 fill:#ff6b6b
    style Fallback fill:#ffa500
```

---

## **8️⃣ BACKEND CONFIGURATION**

```mermaid
graph TB
    subgraph "Configuration Layer"
        Config[config.json]
        Env[Environment Variables]
    end
    
    subgraph "Model Configuration"
        RF[Random Forest Config]
        NLU[NLU Config]
        Emotion[Emotion Config]
    end
    
    subgraph "Storage Configuration"
        JSON[JSON Storage Path]
        CSV[CSV Logs Path]
        Output[Output Directory]
    end
    
    Config --> RF
    Config --> NLU
    Config --> Emotion
    
    Env --> JSON
    Env --> CSV
    Env --> Output
    
    RF --> |n_estimators=100<br/>max_depth=10| MLModel[ML Model]
    NLU --> |threshold=0.25<br/>max_features=500| IntentModel[Intent Matcher]
    
    style Config fill:#87ceeb
    style MLModel fill:#ffd700
    style IntentModel fill:#98fb98
```

---

## **9️⃣ BACKEND PERFORMANCE MONITORING**

```mermaid
flowchart LR
    subgraph "Performance Metrics"
        Response[Response Time]
        Accuracy[Model Accuracy]
        Memory[Memory Usage]
        Intent[Intent Accuracy]
    end
    
    subgraph "Monitoring Tools"
        Timer[Time Tracker]
        Profiler[Memory Profiler]
        Logger[Performance Logger]
    end
    
    subgraph "Alerts & Reports"
        Threshold{Performance<br/>Threshold}
        Alert[Performance Alert]
        Report[Performance Report]
    end
    
    Response --> Timer
    Memory --> Profiler
    Accuracy --> Logger
    Intent --> Logger
    
    Timer --> Threshold
    Profiler --> Threshold
    Logger --> Threshold
    
    Threshold --> |Below Threshold| Alert
    Threshold --> |Above Threshold| Report
    
    style Threshold fill:#ffd700
    style Alert fill:#ff6b6b
    style Report fill:#90ee90
```

---

## **🔟 BACKEND DEPLOYMENT ARCHITECTURE**

```mermaid
graph TB
    subgraph "Application Layer"
        App[chat_assistant.py]
        CLI[Command Line Interface]
    end
    
    subgraph "Service Layer"
        Intent[Intent Service]
        ML[ML Service]
        Profile[Profile Service]
        Adaptive[Adaptive Service]
    end
    
    subgraph "Data Layer"
        JSON[(JSON Files)]
        CSV[(CSV Logs)]
        Model[(Trained Models)]
    end
    
    subgraph "External Dependencies"
        SKL[scikit-learn]
        Pandas[pandas]
        NumPy[numpy]
    end
    
    CLI --> App
    App --> Intent
    App --> ML
    App --> Profile
    App --> Adaptive
    
    Intent --> JSON
    Profile --> JSON
    ML --> Model
    Adaptive --> CSV
    
    App --> SKL
    App --> Pandas
    App --> NumPy
    
    style App fill:#ffd700
    style JSON fill:#87ceeb
    style Model fill:#ff6b6b
```

---

## **📊 BACKEND TECHNICAL SPECIFICATIONS**

### **Core Components:**
- **Main Class**: `TNPSCStudyBuddyV3`
- **ML Engine**: Random Forest (100 estimators)
- **NLU Engine**: TF-IDF + Cosine Similarity
- **Storage**: JSON-based persistence
- **Context**: Conversation memory management

### **Performance Targets:**
- **Response Time**: < 1 second
- **Memory Usage**: < 100MB
- **Model Accuracy**: > 85%
- **Intent Accuracy**: > 90%

### **Scalability:**
- **Single User**: Current implementation
- **Multi User**: Profile-based separation
- **Concurrent**: Thread-safe operations
- **Storage**: File-based (JSON/CSV)

### **Dependencies:**
- **Python**: 3.8+
- **scikit-learn**: ML algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **JSON**: Data persistence

---

**This backend architecture supports the complete TNPSC Study Buddy system with adaptive learning, emotion detection, and personalized user experiences.**