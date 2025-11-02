# 🏗️ Simple Backend Block Diagram

## **TNPSC Study Buddy - Simple Backend Architecture**

```mermaid
flowchart TB
    User[👤 User Input] --> Interface[💬 Chat Interface]
    
    Interface --> Processing[🔧 Input Processing]
    
    Processing --> AI[🧠 AI Engine]
    
    AI --> ML[🤖 ML Model]
    AI --> Knowledge[📚 Knowledge Base]
    AI --> Profile[👤 User Profile]
    
    ML --> Prediction[📊 Difficulty Prediction]
    Knowledge --> Teaching[📖 Concept Teaching]
    Profile --> Personalization[🎯 Personalized Learning]
    
    Prediction --> Response[💭 Response Generation]
    Teaching --> Response
    Personalization --> Response
    
    Response --> Storage[💾 Data Storage]
    Response --> Output[📱 User Output]
    
    Storage --> Files[(📁 JSON Files)]
    
    style User fill:#e1f5e1
    style AI fill:#ffd700
    style ML fill:#ff6b6b
    style Output fill:#ffe1e1
    style Files fill:#dda0dd
```

---

## **Alternative Simple View**

```mermaid
graph LR
    A[User] --> B[Chat System]
    B --> C[AI Brain]
    C --> D[ML Model]
    C --> E[Knowledge]
    C --> F[Profile]
    D --> G[Response]
    E --> G
    F --> G
    G --> H[Storage]
    G --> A
    
    style A fill:#e1f5e1
    style C fill:#ffd700
    style D fill:#ff6b6b
    style G fill:#4ecdc4
    style H fill:#dda0dd
```

---

## **Super Simple Version**

```mermaid
flowchart TD
    Input[📝 User Question] 
    --> Process[⚙️ AI Processing]
    --> Generate[🎯 Smart Response]
    --> Save[💾 Save Progress]
    --> Output[✅ Answer to User]
    
    style Input fill:#e1f5e1
    style Process fill:#ffd700
    style Generate fill:#ff6b6b
    style Output fill:#4ecdc4
```

---

## **Component Overview**

```mermaid
graph TB
    subgraph "Frontend"
        UI[Chat Interface]
    end
    
    subgraph "Backend Core"
        Engine[AI Engine]
        Model[ML Model]
        Data[User Data]
    end
    
    subgraph "Storage"
        Files[JSON Files]
    end
    
    UI --> Engine
    Engine --> Model
    Engine --> Data
    Data --> Files
    
    style Engine fill:#ffd700
    style Model fill:#ff6b6b
    style Files fill:#dda0dd
```

---

**Pick any of these simple diagrams for your presentation! 🎯**