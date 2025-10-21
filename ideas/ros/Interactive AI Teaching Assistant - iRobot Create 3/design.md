# Technical Design Document: Interactive AI Teaching Assistant Robot for Algonquin College

## Architecture Overview

### System Architecture
The platform follows an educational robotics architecture leveraging the iRobot Create® 3 platform with ROS 2, integrating autonomous navigation, AI-powered learning support, and interactive demonstrations.

```
┌─────────────────────────────────────────────────────────────────┐
│          Student Mobile App + Faculty Dashboard (Web)          │
│     Request Assistance, Schedule, Analytics, Visualizations    │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API / WebSocket
┌────────────────────────────┴────────────────────────────────────┐
│              Teaching Coordinator Node                          │
│     Session Management, Student Profiles, Queue Handling        │
└─────┬──────┬──────┬──────┬──────────────────────────────────────┘
      │      │      │      │
┌─────▼──┐ ┌─▼────────┐ ┌──▼──────┐ ┌──▼──────────────────────┐
│Student │ │Voice Q&A │ │Navigate │ │  Interactive Demo      │
│Recogn. │ │RAG+LLM   │ │Nav2     │ │  ROS 2 Viz on Tablet   │
└────────┘ └──────────┘ └─────────┘ └────────────────────────────┘
    │          │            │              │
    ▼          ▼            ▼              ▼
[USB       [USB Mic]  [Create® 3]    [Tablet Display]
Webcam]                [Sensors]
```

### Key Design Principles
- **iRobot Create® 3 Native**: Leverage built-in sensors, safety features, odometry
- **Educational First**: All features designed to enhance learning, not just assist
- **Privacy by Design**: FIPPA compliant, opt-in recognition, encrypted student data
- **Scalable**: Support multiple students per day with queue management
- **Adaptive**: Personalize based on student level and learning history

### Technology Stack
- **Platform**: iRobot Create® 3 + Raspberry Pi 4 (4GB) or Jetson Nano
- **Framework**: ROS 2 Humble with Navigation2
- **AI**: MediaPipe, Whisper, LangChain, LLM (Mistral 7B or GPT-4 API)
- **Storage**: PostgreSQL (student profiles, learning history)
- **Interface**: React (mobile app), Flask (REST API), WebRTC (demonstrations)

## Component Design

### 1. Navigation Node (`campus_navigation_node`)

**Responsibility**: Autonomous navigation using iRobot Create® 3 and Navigation2, responding to student requests and scheduled locations.

**Class Structure**:
```python
class CampusNavigationNode(Node):
    def __init__(self):
        super().__init__('campus_navigation_node')
        self.create3_interface: Create3Interface  # iRobot SDK
        self.nav2_client: Nav2Client
        self.campus_map: OccupancyGrid
        self.schedule_manager: LocationScheduler
        self.request_queue: RequestQueue
        
    def navigate_to_student(self, location: Pose) -> None:
        """Navigate to student who requested assistance"""
        
    def patrol_scheduled_location(self) -> None:
        """Autonomous patrol based on course timetable"""
        
    def handle_obstacle(self, obstacle_detected: bool) -> None:
        """Safe navigation in crowded hallways"""
```

**Publishers**:
- `/robot/position` (geometry_msgs/PoseStamped): Current location
- `/robot/status` (custom_msgs/RobotStatus): Battery, availability

**Parameters**:
- `patrol_locations` (list): CS lab, robotics lab, library study area
- `patrol_schedule` (yaml): Time-based schedule tied to courses
- `safety_clearance_m` (float): Minimum clearance from obstacles (default: 0.5)

**Dependencies**: iRobot Create® 3 ROS 2 API, Navigation2, campus map YAML

### 2. Student Recognition Node (`student_recognition_node`)

**Responsibility**: Detect and recognize registered students, manage opt-in enrollment, retrieve learning profiles for personalization.

**Class Structure**:
```python
class StudentRecognitionNode(Node):
    def __init__(self):
        super().__init__('student_recognition_node')
        self.camera: USBCamera
        self.mediapipe_face: mp.solutions.face_detection.FaceDetection
        self.face_recognizer: FaceRecognitionModel  # dlib or FaceNet
        self.student_db: PostgreSQLClient
        self.enrollment_manager: EnrollmentSystem
        
    def detect_student(self, frame: np.ndarray) -> Optional[StudentProfile]:
        """MediaPipe detection → Face recognition → Profile lookup"""
        
    def enroll_new_student(self, student_id: str, consent: bool) -> None:
        """Opt-in enrollment with photo capture and consent"""
        
    def get_learning_history(self, student_id: str) -> LearningProfile:
        """Retrieve previous interactions and knowledge level"""
```

**Publishers**:
- `/student/recognized` (custom_msgs/StudentProfile): Identified student

**Parameters**:
- `recognition_confidence_threshold` (float): Minimum confidence (default: 0.85)
- `opt_in_required` (bool): Enforce consent (default: true)
- `data_retention_days` (int): Delete after N days (FIPPA compliance)

**Dependencies**: MediaPipe 0.10+, dlib or face_recognition library, PostgreSQL

### 3. Voice Q&A Node (`voice_qa_node`)

**Responsibility**: Handle student questions using Whisper STT, RAG with course materials, and LLM for natural answers.

**Class Structure**:
```python
class VoiceQANode(Node):
    def __init__(self):
        super().__init__('voice_qa_node')
        self.microphone: MicrophoneArray
        self.whisper_model: whisper.Whisper
        self.rag_pipeline: LangChainRAG  # Vector DB + retrieval
        self.llm: LLMClient  # Local (Mistral) or API (GPT-4)
        self.tts_engine: pyttsx3.Engine
        self.conversation_context: ConversationHistory
        
    def transcribe_question(self, audio: np.ndarray) -> str:
        """Whisper speech-to-text"""
        
    def answer_with_rag(self, question: str, context: List[str]) -> str:
        """Retrieve course materials, generate answer with LLM"""
        
    def speak_answer(self, text: str) -> None:
        """Text-to-speech with natural prosody"""
```

**Subscribers**:
- `/student/recognized` (custom_msgs/StudentProfile): For personalization

**Publishers**:
- `/qa/interaction` (custom_msgs/QALog): Log for learning history

**Parameters**:
- `whisper_model_size` (str): base/small (default: base)
- `rag_source_dirs` (list): Course material directories
- `llm_backend` (str): local/openai (default: local)
- `conversation_timeout_minutes` (int): Context window (default: 30)

**Dependencies**: Whisper, LangChain, ChromaDB (vector DB), Mistral-7B or OpenAI API

### 4. Engagement Detection Node (`engagement_detection_node`)

**Responsibility**: Analyze student facial expressions to detect confusion, boredom, focus; adapt explanations accordingly.

**Class Structure**:
```python
class EngagementDetectionNode(Node):
    def __init__(self):
        super().__init__('engagement_detection_node')
        self.camera: USBCamera
        self.mediapipe_face_mesh: mp.solutions.face_mesh.FaceMesh
        self.confusion_classifier: ConfusionModel  # Custom TF model
        self.adaptive_explainer: AdaptiveContentSelector
        
    def analyze_expression(self, landmarks: Dict) -> EngagementState:
        """Classify: focused/confused/bored from facial landmarks"""
        
    def adapt_content(self, state: EngagementState, current_topic: str) -> str:
        """Simplify/elaborate based on confusion detection"""
```

**Publishers**:
- `/student/engagement` (custom_msgs/EngagementLevel): Real-time engagement

**Parameters**:
- `confusion_threshold` (float): Detection sensitivity (default: 0.7)
- `adaptation_trigger` (str): immediate/after_3_detections

**Dependencies**: MediaPipe, TensorFlow (custom confusion model)

### 5. Interactive Demo Node (`interactive_demo_node`)

**Responsibility**: Visualize ROS 2 concepts on tablet, synchronize with verbal explanations, handle voice-triggered demonstrations.

**Class Structure**:
```python
class InteractiveDemoNode(Node):
    def __init__(self):
        super().__init__('interactive_demo_node')
        self.tablet_display: TabletInterface
        self.ros2_visualizer: ROS2TopicVisualizer
        self.demo_library: Dict[str, Demo]  # SLAM, path planning, etc.
        
    def trigger_demo(self, demo_name: str) -> None:
        """Voice command: 'Show me SLAM' → Start demonstration"""
        
    def visualize_ros2_state(self) -> None:
        """Display nodes, topics, sensor data on tablet"""
        
    def explain_concept(self, concept: str) -> None:
        """Synchronized explanation while showing visualization"""
```

**Subscribers**: All relevant ROS 2 topics for visualization

**Publishers**:
- `/demonstration/active` (custom_msgs/Demo): Current demo state

**Parameters**:
- `available_demos` (list): SLAM, path_planning, sensor_fusion, etc.
- `tablet_resolution` (str): 1920x1200 (default for 10" tablet)

**Dependencies**: PyQt5 or React (tablet UI), ROS 2 topic introspection

## Data Models

### PostgreSQL Database Schema

**students** table:
```sql
CREATE TABLE students (
    student_id VARCHAR(20) PRIMARY KEY,
    name_encrypted BYTEA,  -- AES-256 encrypted
    face_encoding BYTEA,   -- Face recognition embeddings
    consent_given BOOLEAN DEFAULT FALSE,
    enrolled_at TIMESTAMP DEFAULT NOW(),
    course_level VARCHAR(50),  -- freshman, sophomore, etc.
    last_interaction TIMESTAMP
);
```

**interaction_history** table:
```sql
CREATE TABLE interaction_history (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(20) REFERENCES students(student_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    topic VARCHAR(100),
    question_text TEXT,
    answer_text TEXT,
    engagement_score FLOAT,
    confusion_detected BOOLEAN
);
```

**course_materials** table:
```sql
CREATE TABLE course_materials (
    id SERIAL PRIMARY KEY,
    course_code VARCHAR(20),
    material_type VARCHAR(50),  -- textbook, lecture, documentation
    content_vector VECTOR(768),  -- For RAG embeddings
    source_file VARCHAR(255),
    page_number INT
);
```

### Custom ROS 2 Messages

**StudentProfile.msg**:
```
std_msgs/Header header
string student_id
string name
string course_level
string[] previous_topics
float32 avg_engagement_score
```

**EngagementLevel.msg**:
```
std_msgs/Header header
string state  # focused/confused/bored
float32 confidence
string suggested_action  # simplify/elaborate/continue
```

## API Design

### REST API (Flask)

**Endpoints**:
- `POST /api/request-assistance`: Student requests robot
  ```json
  {"student_id": "A00123456", "location": {"building": "T", "room": "127"}}
  ```
- `GET /api/schedule`: Get robot schedule
- `POST /api/enroll`: New student enrollment with consent
- `GET /api/faculty/dashboard`: Faculty analytics (aggregated data)
- `DELETE /api/student/{id}/data`: FIPPA-compliant data deletion

### ROS 2 Topics

**QoS Policies**:
- **STUDENT_INTERACTION**: Reliable, transient-local, depth=10
- **NAVIGATION**: Best effort, volatile, depth=5
- **DEMONSTRATION**: Reliable, volatile, depth=1

## Security Considerations

### Privacy (FIPPA Compliance)
- Facial recognition opt-in with explicit consent
- AES-256 encryption for student data at rest
- TLS for all network communication
- Audit logging of data access
- Right to data deletion within 48 hours
- Anonymized aggregated data for faculty dashboards

### Physical Safety
- iRobot Create® 3 built-in cliff detection and bumpers
- 0.5m clearance maintained from students
- Audible alerts in crowded areas
- Emergency stop via mobile app and physical button

## Performance Considerations

### Real-Time Requirements
- Voice recognition: <2 seconds latency
- Student recognition: <1 second
- Navigation planning: <5 seconds
- RAG Q&A: <10 seconds response
- Engagement detection: Real-time (30 FPS)

### Resource Optimization
- Whisper base model (1GB RAM) for speech recognition
- Mistral 7B quantized (4GB) for local LLM
- Face recognition caching for known students
- Course material vector DB indexed for fast retrieval

## Error Handling

### Node-Level Recovery
```python
class VoiceQANode(Node):
    def answer_with_rag(self, question):
        try:
            context = self.rag_pipeline.retrieve(question)
            answer = self.llm.generate(question, context)
            return answer
        except LLMTimeoutError:
            return "I'm having trouble processing that. Would you like me to connect you with a teaching assistant?"
        except RAGRetrievalError:
            return "I don't have information on that topic in my database. Let me note this for the instructor."
```

### System-Level Safeguards
- **Graceful Degradation**: Continue basic navigation if cameras fail
- **Fallback Options**: Connect to human TA if LLM fails repeatedly
- **Battery Management**: Auto-return to charging dock below 20%
- **Queue Management**: Inform students of wait time if multiple requests

This design provides a robust, educational, privacy-compliant foundation for the AI teaching assistant robot.
