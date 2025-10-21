# Technical Design Document: Adaptive Learning Companion Robot for Special Needs Education

## Architecture Overview

### System Architecture
The platform follows a therapeutic robotics architecture with five specialized ROS 2 nodes orchestrated around child safety, privacy, and adaptive learning principles.

```
┌─────────────────────────────────────────────────────────────────┐
│              Caregiver Dashboard (Web/Mobile)                   │
│         Real-time Monitoring, Progress Tracking, Config         │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket/REST
┌────────────────────────────┴────────────────────────────────────┐
│                  Session Coordinator Node                       │
│      State Management, Safety Monitoring, Data Aggregation      │
└─────┬──────┬──────┬──────┬──────────────────────────────────────┘
      │      │      │      │
┌─────▼──┐ ┌─▼──────┐ ┌───▼─────┐ ┌──▼──────────────────────────┐
│Emotion │ │ Speech │ │Learning │ │   Motion & Safety           │
│Detect  │ │Interface│ │ Engine  │ │   Therapeutic Movement      │
└────────┘ └─────────┘ └─────────┘ └─────────────────────────────┘
    │          │            │              │
    ▼          ▼            ▼              ▼
[RGB-D    [Microphone  [SQLite DB]  [Robot Actuators]
 Camera]    Array]                   [Emergency Stop]
```

### Key Design Principles
- **Child Safety First**: All operations filtered through safety validator; emergency stop always accessible
- **Privacy by Design**: Local-first processing; encrypted data storage; opt-in cloud features
- **Therapeutic Validity**: Evidence-based interaction patterns reviewed by clinical advisors
- **Neurodiversity Inclusive**: Accommodates diverse expression patterns and sensory needs
- **Graceful Degradation**: System adapts when sensors fail; never crashes during child interaction

### Technology Stack
- **ROS 2**: Humble/Iron (Python rclpy client)
- **AI Models**: MediaPipe Holistic, Whisper (base/small), TensorFlow/PyTorch emotion classifiers
- **Data**: SQLite with AES-256 encryption
- **Dashboard**: WebSocket (Socket.io) + REST API, responsive web interface
- **Platform**: Ubuntu 22.04 LTS, Python 3.10+

## Component Design

### 1. Emotion & Engagement Detection Node (`emotion_detection_node`)

**Responsibility**: Analyze child's facial expressions, gaze, and posture to detect emotional state and engagement level while respecting neurodiversity.

**Class Structure**:
```python
class EmotionDetectionNode(Node):
    def __init__(self):
        super().__init__('emotion_detection_node')
        self.mediapipe_holistic: mp.solutions.holistic.Holistic
        self.emotion_classifier: TensorFlowModel  # Custom pediatric model
        self.engagement_scorer: EngagementAnalyzer
        self.camera: RealSenseCamera
        self.neurodiversity_adapter: ExpressionNormalizer
        
    def camera_callback(self) -> None:
        """Process RGB-D frame, detect features, classify emotion"""
        
    def classify_emotion(self, landmarks: Dict) -> EmotionState:
        """6-class emotion classifier accounting for neurodiversity"""
        
    def compute_engagement(self, face_data: Dict, pose_data: Dict) -> EngagementLevel:
        """Gaze + posture → focused/distracted/overwhelmed"""
        
    def publish_states(self, emotion: EmotionState, engagement: EngagementLevel) -> None:
        """Publish to /perception/emotion and /perception/engagement"""
```

**Publishers**:
- `/perception/emotion` (custom_msgs/EmotionState): Emotion classification with confidence
- `/perception/engagement` (custom_msgs/EngagementLevel): Engagement state and gaze metrics

**Parameters**:
- `camera_device` (str): RealSense device ID
- `fps_target` (int): Processing frame rate (default: 15)
- `emotion_threshold` (float): Minimum confidence for emotion classification (default: 0.6)
- `neurodiversity_mode` (bool): Enable expression variation accommodation (default: true)

**Dependencies**: MediaPipe 0.10+, TensorFlow 2.13+, pyrealsense2, NumPy

### 2. Adaptive Speech Interface Node (`speech_interface_node`)

**Responsibility**: Handle child speech recognition, intent parsing, response generation, and adaptive text-to-speech with age-appropriate vocabulary.

**Class Structure**:
```python
class SpeechInterfaceNode(Node):
    def __init__(self):
        super().__init__('speech_interface_node')
        self.whisper_model: whisper.Whisper
        self.audio_stream: sounddevice.InputStream
        self.vad: VoiceActivityDetector
        self.intent_parser: ChildIntentParser
        self.context_manager: ConversationContext
        self.response_generator: AdaptiveResponseGenerator
        self.tts_engine: TextToSpeechWithProsody
        
    def audio_callback(self, indata: np.ndarray) -> None:
        """Buffer audio, apply VAD, trigger transcription"""
        
    def transcribe_child_speech(self, audio: np.ndarray) -> str:
        """Whisper with child-speech optimizations"""
        
    def parse_intent(self, transcript: str, context: Dict) -> Intent:
        """Extract meaning accounting for developmental speech patterns"""
        
    def generate_response(self, intent: Intent, emotion: EmotionState) -> str:
        """Create age-appropriate, emotionally-aware response"""
        
    def speak(self, text: str, pace: float, tone: str) -> None:
        """TTS with adjustable pace and emotional tone"""
```

**Subscribers**:
- `/perception/emotion` (custom_msgs/EmotionState): For emotionally-aware responses

**Publishers**:
- `/audio/transcript` (std_msgs/String): Raw transcription
- `/audio/intent` (custom_msgs/Intent): Parsed command/response
- `/audio/speaking` (std_msgs/Bool): Robot speaking status

**Parameters**:
- `whisper_model_size` (str): Model size (tiny/base/small, default: base)
- `vad_threshold` (float): Voice activity threshold (default: 0.5)
- `vocabulary_level` (int): Complexity level 1-3 (default: 2)
- `speech_pace` (float): TTS speed multiplier (default: 0.9)

**Dependencies**: OpenAI Whisper, sounddevice, pyttsx3/gTTS

### 3. Personalized Learning Engine Node (`learning_engine_node`)

**Responsibility**: Manage curriculum, track progress, adapt difficulty, generate activities, and provide rewards based on child's performance and emotional state.

**Class Structure**:
```python
class LearningEngineNode(Node):
    def __init__(self):
        super().__init__('learning_engine_node')
        self.db: SQLiteDatabase  # Encrypted learner profiles
        self.curriculum: CurriculumManager
        self.adaptive_algorithm: DifficultyAdapter
        self.activity_generator: ActivityContentGenerator
        self.performance_tracker: PerformanceMonitor
        self.reward_system: RewardFeedbackGenerator
        
    def select_next_activity(self, profile: LearnerProfile) -> Activity:
        """Choose activity based on performance, engagement, goals"""
        
    def present_activity(self, activity: Activity) -> None:
        """Generate content and publish to robot/dashboard"""
        
    def process_response(self, response: ActivityResponse) -> None:
        """Track performance, update difficulty, provide feedback"""
        
    def adapt_difficulty(self, performance: PerformanceMetrics) -> None:
        """Adjust level: 3 correct = up, 3 incorrect = down"""
        
    def generate_reward(self, correct: bool) -> Reward:
        """Create age-appropriate visual/auditory reward"""
```

**Subscribers**:
- `/perception/engagement` (custom_msgs/EngagementLevel): Adapt to engagement drops
- `/learning/response` (custom_msgs/ActivityResponse): Child's answers

**Publishers**:
- `/learning/activity` (custom_msgs/ActivityState): Current activity details
- `/learning/performance` (custom_msgs/PerformanceMetric): Progress updates
- `/learning/reward` (custom_msgs/RewardFeedback): Visual/audio rewards

**Parameters**:
- `database_path` (str): SQLite database location
- `difficulty_threshold` (int): Responses before difficulty change (default: 3)
- `session_duration_minutes` (int): Target session length (default: 20)
- `activity_types` (list): Enabled activities (default: all 5)

**Dependencies**: SQLite3, NumPy, pandas

### 4. Therapeutic Motion Node (`motion_node`)

**Responsibility**: Generate child-safe, predictable movements following sensory integration therapy principles with multiple safety layers.

**Class Structure**:
```python
class TherapeuticMotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')
        self.robot_interface: RobotActuatorInterface
        self.motion_planner: SensoryIntegrationPlanner
        self.safety_validator: ChildSafetyValidator
        self.gesture_recognizer: GestureInteractionHandler
        self.calming_protocol: CalmingMovementGenerator
        self.emergency_stop: EmergencyStopHandler
        
    def motion_command_callback(self, cmd: MotionCommand) -> None:
        """Validate, plan, execute motion with safety checks"""
        
    def plan_therapeutic_movement(self, goal: MotionGoal) -> Trajectory:
        """Generate predictable, gentle trajectories"""
        
    def validate_safety(self, trajectory: Trajectory) -> bool:
        """Check speed, force limits, proximity to child"""
        
    def execute_movement(self, trajectory: Trajectory) -> None:
        """Send to actuators with continuous monitoring"""
        
    def trigger_calming_mode(self) -> None:
        """Slow breathing guidance, gentle movements"""
        
    def emergency_stop(self) -> None:
        """Immediate halt of all motion"""
```

**Subscribers**:
- `/perception/emotion` (custom_msgs/EmotionState): Trigger calming on anxiety
- `/motion/command` (custom_msgs/MotionCommand): Movement requests

**Publishers**:
- `/motion/status` (custom_msgs/MotionStatus): Current movement state
- `/safety/status` (custom_msgs/SafetyStatus): Safety system state

**Parameters**:
- `max_velocity` (float): Maximum movement speed (default: 0.3 m/s)
- `max_acceleration` (float): Maximum acceleration (default: 0.2 m/s²)
- `sensory_mode` (str): full/reduced/quiet (default: full)
- `emergency_stop_pin` (int): GPIO pin for physical e-stop

**Dependencies**: ROS 2 Control, robot-specific drivers

### 5. Session Coordinator Node (`session_coordinator_node`)

**Responsibility**: Orchestrate all nodes, manage session state, aggregate data, enforce safety protocols, communicate with dashboard.

**Class Structure**:
```python
class SessionCoordinatorNode(Node):
    def __init__(self):
        super().__init__('session_coordinator_node')
        self.state_machine: SessionStateMachine
        self.safety_monitor: SafetyMonitoringSystem
        self.data_aggregator: SessionDataAggregator
        self.dashboard_server: WebSocketDashboardServer
        self.privacy_enforcer: DataPrivacyController
        
    def start_session(self, learner_id: str) -> None:
        """Initialize session, load profile, start nodes"""
        
    def monitor_safety(self) -> None:
        """Continuous safety checks across all systems"""
        
    def aggregate_session_data(self) -> SessionSummary:
        """Compile performance, engagement, emotions"""
        
    def send_dashboard_update(self, update: Dict) -> None:
        """Real-time WebSocket push to caregiver dashboard"""
        
    def end_session(self) -> SessionSummary:
        """Save progress, generate summary, cleanup"""
```

**Subscribers**: All node topics for monitoring and coordination

**Publishers**:
- `/session/state` (custom_msgs/SessionState): Current session status
- `/safety/alert` (custom_msgs/SafetyAlert): Critical safety events

**Services**:
- `/session/start` (custom_srvs/StartSession): Begin learning session
- `/session/stop` (custom_srvs/StopSession): End session gracefully
- `/session/emergency_stop` (std_srvs/Trigger): Immediate system halt

**Parameters**:
- `dashboard_port` (int): WebSocket server port (default: 8080)
- `safety_check_hz` (float): Safety monitoring frequency (default: 10 Hz)

## Data Models

### Custom ROS 2 Messages

**EmotionState.msg**:
```
std_msgs/Header header
string emotion                # happy/sad/frustrated/confused/anxious/neutral
float32 confidence            # 0.0 to 1.0
string previous_emotion       # For transition tracking
duration emotion_duration     # Time in this state
```

**EngagementLevel.msg**:
```
std_msgs/Header header
string level                  # focused/distracted/overwhelmed
float32 gaze_score           # 0.0 (away) to 1.0 (direct)
float32 posture_score        # 0.0 (slouched) to 1.0 (attentive)
float32 overall_engagement   # Combined score
```

**Intent.msg**:
```
std_msgs/Header header
string intent_type           # command/question/statement/unclear
string[] extracted_entities  # Key words/concepts
string original_text         # Transcript
float32 confidence
```

**ActivityState.msg**:
```
std_msgs/Header header
string activity_type         # shapes/colors/numbers/letters/emotions
uint8 difficulty_level       # 1=easy, 2=medium, 3=hard
string activity_content      # JSON-encoded activity data
duration time_started
```

**MotionCommand.msg**:
```
std_msgs/Header header
string command_type          # gesture/activity/calming/stop
float32 speed_factor         # 0.0 to 1.0
bool requires_safety_check   # Force validation
```

**SafetyStatus.msg**:
```
std_msgs/Header header
bool system_safe             # Overall safety state
bool emergency_stop_active
string[] active_warnings     # List of current safety concerns
float32 proximity_cm         # Child distance from robot
```

### Database Schema (SQLite)

**learner_profiles** table:
```sql
CREATE TABLE learner_profiles (
    id TEXT PRIMARY KEY,
    name_encrypted BLOB,  -- AES-256 encrypted
    age INTEGER,
    diagnosis TEXT,  -- ASD/ADHD/learning_disability
    sensory_preferences TEXT,  -- JSON: {motion: reduced, sound: quiet}
    created_at INTEGER,
    updated_at INTEGER
);
```

**sessions** table:
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    learner_id TEXT,
    start_time INTEGER,
    end_time INTEGER,
    total_activities INTEGER,
    avg_engagement_score REAL,
    emotions_json TEXT,  -- Emotion timeline
    FOREIGN KEY (learner_id) REFERENCES learner_profiles(id)
);
```

**activity_performance** table:
```sql
CREATE TABLE activity_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    activity_type TEXT,
    difficulty_level INTEGER,
    correct BOOLEAN,
    response_time_ms INTEGER,
    engagement_during REAL,
    timestamp INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

## API Design

### ROS 2 Topic Interface

**QoS Policies**:
- **SAFETY_CRITICAL**: Reliable, volatile, depth=1, deadline=100ms (safety messages)
- **REAL_TIME**: Best effort, volatile, depth=5 (perception data)
- **PERSISTENT**: Reliable, transient-local, depth=10 (session data, commands)

| Topic | Message Type | QoS | Description |
|-------|-------------|-----|-------------|
| `/perception/emotion` | EmotionState | REAL_TIME | Emotion classification |
| `/perception/engagement` | EngagementLevel | REAL_TIME | Engagement metrics |
| `/audio/transcript` | std_msgs/String | PERSISTENT | Speech transcription |
| `/audio/intent` | Intent | PERSISTENT | Parsed intent |
| `/learning/activity` | ActivityState | PERSISTENT | Current activity |
| `/learning/performance` | PerformanceMetric | PERSISTENT | Progress data |
| `/motion/command` | MotionCommand | PERSISTENT | Movement requests |
| `/safety/status` | SafetyStatus | SAFETY_CRITICAL | Safety state |
| `/session/state` | SessionState | PERSISTENT | Session status |

### Dashboard API (WebSocket + REST)

**WebSocket Events** (Server → Client):
```json
{
  "event": "emotion_update",
  "data": {"emotion": "happy", "confidence": 0.85}
}

{
  "event": "activity_complete",
  "data": {"correct": true, "new_score": 8}
}

{
  "event": "milestone_achieved",
  "data": {"milestone": "10_shapes_mastered"}
}
```

**REST Endpoints**:
- `GET /api/profiles`: List learner profiles
- `POST /api/profiles`: Create new profile
- `GET /api/sessions/:id`: Session details
- `GET /api/progress/:learnerId`: Historical progress
- `POST /api/config/preferences`: Update sensory preferences
- `DELETE /api/data/:learnerId`: Delete all child data (GDPR)

## Security Considerations

### Child Data Privacy (COPPA/GDPR)
- All personal data encrypted at rest with AES-256
- Parental consent flow before any data collection
- Data minimization: only collect essential therapeutic data
- Right to access, export, and delete all data
- Local-first: no cloud transmission without explicit opt-in
- Audit logging of all data access

### Physical Safety
- Multi-layer safety validation before any motion
- Hardware emergency stop button wired to motor controllers
- Proximity sensors prevent movement when child too close
- Force-limited actuators prevent injury
- Stress detection triggers automatic calming mode
- Watchdog timer halts system if coordinator node fails

### System Security
- No network ports exposed except dashboard (localhost only)
- WebSocket authentication with session tokens
- Input validation on all external data
- Sanitized speech transcriptions (no command injection)
- Secure boot and file system integrity checks

## Performance Considerations

### Real-time Processing
- Emotion detection: 15 FPS with <500ms latency (66ms per frame budget)
- Speech recognition: <2s for typical child utterance (3-5 words)
- Safety monitoring: 10 Hz continuous checks (100ms cycle)
- Dashboard updates: <1s from event to UI display

### Resource Optimization
- MediaPipe GPU acceleration when available (3x speedup)
- Whisper model caching (avoid reload between sessions)
- SQLite connection pooling for dashboard queries
- WebSocket message batching (reduce overhead)

### Scalability
- Single-child focus: no multi-user concurrency needed
- Database sharding by learner_id for multiple children
- Stateless dashboard server for horizontal scaling

## Error Handling

### Node-level Recovery
```python
class EmotionDetectionNode(Node):
    def camera_callback(self):
        try:
            frame = self.camera.read()
            results = self.mediapipe_holistic.process(frame)
            self.classify_emotion(results)
        except CameraDisconnectedError:
            self.get_logger().warn("Camera disconnected, using last known emotion")
            self.degrade_gracefully()  # Continue without vision
        except ModelInferenceError as e:
            self.get_logger().error(f"Emotion model failed: {e}")
            self.fallback_to_neutral_emotion()
        except Exception as e:
            self.get_logger().critical(f"Unexpected error: {e}")
            self.request_coordinator_intervention()
```

### System-level Safeguards
- **Watchdog**: Coordinator monitors node heartbeats, restarts failed nodes
- **Graceful Degradation**: System continues with reduced functionality if sensors fail
- **Safe Defaults**: On any error during child interaction, default to calming, non-moving state
- **Session Persistence**: All progress saved every 30 seconds to prevent data loss
- **Automatic Recovery**: Crashed nodes restart with exponential backoff (max 3 attempts)

### Safety Protocols
- Any safety violation immediately triggers emergency stop
- Coordinator notifies caregiver dashboard of all errors
- Critical errors (physical safety) logged to append-only audit file
- Session automatically ends if errors persist beyond threshold

This design provides a robust, safe, and therapeutically sound foundation for the adaptive learning companion robot.
