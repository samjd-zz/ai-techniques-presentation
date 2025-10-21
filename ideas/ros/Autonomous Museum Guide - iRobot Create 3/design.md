# Technical Design Document: Autonomous Art Gallery Guide & Cultural Curator Robot

## Architecture Overview

### System Architecture
The platform follows a cultural experience robotics architecture leveraging the iRobot Create® 3 platform with ROS 2, integrating autonomous navigation, AI-powered artwork recognition, interactive storytelling, and visitor engagement detection.

```
┌─────────────────────────────────────────────────────────────────┐
│       Gallery Staff Dashboard + Visitor Mobile App              │
│   Analytics, Content Management, Tour Scheduling, Bookings     │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API / WebSocket
┌────────────────────────────┴────────────────────────────────────┐
│              Tour Coordinator Node                              │
│    Visitor Management, Route Planning, Queue Handling           │
└─────┬──────┬──────┬──────┬──────────────────────────────────────┘
      │      │      │      │
┌─────▼──┐ ┌─▼────────┐ ┌──▼──────┐ ┌──▼──────────────────────┐
│Artwork │ │Storytell │ │Navigate │ │  Engagement Detection  │
│Recogn. │ │RAG+LLM   │ │Nav2     │ │  MediaPipe Emotions    │
└────────┘ └──────────┘ └─────────┘ └────────────────────────────┘
    │          │            │              │
    ▼          ▼            ▼              ▼
[Camera]   [Mic Array] [Create® 3]    [Tablet Display]
           [Speaker]    [Sensors]      [Period Music]
```

### Key Design Principles
- **iRobot Create® 3 Native**: Leverage platform's sensors and safety features
- **Visitor-Centric**: All decisions optimize for engagement and learning
- **Cultural Authenticity**: Curator-validated content with proper attribution
- **Adaptive Experience**: Real-time adjustment based on visitor engagement
- **Multilingual by Default**: Support 10+ languages without complexity

### Technology Stack
- **Platform**: iRobot Create® 3 + Raspberry Pi 4 (8GB) or Jetson Nano
- **Framework**: ROS 2 Humble with Navigation2
- **AI**: CLIP (artwork recognition), Whisper, MediaPipe, LangChain, LLM (GPT-4/Mistral)
- **Storage**: PostgreSQL (artwork metadata, visitor profiles)
- **Interface**: React (staff dashboard), Flask (REST API)

## Component Design

### 1. Gallery Navigation Node (`gallery_navigation_node`)

**Responsibility**: Autonomous navigation through gallery spaces using iRobot Create® 3 and Navigation2, with safety zones around artworks and crowd-aware routing.

**Class Structure**:
```python
class GalleryNavigationNode(Node):
    def __init__(self):
        super().__init__('gallery_navigation_node')
        self.create3_interface: Create3Interface
        self.nav2_client: Nav2Client
        self.gallery_map: OccupancyGrid
        self.artwork_locations: Dict[str, Pose]
        self.safety_zones: List[SafetyZone]  # 1m around artworks
        self.crowd_detector: CrowdDetector
        
    def navigate_to_artwork(self, artwork_id: str) -> None:
        """Navigate to specific artwork with safety checks"""
        
    def enforce_safety_zones(self, position: Pose) -> bool:
        """Verify robot maintains >1m from artworks"""
        
    def detect_and_avoid_crowds(self) -> None:
        """Dynamic rerouting if pathway is crowded"""
```

**Publishers**:
- `/robot/location` (geometry_msgs/PoseStamped): Current position
- `/robot/battery` (sensor_msgs/BatteryState): Battery level

**Parameters**:
- `artwork_clearance_m` (float): Minimum distance from artworks (default: 1.0)
- `visitor_clearance_m` (float): Minimum distance from visitors (default: 0.5)
- `max_speed_mps` (float): Maximum speed in gallery (default: 0.3)

**Dependencies**: iRobot Create® 3 ROS 2 API, Navigation2, gallery map YAML

### 2. Artwork Recognition Node (`artwork_recognition_node`)

**Responsibility**: Identify artworks using CLIP vision model, retrieve metadata from museum database, position robot optimally for visitor viewing.

**Class Structure**:
```python
class ArtworkRecognitionNode(Node):
    def __init__(self):
        super().__init__('artwork_recognition_node')
        self.camera: USBCamera
        self.clip_model: CLIPModel  # OpenAI CLIP
        self.artwork_db: ArtworkDatabase
        self.positioning_optimizer: ViewingPositionOptimizer
        
    def recognize_artwork(self, image: np.ndarray) -> Optional[Artwork]:
        """CLIP-based zero-shot artwork identification"""
        
    def retrieve_metadata(self, artwork_id: str) -> ArtworkMetadata:
        """Database lookup for artist, period, provenance, etc."""
        
    def optimize_viewing_position(self, artwork: Artwork) -> Pose:
        """Calculate best robot position for visitor viewing"""
```

**Publishers**:
- `/artwork/identified` (custom_msgs/ArtworkInfo): Recognized artwork details

**Parameters**:
- `clip_model_version` (str): Model version (ViT-B/32, default)
- `recognition_threshold` (float): Minimum confidence (default: 0.9)
- `fallback_qr_enabled` (bool): Use QR codes if vision fails (default: true)

**Dependencies**: CLIP (transformers library), PostgreSQL, OpenCV

### 3. Interactive Storytelling Node (`interactive_storytelling_node`)

**Responsibility**: Generate engaging narratives about artworks using RAG and LLM, handle voice Q&A, synchronize with tablet display and period music.

**Class Structure**:
```python
class InteractiveStorytellingNode(Node):
    def __init__(self):
        super().__init__('interactive_storytelling_node')
        self.microphone: MicrophoneArray
        self.whisper_model: whisper.Whisper
        self.rag_pipeline: LangChainRAG  # Art history knowledge base
        self.llm: LLMClient  # GPT-4 or Mistral 7B
        self.tts_engine: TTS
        self.tablet_display: TabletInterface
        self.music_player: PeriodMusicPlayer
        
    def transcribe_question(self, audio: np.ndarray) -> str:
        """Whisper STT for visitor questions"""
        
    def generate_narrative(self, artwork: Artwork, complexity: str) -> Story:
        """LLM creates engaging story with RAG context"""
        
    def present_story(self, story: Story) -> None:
        """Synchronized voice + visuals + music"""
```

**Subscribers**:
- `/artwork/identified` (custom_msgs/ArtworkInfo): Current artwork

**Publishers**:
- `/story/current` (custom_msgs/Story): Current presentation

**Parameters**:
- `whisper_model_size` (str): base/small (default: base)
- `llm_backend` (str): openai/local (default: openai)
- `complexity_levels` (list): casual, enthusiast, expert
- `presentation_length_minutes` (float): Target length (default: 3.0)

**Dependencies**: Whisper, LangChain, ChromaDB, OpenAI API or local LLM

### 4. Engagement Detection Node (`engagement_detection_node`)

**Responsibility**: Analyze visitor facial expressions and body language to detect engagement levels, adapt presentations in real-time.

**Class Structure**:
```python
class EngagementDetectionNode(Node):
    def __init__(self):
        super().__init__('engagement_detection_node')
        self.camera: USBCamera
        self.mediapipe_face: mp.solutions.face_mesh.FaceMesh
        self.emotion_classifier: EmotionClassifier  # Custom TF model
        self.dwell_timer: DwellTimeTracker
        self.adaptive_controller: AdaptiveContentController
        
    def detect_emotions(self, frame: np.ndarray) -> EmotionState:
        """MediaPipe landmarks → emotion classification"""
        
    def calculate_engagement_score(self, emotions: EmotionState, 
                                   dwell_time: float) -> float:
        """Score 0-1: fascinated/neutral/bored/confused"""
        
    def adapt_presentation(self, score: float) -> ContentAdjustment:
        """Decide to simplify/elaborate/skip/continue"""
```

**Publishers**:
- `/visitor/engagement` (custom_msgs/EngagementScore): Real-time engagement

**Parameters**:
- `emotion_detection_threshold` (float): Confidence threshold (default: 0.7)
- `engagement_window_seconds` (int): Moving average window (default: 10)
- `adaptation_trigger` (str): immediate/after_3_low_scores

**Dependencies**: MediaPipe, TensorFlow (custom emotion model)

### 5. Tour Planning Node (`tour_planning_node`)

**Responsibility**: Create personalized tour routes based on visitor preferences, time constraints, and detected interests; optimize gallery traversal.

**Class Structure**:
```python
class TourPlanningNode(Node):
    def __init__(self):
        super().__init__('tour_planning_node')
        self.gallery_graph: GalleryGraph
        self.route_optimizer: RouteOptimizer  # TSP solver
        self.visitor_profiler: VisitorProfiler
        self.queue_manager: TourQueueManager
        
    def create_tour_route(self, preferences: VisitorPreferences) -> TourRoute:
        """Optimize route based on time, interests, gallery layout"""
        
    def dynamically_adjust_route(self, current_position: Pose,
                                 time_remaining: float) -> TourRoute:
        """Re-plan if visitor lingers or skips artworks"""
        
    def manage_multi_visitor_queue(self) -> List[ScheduledTour]:
        """Schedule and coordinate multiple concurrent tours"""
```

**Subscribers**:
- `/visitor/engagement` (custom_msgs/EngagementScore): For interest detection

**Publishers**:
- `/tour/route` (custom_msgs/TourRoute): Planned tour waypoints
- `/tour/status` (custom_msgs/TourStatus): Current progress

**Parameters**:
- `default_tour_minutes` (int): Standard tour length (default: 60)
- `artworks_per_hour` (int): Presentation rate (default: 8)
- `optimization_algorithm` (str): greedy/genetic/simulated_annealing

**Dependencies**: NetworkX (graph algorithms), OR-Tools (optimization)

## Data Models

### PostgreSQL Database Schema

**artworks** table:
```sql
CREATE TABLE artworks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(255),
    creation_year INT,
    period VARCHAR(100),  -- Renaissance, Impressionism, Modern, etc.
    medium VARCHAR(100),  -- Oil on canvas, Bronze sculpture, etc.
    dimensions VARCHAR(100),
    provenance TEXT,
    current_location VARCHAR(100),  -- Gallery room identifier
    position_x FLOAT,  -- Coordinates for navigation
    position_y FLOAT,
    clip_embedding VECTOR(512),  -- For artwork recognition
    image_url VARCHAR(500),
    qr_code VARCHAR(100),  -- Fallback identifier
    curator_notes TEXT
);
```

**artwork_content** table:
```sql
CREATE TABLE artwork_content (
    id SERIAL PRIMARY KEY,
    artwork_id INT REFERENCES artworks(id),
    language VARCHAR(10),  -- en, fr, es, etc.
    complexity_level VARCHAR(20),  -- casual, enthusiast, expert
    story_text TEXT,
    artist_biography TEXT,
    historical_context TEXT,
    symbolism_analysis TEXT,
    fun_facts TEXT[],
    related_artworks INT[],
    period_music_url VARCHAR(500)
);
```

**visitor_tours** table:
```sql
CREATE TABLE visitor_tours (
    id SERIAL PRIMARY KEY,
    visitor_id VARCHAR(50),
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    preferences JSONB,  -- {periods: [...], artists: [...], time: 60}
    route JSONB,  -- [artwork_ids in order]
    completion_status VARCHAR(20),  -- completed, abandoned, in_progress
    satisfaction_rating INT,
    feedback TEXT
);
```

**engagement_analytics** table:
```sql
CREATE TABLE engagement_analytics (
    id SERIAL PRIMARY KEY,
    artwork_id INT REFERENCES artworks(id),
    visitor_id VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW(),
    engagement_score FLOAT,
    dwell_time_seconds INT,
    questions_asked TEXT[],
    emotion_detected VARCHAR(50)
);
```

### Custom ROS 2 Messages

**ArtworkInfo.msg**:
```
std_msgs/Header header
string artwork_id
string title
string artist
int32 creation_year
string period
geometry_msgs/Pose location
float32 recognition_confidence
string[] related_artwork_ids
```

**EngagementScore.msg**:
```
std_msgs/Header header
string emotion  # fascinated/engaged/neutral/confused/bored
float32 score  # 0.0 to 1.0
float32 dwell_time_seconds
string suggested_action  # continue/elaborate/simplify/skip
```

**TourRoute.msg**:
```
std_msgs/Header header
string[] artwork_ids
geometry_msgs/Pose[] waypoints
duration estimated_duration
string visitor_preferences
int32 current_artwork_index
```

## API Design

### REST API (Flask)

**Endpoints**:
- `POST /api/tour/start`: Begin new tour
  ```json
  {
    "visitor_id": "optional",
    "preferences": {
      "periods": ["impressionism", "modern"],
      "artists": ["Monet", "Picasso"],
      "time_minutes": 60,
      "complexity": "casual",
      "language": "en"
    }
  }
  ```
- `GET /api/tour/{id}/status`: Check tour progress
- `POST /api/artwork/add`: Curator adds new artwork
- `GET /api/analytics/engagement`: View engagement data
- `GET /api/analytics/popular`: Most popular artworks

### ROS 2 Topics

**QoS Policies**:
- **VISITOR_INTERACTION**: Reliable, volatile, depth=10
- **NAVIGATION**: Best effort, volatile, depth=5
- **ARTWORK_DATA**: Reliable, transient-local, depth=1

## Security Considerations

### Visitor Privacy
- Facial images processed in real-time, not stored
- Visitor IDs anonymized (UUID), no PII collection
- Engagement data aggregated for analytics
- GDPR/privacy law compliant

### Content Integrity
- Curator authentication for content updates
- Version control for artwork metadata
- Audit logging of all content changes
- Backup and recovery procedures

### Physical Safety
- iRobot Create® 3 safety features (cliff detection, bumpers)
- 1m safety zones around all artworks (enforced via virtual barriers)
- Emergency stop via staff remote control
- Automatic shutdown if safety zone violated

## Performance Considerations

### Real-Time Requirements
- Artwork recognition: <3 seconds
- Voice response: <5 seconds from question to answer start
- Navigation planning: <10 seconds for full gallery route
- Engagement detection: Real-time (30 FPS)
- Tablet display: 30+ FPS for smooth visuals

### Resource Optimization
- CLIP model inference optimized with TensorRT (Jetson) or ONNX Runtime (Pi)
- LLM responses cached for common questions
- Period music pre-loaded in memory
- Route optimization pre-computed for popular preferences

## Error Handling

### Node-Level Recovery
```python
class ArtworkRecognitionNode(Node):
    def recognize_artwork(self, image):
        try:
            embedding = self.clip_model.encode_image(image)
            artwork = self.artwork_db.find_similar(embedding)
            if artwork.confidence < 0.9:
                # Fallback to QR code
                qr_code = self.detect_qr_code(image)
                artwork = self.artwork_db.get_by_qr(qr_code)
            return artwork
        except CLIPModelError:
            self.get_logger().error("CLIP model failure, using QR fallback")
            return self.fallback_qr_recognition(image)
        except DatabaseError:
            self.get_logger().error("Database connection lost")
            return self.use_cached_artwork_data()
```

### System-Level Safeguards
- **Graceful Degradation**: Continue tours with basic features if AI models fail
- **Queue Management**: Inform waiting visitors if robot needs maintenance
- **Battery Management**: Auto-return to charging below 20%, resume tours after charging
- **Content Fallback**: Pre-loaded content if RAG/LLM unavailable

This design provides a robust, engaging, culturally authentic foundation for the museum guide robot.
