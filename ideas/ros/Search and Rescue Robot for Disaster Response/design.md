# Technical Design Document: AI-Powered Search and Rescue Robot for Disaster Response

## Architecture Overview

### System Architecture
The platform follows a disaster response robotics architecture with five specialized ROS 2 nodes orchestrated around survivor detection, autonomous navigation, and fail-safe operation in extreme environments.

```
┌─────────────────────────────────────────────────────────────────┐
│          Incident Command Dashboard (Web)                       │
│     Live Video, 3D Map, Survivor Locations, Telemetry          │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebRTC/WebSocket/LoRa
┌────────────────────────────┴────────────────────────────────────┐
│                  Mission Coordinator Node                       │
│      Multi-Robot Orchestration, Safety Monitoring, Fusion       │
└─────┬──────┬──────┬──────┬───────────────────────────────────┘
      │      │      │      │
┌─────▼──┐ ┌─▼────────┐ ┌─▼──────┐ ┌──▼─────────────────────────┐
│Survivor│ │  Audio   │ │ SLAM & │ │   Structural Assessment   │
│Detect  │ │Localize  │ │Navigate│ │   Crack Detection AI      │
└────────┘ └──────────┘ └────────┘ └────────────────────────────┘
    │          │            │              │
    ▼          ▼            ▼              ▼
[Thermal + [8-Mic     [LIDAR +    [Multi-angle
 RGB-D]      Array]    IMU]         Cameras]
```

### Key Design Principles
- **Life-Saving Priority**: All decisions optimize for survivor detection speed and accuracy
- **Fail-Safe Operation**: Multiple redundancy layers; graceful degradation in sensor failures
- **Extreme Durability**: IP67 rated, -20°C to 60°C operation, 2m drop resistance
- **Multi-Modal Fusion**: Visual + thermal + audio for high-confidence survivor detection
- **Autonomous Intelligence**: Navigate rubble without constant human control

### Technology Stack
- **ROS 2**: Humble/Iron with DDS middleware
- **AI Models**: MediaPipe, Whisper, TensorFlow custom survivor detection
- **SLAM**: RTAB-Map or Cartographer for 3D mapping
- **Communication**: WebRTC (video), LoRa (long-range command), mesh networking
- **Platform**: Ubuntu 22.04 LTS on NVIDIA Jetson (ruggedized)

## Component Design

### 1. Survivor Detection Node (`survivor_detection_node`)

**Responsibility**: Fuse thermal imaging and RGB-D camera data to detect trapped humans in rubble with high accuracy and low false positive rates.

**Class Structure**:
```python
class SurvivorDetectionNode(Node):
    def __init__(self):
        super().__init__('survivor_detection_node')
        self.thermal_camera: ThermalCamera  # FLIR or equivalent
        self.rgbd_camera: RealSenseCamera   # RGB-D for depth
        self.mediapipe_pose: mp.solutions.pose.Pose
        self.survivor_classifier: TensorFlowModel  # Custom disaster training
        self.fusion_engine: MultiModalFusion
        
    def thermal_callback(self) -> None:
        """Process thermal frame for heat signatures"""
        
    def rgbd_callback(self) -> None:
        """Process RGB-D for human pose detection"""
        
    def fuse_detections(self, thermal_hits: List, visual_hits: List) -> List[Survivor]:
        """Multi-modal fusion for high-confidence detections"""
        
    def publish_survivors(self, survivors: List[Survivor]) -> None:
        """Publish to /survivors/detected with GPS/SLAM coordinates"""
```

**Publishers**:
- `/survivors/detected` (custom_msgs/SurvivorDetection): Position, confidence, modality

**Parameters**:
- `thermal_threshold` (float): Minimum temperature delta for human detection (default: 5°C)
- `fusion_confidence_min` (float): Minimum confidence for publishing (default: 0.75)
- `detection_range_meters` (float): Maximum detection distance (default: 15.0)

**Dependencies**: MediaPipe 0.10+, TensorFlow 2.13+, pyrealsense2, thermal camera SDK

### 2. Audio Localization Node (`audio_localization_node`)

**Responsibility**: Detect and localize survivor voices, breathing sounds, and tapping patterns using 8-microphone array with beamforming.

**Class Structure**:
```python
class AudioLocalizationNode(Node):
    def __init__(self):
        super().__init__('audio_localization_node')
        self.mic_array: MicrophoneArray  # 8-channel circular array
        self.whisper_model: whisper.Whisper
        self.beamformer: BeamformingProcessor
        self.doa_calculator: DirectionOfArrival
        self.sound_classifier: SoundTypeClassifier
        self.two_way_comm: TwoWayAudioSystem
        
    def audio_callback(self, audio_data: np.ndarray) -> None:
        """Process 8-channel audio input"""
        
    def detect_survivor_sounds(self, audio: np.ndarray) -> List[AudioDetection]:
        """Whisper + classifier for voice/breathing/tapping"""
        
    def localize_sound_source(self, detection: AudioDetection) -> Position3D:
        """Calculate DoA and estimate 3D position"""
        
    def communicate_with_survivor(self, message: str) -> None:
        """Two-way audio for rescuer-survivor communication"""
```

**Publishers**:
- `/survivors/audio` (custom_msgs/AudioAlert): Direction, sound type, confidence

**Parameters**:
- `whisper_model_size` (str): Model size (base/small, default: base)
- `beamforming_angle_resolution` (float): Angular resolution in degrees (default: 5.0)
- `min_snr_db` (float): Minimum signal-to-noise ratio (default: 10.0)

**Dependencies**: OpenAI Whisper, sounddevice, scipy (beamforming), NumPy

### 3. SLAM & Navigation Node (`slam_navigation_node`)

**Responsibility**: Generate 3D map, localize robot, plan paths through rubble, and execute autonomous navigation with stability monitoring.

**Class Structure**:
```python
class SLAMNavigationNode(Node):
    def __init__(self):
        super().__init__('slam_navigation_node')
        self.lidar: VelodyneLidar
        self.slam_engine: RTABMap  # or Cartographer
        self.imu: IMUSensor
        self.path_planner: RubblePathPlanner
        self.motion_controller: MotionController
        self.traversability_analyzer: TerrainAnalyzer
        
    def slam_callback(self) -> None:
        """Update 3D map and robot pose"""
        
    def plan_path(self, goal: Position3D) -> Path:
        """A* path planning avoiding unstable terrain"""
        
    def assess_traversability(self, terrain: OccupancyGrid) -> TraversabilityMap:
        """Classify surfaces: stable/caution/dangerous"""
        
    def execute_navigation(self, path: Path) -> None:
        """Motion commands with stability monitoring"""
```

**Subscribers**:
- `/survivors/detected`, `/survivors/audio`: Navigate to survivor locations

**Publishers**:
- `/navigation/map` (nav_msgs/OccupancyGrid): 3D SLAM map
- `/robot/pose` (geometry_msgs/PoseStamped): Current position

**Parameters**:
- `slam_algorithm` (str): rtabmap/cartographer (default: rtabmap)
- `max_climb_angle_deg` (float): Maximum climbable slope (default: 45.0)
- `stability_threshold` (float): IMU threshold for tip-over warning (default: 0.7)

**Dependencies**: ROS 2 Navigation2, RTAB-Map, LIDAR drivers, IMU integration

### 4. Structural Assessment Node (`structural_assessment_node`)

**Responsibility**: Analyze building damage through computer vision crack detection and risk scoring to inform rescuer safety decisions.

**Class Structure**:
```python
class StructuralAssessmentNode(Node):
    def __init__(self):
        super().__init__('structural_assessment_node')
        self.inspection_cameras: List[Camera]  # Multi-angle
        self.crack_detector: CrackDetectionCNN
        self.risk_scorer: StructuralRiskAnalyzer
        self.change_detector: ChangeDetectionSystem
        
    def inspect_structure(self, images: List[np.ndarray]) -> StructuralAssessment:
        """Multi-view crack detection and damage analysis"""
        
    def calculate_risk_score(self, damage: DamageAnalysis) -> RiskLevel:
        """Scoring: safe/caution/danger based on crack patterns"""
        
    def monitor_changes(self, baseline: StructuralAssessment) -> List[Change]:
        """Detect new cracks or shifts during rescue operations"""
```

**Publishers**:
- `/structure/assessment` (custom_msgs/StructuralRisk): Risk level, crack locations

**Parameters**:
- `crack_width_threshold_mm` (float): Minimum crack width to report (default: 2.0)
- `risk_threshold_high` (float): Threshold for danger classification (default: 0.7)

**Dependencies**: TensorFlow (crack detection model), OpenCV, structure-from-motion libraries

### 5. Mission Coordinator Node (`mission_coordinator_node`)

**Responsibility**: Orchestrate all subsystems, fuse multi-modal detections, manage mission state, communicate with incident command.

**Class Structure**:
```python
class MissionCoordinatorNode(Node):
    def __init__(self):
        super().__init__('mission_coordinator_node')
        self.mission_state: MissionStateMachine
        self.detection_fusion: MultiModalFusionEngine
        self.webrtc_server: WebRTCStreamingServer
        self.lora_radio: LoRaRadio  # Long-range command link
        self.mesh_network: MeshNetworkManager
        self.safety_monitor: SafetyMonitoringSystem
        
    def fuse_detections(self) -> List[ConfirmedSurvivor]:
        """Combine visual, thermal, audio for high-confidence survivors"""
        
    def coordinate_multi_robot(self, robots: List[RobotStatus]) -> None:
        """Mesh networking and area division"""
        
    def stream_to_command(self) -> None:
        """WebRTC video + WebSocket telemetry to dashboard"""
```

**Subscribers**: All other node topics for fusion and monitoring

**Publishers**:
- `/mission/status` (custom_msgs/MissionStatus): Current mission state
- `/robot/emergency_stop` (std_msgs/Bool): Emergency halt

**Services**:
- `/mission/start` (std_srvs/Trigger): Begin search operation
- `/mission/mark_survivor` (custom_srvs/MarkSurvivor): Operator override

**Parameters**:
- `webrtc_port` (int): Streaming server port (default: 8443)
- `lora_frequency_mhz` (float): LoRa radio frequency (default: 915.0)

## Data Models

### Custom ROS 2 Messages

**SurvivorDetection.msg**:
```
std_msgs/Header header
geometry_msgs/Point position      # 3D position (GPS or SLAM coordinates)
string detection_modality         # visual/thermal/audio/multi-modal
float32 confidence                # 0.0 to 1.0
string survivor_id                # Unique identifier
bool vital_signs_detected         # Movement/breathing detected
```

**AudioAlert.msg**:
```
std_msgs/Header header
geometry_msgs/Vector3 direction   # DoA vector
string sound_type                 # voice/breathing/tapping/rubble_shift
float32 urgency_score            # 0.0 (calm) to 1.0 (distress)
string transcription             # Whisper transcript if voice
```

**StructuralRisk.msg**:
```
std_msgs/Header header
string risk_level                # safe/caution/danger
geometry_msgs/Point[] crack_locations
float32 collapse_probability     # 0.0 to 1.0
string[] recommendations         # Actions for rescuers
```

## API Design

### ROS 2 Topic Interface

**QoS Policies**:
- **LIFE_CRITICAL**: Reliable, volatile, depth=1, deadline=500ms (survivor detections, safety)
- **MAPPING**: Best effort, transient-local, depth=10 (SLAM maps)
- **TELEMETRY**: Best effort, volatile, depth=5 (status, diagnostics)

### Dashboard API (WebRTC + WebSocket)

**WebSocket Events** (Robot → Dashboard):
```json
{
  "event": "survivor_detected",
  "data": {
    "position": {"x": 10.5, "y": -3.2, "z": 1.8},
    "confidence": 0.92,
    "modality": "multi-modal",
    "timestamp": "2025-01-15T14:32:10Z"
  }
}
```

**REST Endpoints**:
- `GET /api/mission/status`: Current mission state
- `POST /api/mission/waypoint`: Set navigation goal
- `POST /api/emergency_stop`: Halt robot immediately

## Security Considerations

### Communication Security
- AES-256 encryption for video/data over LoRa
- Certificate-based authentication for WebRTC
- Mesh network with node verification

### Physical Security
- Emergency stop: remote command, voice, physical button
- Tip-over detection with auto-shutdown
- Battery thermal management

## Performance Considerations

### Real-Time Processing
- Survivor detection: <200ms latency (target: 100ms)
- Audio localization: <500ms for responsive communication
- Navigation control: 10 Hz minimum for stability
- Video streaming: 15-30 FPS with <500ms end-to-end

### Resource Optimization
- GPU acceleration for MediaPipe and TensorFlow (NVIDIA Jetson)
- Model quantization for Whisper (INT8) to reduce memory
- Adaptive video compression based on available bandwidth

## Error Handling

### Node-Level Recovery
```python
class SurvivorDetectionNode(Node):
    def thermal_callback(self):
        try:
            frame = self.thermal_camera.read()
            detections = self.process_thermal(frame)
        except CameraDisconnectedError:
            self.get_logger().warn("Thermal camera lost, using RGB-D only")
            self.degrade_to_visual_only()
        except Exception as e:
            self.get_logger().error(f"Detection failed: {e}")
            self.request_coordinator_intervention()
```

### Mission-Level Safeguards
- **Watchdog**: Coordinator monitors node heartbeats, restarts failures
- **Sensor Fusion Fallback**: Continue with available modalities if sensors fail
- **Communication Loss**: Switch to autonomous search pattern, periodic check-ins
- **Low Battery**: Auto-return to base below 15% charge
- **Tip-Over**: Emergency stop, alert operators, wait for manual recovery

This design provides a robust, life-saving foundation for the search and rescue robot system.
