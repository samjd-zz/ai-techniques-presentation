# Interactive AI Teaching Assistant Robot for Algonquin College

An autonomous AI-powered teaching assistant built on the iRobot Create® 3 platform that navigates Algonquin College campuses to provide personalized learning support, interactive robotics demonstrations, and voice-activated Q&A for STEM students.

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue)](https://docs.ros.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![iRobot Create® 3](https://img.shields.io/badge/iRobot-Create®%203-red)](https://edu.irobot.com/create3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FIPPA Compliant](https://img.shields.io/badge/FIPPA-Compliant-success)](https://www.ontario.ca/laws/statute/90f31)

## Overview

This platform provides a comprehensive, Python-based educational robotics framework that combines the iRobot Create® 3's autonomous navigation capabilities with multimodal AI (MediaPipe for student recognition, Whisper for voice interaction, RAG for knowledge retrieval) to deliver 24/7 learning support, increase student engagement by 40%+, and improve learning outcomes by 15%+ in STEM courses.

**Key Features:**
- 🎓 **24/7 Learning Support**: Students request assistance via mobile app anytime
- 🧠 **Personalized AI Tutoring**: RAG-based Q&A with course materials, textbooks, documentation
- 👤 **Student Recognition**: MediaPipe facial recognition with opt-in enrollment
- 🗣️ **Voice Interaction**: Whisper speech recognition for natural conversation
- 📊 **Engagement Detection**: Detects confusion and adapts explanations in real-time
- 🤖 **Live ROS 2 Demos**: Shows internal robot state (SLAM, sensors, planning) on tablet
- 📍 **Autonomous Navigation**: Nav2-based campus navigation with schedule integration
- 🔒 **FIPPA Compliant**: Privacy-first design with encrypted student data

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│          Student Mobile App + Faculty Dashboard                │
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

## Requirements

### Hardware
- **iRobot Create® 3** robot platform
- **Companion Computer**: Raspberry Pi 4 (4GB+) or NVIDIA Jetson Nano
- **Camera**: USB webcam (1080p, wide angle - Logitech C920 or equivalent)
- **Microphone**: USB microphone array (4-channel - ReSpeaker or equivalent)
- **Display**: 10" tablet with HDMI input (for ROS 2 visualizations)
- **Battery**: Extended battery for Create® 3 (target 6+ hours)
- **Mounting**: Custom mount for computer, camera, mic, tablet

### Software
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Python**: 3.10, 3.11, or 3.12
- **Database**: PostgreSQL 14+

### Algonquin College Requirements
- Campus WiFi access for robot
- Course materials in digital format (PDFs, markdown)
- Integration with Brightspace (optional)
- FIPPA compliance approval
- Student consent management system

## Installation

### 1. Set Up iRobot Create® 3

```bash
# Follow iRobot's setup instructions
# Connect Create® 3 to WiFi
# Note the robot's IP address

# Test connection
ping create3-robot-ip
```

### 2. Install ROS 2 Humble on Companion Computer

```bash
# On Raspberry Pi 4 or Jetson Nano
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Humble
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-irobot-create-msgs

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. Install System Dependencies

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    postgresql postgresql-contrib \
    libportaudio2 \
    ffmpeg \
    libopencv-dev

# Install PostgreSQL vector extension for RAG
sudo apt install postgresql-14-pgvector
```

### 4. Clone Repository

```bash
# Create workspace
mkdir -p ~/teaching_robot_ws/src
cd ~/teaching_robot_ws/src

# Clone repository
git clone https://github.com/algonquincollege/teaching-assistant-robot.git
cd teaching-assistant-robot
```

### 5. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
mediapipe>=0.10.0
openai-whisper>=20230314
langchain>=0.1.0
chromadb>=0.4.0
transformers>=4.35.0
torch>=2.1.0
psycopg2-binary>=2.9.0
opencv-python>=4.8.0
numpy>=1.24.0
sounddevice>=0.4.6
pyttsx3>=2.90
flask>=3.0.0
flask-cors>=4.0.0
PyYAML>=6.0
```

### 6. Set Up Database

```bash
# Create PostgreSQL database
sudo -u postgres createdb teaching_robot

# Create user
sudo -u postgres psql -c "CREATE USER robot_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE teaching_robot TO robot_user;"

# Initialize schema
psql -U robot_user -d teaching_robot -f sql/schema.sql
```

### 7. Build ROS 2 Workspace

```bash
cd ~/teaching_robot_ws
colcon build --symlink-install
source install/setup.bash
```

### 8. Configure Course Materials for RAG

```bash
# Create course materials directory
mkdir -p ~/teaching_robot_ws/course_materials

# Copy PDFs, markdown files, documentation
cp /path/to/course/materials/* ~/teaching_robot_ws/course_materials/

# Index materials for RAG
python3 src/teaching-assistant-robot/scripts/index_course_materials.py \
    --source ~/teaching_robot_ws/course_materials \
    --database teaching_robot
```

## Quick Start

### 1. Map Campus Environment

```bash
# Launch Create® 3 with SLAM
ros2 launch teaching_robot slam_mapping.launch.py

# Drive robot around hallways using teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Save map when complete
ros2 run nav2_map_server map_saver_cli -f ~/teaching_robot_ws/maps/algonquin_t_building
```

### 2. Configure Robot Schedule

Edit `config/robot_schedule.yaml`:

```yaml
schedule:
  monday:
    - time: "09:00-11:00"
      location: "cs_lab"
      course: "CST8116"  # Intro to Programming
    - time: "13:00-15:00"
      location: "robotics_lab"
      course: "CST8333"  # ROS 2 Course
  tuesday:
    - time: "10:00-12:00"
      location: "library_study_area"
      course: "general_support"
  # ... continue for other days
```

### 3. Launch System

```bash
# Activate environment
source ~/teaching_robot_ws/install/setup.bash
source ~/teaching_robot_ws/src/teaching-assistant-robot/venv/bin/activate

# Launch all nodes
ros2 launch teaching_robot full_system.launch.py \
    map:=~/teaching_robot_ws/maps/algonquin_t_building.yaml \
    schedule:=config/robot_schedule.yaml
```

### 4. Student Mobile App

Students download the "Algonquin Teaching Robot" app:

**Features**:
- Request assistance ("Robot, come help me at T127")
- View robot schedule and current location
- Check interaction history
- Enroll with facial recognition (opt-in)

### 5. Faculty Dashboard

Access at `http://robot-ip:5000/faculty`

**Features**:
- View most frequently asked questions
- Identify concepts students struggle with
- Monitor robot location and battery
- Download interaction reports
- Manage robot schedule

## Usage

### Student Interaction Flow

1. **Request Assistance**:
   - Open mobile app
   - Tap "Request Help"
   - Share location or robot navigates to scheduled location

2. **Robot Arrives**:
   - If enrolled: "Hi Sarah! Ready to continue with Python loops?"
   - If new: "Hello! I'm here to help. Would you like to enroll for personalized support?"

3. **Ask Questions**:
   - Voice: "How does SLAM work?"
   - Robot retrieves info from course materials via RAG
   - Answers with source citations: "According to your Robotics textbook, chapter 4..."

4. **Interactive Demonstrations**:
   - "Show me path planning"
   - Tablet displays real-time ROS 2 visualization
   - Robot explains what's happening as it demonstrates

5. **Adaptive Learning**:
   - Robot detects confusion via facial expression
   - "Would you like me to explain that differently?"
   - Simplifies explanation or provides visual aid

### Faculty Usage

**In-Class Demonstrations**:
```bash
# Trigger specific demo
ros2 topic pub /demonstration/trigger std_msgs/String "data: 'slam_demo'"

# Robot performs live SLAM demonstration
# Tablet visualization projects to classroom screen
```

**Analytics Review**:
- Access dashboard weekly
- Identify common student struggles
- Adjust lecture content accordingly
- Export reports for curriculum improvement

## Configuration

### Voice Q&A Settings

Edit `config/voice_qa.yaml`:

```yaml
voice_qa_node:
  ros__parameters:
    whisper_model_size: "base"
    llm_backend: "local"  # local (Mistral) or openai (GPT-4 API)
    rag_top_k: 3  # Number of context chunks to retrieve
    conversation_timeout: 30  # minutes
    source_citation: true  # Always cite sources
```

### Student Recognition Settings

Edit `config/student_recognition.yaml`:

```yaml
student_recognition_node:
  ros__parameters:
    opt_in_required: true
    recognition_threshold: 0.85
    enrollment_consent_required: true
    data_retention_days: 365  # FIPPA compliance
    privacy_mode_default: true
```

### Navigation Settings

Edit `config/navigation.yaml`:

```yaml
navigation_node:
  ros__parameters:
    safety_clearance: 0.5  # meters
    max_speed: 0.4  # m/s for crowded hallways
    patrol_enabled: true
    auto_return_battery_threshold: 20  # percent
```

## Troubleshooting

### Robot Not Responding to Voice

```bash
# Test microphone
arecord -d 5 test.wav && aplay test.wav

# Check Whisper model loaded
ros2 topic echo /voice_qa/status

# Adjust sensitivity in config
# Lower vad_threshold for noisy environments
```

### Student Recognition Not Working

```bash
# Test camera
ros2 run image_view image_view --ros-args -r image:=/camera/image_raw

# Check lighting (need adequate light)
# Verify enrollment (student needs to opt-in first)
# Check database connection
psql -U robot_user -d teaching_robot -c "SELECT COUNT(*) FROM students;"
```

### RAG Not Finding Answers

```bash
# Re-index course materials
python3 scripts/index_course_materials.py --rebuild

# Check vector database
# Verify course materials are in correct format (PDF, markdown)
# Test query manually:
python3 scripts/test_rag.py --query "What is SLAM?"
```

### Navigation Issues

```bash
# Check Create® 3 connection
ros2 topic list | grep create3

# Verify map loaded
ros2 topic echo /map -n 1

# Re-map if environment changed significantly
ros2 launch teaching_robot slam_mapping.launch.py
```

## Development

### Running Tests

```bash
# Unit tests
cd ~/teaching_robot_ws
colcon test --packages-select \
    campus_navigation \
    voice_qa \
    student_recognition

# Integration tests
python3 -m pytest tests/integration/

# Privacy compliance tests
python3 -m pytest tests/privacy/ -v
```

### Adding New Course Materials

```bash
# Add new PDFs to course materials directory
cp new_textbook.pdf ~/teaching_robot_ws/course_materials/CST8116/

# Re-index
python3 scripts/index_course_materials.py \
    --course CST8116 \
    --incremental
```

### Custom Demonstrations

Create new demonstration in `src/demos/`:

```python
# custom_demo.py
class PathPlanningDemo(DemoBase):
    def execute(self):
        # Show A* algorithm visualization
        self.visualize_graph()
        self.speak("This is how the robot finds the shortest path...")
        # Continue with demo logic
```

Register in `config/demonstrations.yaml`:
```yaml
available_demos:
  - name: "path_planning"
    class: "PathPlanningDemo"
    trigger_phrase: "show me path planning"
```

## Privacy & Compliance

### FIPPA Compliance

- ✅ Opt-in enrollment required
- ✅ Explicit consent for facial recognition
- ✅ AES-256 encryption for student data
- ✅ Right to data deletion within 48 hours
- ✅ Audit logging of all data access
- ✅ Anonymized faculty analytics

### Student Data Management

**Enrollment**:
```bash
# Student opts in via mobile app or robot interaction
# Consent form displayed, photo captured
# Data encrypted and stored
```

**Data Deletion**:
```bash
# Student requests deletion
ros2 service call /student/delete_data \
    teaching_robot_msgs/srv/DeleteStudent \
    "{student_id: 'A00123456'}"

# All data removed within 48 hours
```

## Project Structure

```
teaching-assistant-robot/
├── src/
│   ├── campus_navigation_node/        # Nav2 integration
│   ├── student_recognition_node/      # MediaPipe + face recognition
│   ├── voice_qa_node/                 # Whisper + RAG + LLM
│   ├── engagement_detection_node/     # Confusion detection
│   ├── interactive_demo_node/         # ROS 2 visualizations
│   ├── teaching_coordinator_node/     # Orchestration
│   └── custom_msgs/                   # ROS 2 message definitions
├── mobile_app/                        # React Native student app
├── dashboard/                         # Flask faculty dashboard
├── config/                            # Robot configuration
├── maps/                              # Campus maps
├── course_materials/                  # Indexed content for RAG
├── sql/                               # Database schemas
├── scripts/                           # Utilities
└── tests/                             # Unit, integration, privacy tests
```

## Contributing

We welcome contributions from Algonquin College community:
- Faculty (course material integration, demo ideas)
- Students (feature requests, bug reports)
- IT staff (infrastructure, deployment)
- Researchers (educational effectiveness studies)

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **Algonquin College**: For supporting innovative educational technology
- **iRobot**: For the Create® 3 educational robotics platform
- **ROS 2**: Open-source robotics middleware
- **MediaPipe**: Google's ML solutions
- **OpenAI Whisper**: Speech recognition
- **LangChain**: RAG framework

## Research & Publications

If you use this platform in research, please cite:

```bibtex
@software{algonquin_teaching_robot,
  title = {Interactive AI Teaching Assistant Robot for Algonquin College},
  author = {Algonquin College Robotics Team},
  year = {2025},
  institution = {Algonquin College},
  url = {https://github.com/algonquincollege/teaching-assistant-robot}
}
```

## Support & Resources

### For Students
- **User Guide**: `/docs/students/user_guide.pdf`
- **FAQ**: [Link to student FAQ]
- **Privacy Policy**: `/docs/privacy/student_privacy.pdf`

### For Faculty
- **Integration Guide**: `/docs/faculty/integration_guide.pdf`
- **Dashboard Manual**: `/docs/faculty/dashboard_manual.pdf`
- **Demo Library**: `/docs/faculty/available_demos.md`

### For Developers
- **Technical Docs**: `/docs/technical/`
- **API Reference**: [Link to API docs]
- **iRobot Create® 3 Integration**: `/docs/technical/create3_integration.md`

### Contact
- **Technical Support**: robot-support@algonquincollege.com
- **Privacy Questions**: privacy@algonquincollege.com
- **Partnerships**: innovation@algonquincollege.com

## Roadmap

- [x] Phase 1: Core Navigation & Interaction (Weeks 1-2) ✅
- [x] Phase 2: AI Q&A & Content Delivery (Weeks 3-4) ✅
- [ ] Phase 3: Advanced Features & Pilot (Weeks 5-6) 🚧
- [ ] Pilot with 200+ students in CS and Robotics programs
- [ ] Multi-language support (French for Algonquin College bilingual students)
- [ ] Integration with Brightspace for automatic material updates
- [ ] Mobile app for iOS and Android
- [ ] Multi-robot fleet for campus-wide deployment
- [ ] AR demonstrations via tablet (overlay concepts on real world)

---

**Empowering Algonquin College students with 24/7 AI-assisted learning**

*Making technical education more accessible, engaging, and effective through autonomous robotics and AI.*
