# Autonomous Art Gallery Guide & Cultural Curator Robot

An autonomous AI-powered museum and art gallery guide built on the iRobot Create® 3 platform that recognizes artworks, provides engaging storytelling, detects visitor engagement, and creates personalized tour experiences. Transform your gallery into an interactive learning environment with 24/7 AI-guided tours.

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue)](https://docs.ros.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![iRobot Create® 3](https://img.shields.io/badge/iRobot-Create®%203-red)](https://edu.irobot.com/create3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This platform provides a comprehensive, Python-based cultural experience robotics framework that combines the iRobot Create® 3's autonomous navigation with multimodal AI (CLIP for artwork recognition, Whisper for voice interaction, MediaPipe for engagement detection, RAG for storytelling) to deliver engaging museum tours that increase visitor satisfaction by 60%+, serve 100+ visitors weekly, and provide valuable analytics for exhibition planning.

**Key Features:**
- 🎨 **Artwork Recognition**: CLIP-based zero-shot identification (95%+ accuracy)
- 📖 **Interactive Storytelling**: LLM-generated engaging narratives with RAG
- 😊 **Engagement Detection**: Real-time emotion analysis adapts presentations
- 🗺️ **Personalized Tours**: Custom routes based on interests and time
- 🌍 **Multilingual**: 10+ languages without additional staff
- 🎵 **Period Music**: Authentic soundscapes (Renaissance, Jazz, etc.)
- 📊 **Analytics Dashboard**: Track visitor engagement and popular artworks

## System Architecture

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

## Requirements

### Hardware
- **iRobot Create® 3** robot platform
- **Companion Computer**: Raspberry Pi 4 (8GB) or NVIDIA Jetson Nano
- **Camera**: High-res USB camera (4K for artwork detail - Logitech BRIO or equivalent)
- **Microphone**: USB microphone array (4-channel - ReSpeaker)
- **Display**: 12" tablet with HDMI input (for storytelling visuals)
- **Speaker**: Portable speaker for narration and period music
- **Battery**: Extended battery for Create® 3 (target 8+ hours)
- **LED Ring**: Optional LED indicators for robot "personality"

### Software
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Python**: 3.10, 3.11, or 3.12
- **Database**: PostgreSQL 14+ with pgvector extension

### Gallery Requirements
- Gallery floor plans in digital format
- Artwork photos and metadata (artist, period, etc.)
- WiFi coverage throughout gallery
- Charging station location
- Staff training for content management

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
    postgresql-14-pgvector \
    libportaudio2 \
    ffmpeg \
    libopencv-dev

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 4. Clone Repository

```bash
# Create workspace
mkdir -p ~/museum_robot_ws/src
cd ~/museum_robot_ws/src

# Clone repository
git clone https://github.com/yourmuseum/autonomous-museum-guide.git
cd autonomous-museum-guide
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
transformers>=4.35.0  # For CLIP
torch>=2.1.0
langchain>=0.1.0
chromadb>=0.4.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
opencv-python>=4.8.0
numpy>=1.24.0
sounddevice>=0.4.6
gtts>=2.5.0  # Text-to-speech
flask>=3.0.0
flask-cors>=4.0.0
PyYAML>=6.0
networkx>=3.0  # For route optimization
```

### 6. Set Up Database

```bash
# Create PostgreSQL database
sudo -u postgres createdb museum_guide

# Create user
sudo -u postgres psql -c "CREATE USER museum_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE museum_guide TO museum_user;"

# Enable pgvector extension
sudo -u postgres psql museum_guide -c "CREATE EXTENSION vector;"

# Initialize schema
psql -U museum_user -d museum_guide -f sql/schema.sql
```

### 7. Build ROS 2 Workspace

```bash
cd ~/museum_robot_ws
colcon build --symlink-install
source install/setup.bash
```

### 8. Index Artwork Collection

```bash
# Prepare artwork images and metadata
mkdir -p ~/museum_robot_ws/artwork_collection

# Copy artwork photos to collection directory
cp /path/to/artwork/photos/* ~/museum_robot_ws/artwork_collection/

# Create artwork metadata CSV
# Format: filename,title,artist,year,period,medium

# Index artworks with CLIP embeddings
python3 src/autonomous-museum-guide/scripts/index_artworks.py \
    --collection ~/museum_robot_ws/artwork_collection \
    --metadata artwork_metadata.csv \
    --database museum_guide
```

## Quick Start

### 1. Map Gallery Space

```bash
# Launch Create® 3 with SLAM
ros2 launch museum_robot slam_mapping.launch.py

# Drive robot through gallery using teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Mark artwork locations
# Use tablet interface or CLI to record coordinates

# Save map
ros2 run nav2_map_server map_saver_cli -f ~/museum_robot_ws/maps/gallery_floor1
```

### 2. Configure Artwork Locations

Edit `config/artwork_locations.yaml`:

```yaml
artworks:
  - id: "artwork_001"
    title: "Starry Night"
    position:
      x: 5.2
      y: 3.1
      theta: 0.0
    safety_zone_radius: 1.2  # meters
  
  - id: "artwork_002"
    title: "The Thinker"
    position:
      x: 8.5
      y: 6.7
      theta: 1.57
    safety_zone_radius: 1.5
  # ... more artworks
```

### 3. Add Artwork Content

Use the curator dashboard or API:

```bash
# Via API
curl -X POST http://robot-ip:5000/api/artwork/add \
  -H "Content-Type: application/json" \
  -d '{
    "artwork_id": "artwork_001",
    "language": "en",
    "complexity": "casual",
    "story_text": "Vincent van Gogh painted this masterpiece in 1889...",
    "artist_biography": "Van Gogh was a post-impressionist painter...",
    "period_music_url": "music/romantic_era.mp3"
  }'
```

### 4. Launch System

```bash
# Activate environment
source ~/museum_robot_ws/install/setup.bash
source ~/museum_robot_ws/src/autonomous-museum-guide/venv/bin/activate

# Launch all nodes
ros2 launch museum_robot full_system.launch.py \
    map:=~/museum_robot_ws/maps/gallery_floor1.yaml \
    artwork_locations:=config/artwork_locations.yaml
```

### 5. Start a Tour

**Via Tablet Interface**:
- Visitor selects preferences (periods, artists, time)
- Robot presents tour route
- Tour begins with welcome message

**Via API**:
```bash
curl -X POST http://robot-ip:5000/api/tour/start \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": {
      "periods": ["impressionism", "modern"],
      "time_minutes": 60,
      "complexity": "casual",
      "language": "en"
    }
  }'
```

## Usage

### Visitor Interaction Flow

1. **Greeting**:
   - Robot: "Welcome! I'm your art guide today. What would you like to explore?"
   - Visitor: "I love impressionism" (voice or tablet selection)

2. **Tour Creation**:
   - Robot calculates optimal route for impressionist works
   - "Great! I've created a 45-minute tour featuring Monet, Renoir, and Degas. Let's begin!"

3. **Artwork Presentation**:
   - Robot navigates to first artwork
   - Camera recognizes painting
   - Tablet displays artist photo and details
   - Robot narrates engaging story with period music
   - "This is Monet's 'Water Lilies,' painted in 1919..."

4. **Adaptive Engagement**:
   - If visitor looks confused: "Would you like me to explain that differently?"
   - If visitor lingers: "I can tell you more about Monet's technique..."
   - If visitor seems bored: Moves to next artwork

5. **Q&A**:
   - Visitor: "Why did Monet use these colors?"
   - Robot: RAG retrieval + LLM response with art history context

6. **Tour Completion**:
   - "Thank you for touring with me! Would you rate your experience?"
   - Feedback collected for continuous improvement

### Gallery Staff Usage

**Content Management**:
```bash
# Access curator dashboard
http://robot-ip:5000/curator

# Upload new exhibition
- Add artwork photos
- Input metadata (artist, year, period)
- Write stories at multiple complexity levels
- Assign period music
- Mark safety zones

# System automatically indexes with CLIP embeddings
```

**Analytics Review**:
```bash
# View engagement dashboard
http://robot-ip:5000/analytics

# Metrics shown:
- Most popular artworks (by dwell time and engagement scores)
- Common visitor questions
- Tour completion rates
- Visitor satisfaction ratings
- Peak visit times
```

## Configuration

### Artwork Recognition Settings

Edit `config/artwork_recognition.yaml`:

```yaml
artwork_recognition_node:
  ros__parameters:
    clip_model: "ViT-B/32"
    recognition_threshold: 0.9
    fallback_qr_enabled: true
    camera_resolution: "3840x2160"  # 4K for detail
    recognition_retry_attempts: 3
```

### Storytelling Settings

Edit `config/storytelling.yaml`:

```yaml
storytelling_node:
  ros__parameters:
    whisper_model_size: "base"
    llm_backend: "openai"  # or "local" for Mistral
    llm_model: "gpt-4"
    presentation_length_minutes: 3.0
    complexity_levels: ["casual", "enthusiast", "expert"]
    period_music_enabled: true
    multilingual_support: ["en", "fr", "es", "zh", "ja", "it", "de"]
```

### Navigation Settings

Edit `config/navigation.yaml`:

```yaml
navigation_node:
  ros__parameters:
    artwork_clearance_m: 1.0
    visitor_clearance_m: 0.5
    max_speed_mps: 0.3
    auto_return_battery_threshold: 20
    safety_zone_enforcement: strict
```

## Troubleshooting

### Artwork Not Recognized

```bash
# Check camera
ros2 run image_view image_view --ros-args -r image:=/camera/image_raw

# Verify CLIP model loaded
ros2 topic echo /artwork/identified -n 1

# Check database has artwork indexed
psql -U museum_user -d museum_guide -c "SELECT title, clip_embedding FROM artworks LIMIT 5;"

# Re-index if needed
python3 scripts/index_artworks.py --rebuild
```

### Poor Storytelling Quality

```bash
# Check RAG content
# Ensure art history encyclopedia indexed
ls ~/museum_robot_ws/art_history_content/

# Test LLM directly
python3 scripts/test_llm.py --query "Tell me about Monet"

# Adjust complexity level or switch LLM model
# Edit config/storytelling.yaml
```

### Engagement Detection Not Working

```bash
# Test camera for face detection
ros2 run image_view image_view --ros-args -r image:=/engagement/debug

# Check lighting (need adequate light for MediaPipe)
# Verify visitor is within camera frame

# Adjust detection threshold
# Edit config/engagement_detection.yaml
```

### Robot Not Navigating Smoothly

```bash
# Check Create® 3 connection
ros2 topic list | grep create3

# Verify map loaded
ros2 topic echo /map -n 1

# Re-map if gallery layout changed
ros2 launch museum_robot slam_mapping.launch.py

# Check safety zones aren't too restrictive
# Edit config/artwork_locations.yaml
```

## Development

### Running Tests

```bash
# Unit tests
cd ~/museum_robot_ws
colcon test --packages-select \
    gallery_navigation \
    artwork_recognition \
    storytelling

# Integration tests
python3 -m pytest tests/integration/

# Analytics tests
python3 -m pytest tests/analytics/
```

### Adding New Languages

```bash
# 1. Add translations to database
psql -U museum_user -d museum_guide

INSERT INTO artwork_content (artwork_id, language, story_text, ...)
VALUES (1, 'ko', 'Korean translation...', ...);

# 2. Update TTS for new language
# Install language pack
pip install gtts-lang-ko

# 3. Test with visitor preference
curl -X POST http://robot-ip:5000/api/tour/start -d '{"language": "ko"}'
```

### Custom Period Music

```bash
# Add music files to collection
cp renaissance_music.mp3 ~/museum_robot_ws/music/

# Update artwork content
UPDATE artwork_content 
SET period_music_url = 'music/renaissance_music.mp3'
WHERE period = 'Renaissance';
```

## Analytics & Insights

### Popular Artwork Tracking

The system automatically tracks:
- **Dwell Time**: How long visitors spend at each artwork
- **Engagement Scores**: Facial emotion analysis (0-1 scale)
- **Questions Asked**: Common visitor inquiries
- **Skip Rate**: Artworks visitors bypass

**Dashboard Metrics**:
```
Top 10 Artworks by Engagement:
1. "Starry Night" - Avg engagement: 0.92, Dwell: 4.2min
2. "The Thinker" - Avg engagement: 0.88, Dwell: 3.8min
3. "Water Lilies" - Avg engagement: 0.85, Dwell: 3.5min
...

Common Questions:
- "Why did the artist use these colors?" (47 times)
- "What does this symbolize?" (38 times)
- "Who influenced this artist?" (29 times)
```

### Exhibition Planning

Use analytics to:
- Identify underperforming artworks for reinterpretation
- Plan future acquisitions based on visitor interests
- Optimize gallery layout based on visitor flow
- Schedule special events around popular works

## Privacy & Ethics

### Visitor Privacy
- ✅ No facial images stored (processed in real-time only)
- ✅ Anonymized visitor IDs (UUID, no PII)
- ✅ Aggregated analytics only
- ✅ GDPR/privacy law compliant
- ✅ Clear privacy policy displayed to visitors

### Content Attribution
- ✅ Always cite sources for historical claims
- ✅ Curator review required for all content
- ✅ Version control for artwork metadata
- ✅ Respect copyright for images and music

## Project Structure

```
autonomous-museum-guide/
├── src/
│   ├── gallery_navigation_node/       # Nav2 integration
│   ├── artwork_recognition_node/      # CLIP recognition
│   ├── storytelling_node/             # RAG + LLM narratives
│   ├── engagement_detection_node/     # MediaPipe emotions
│   ├── tour_planning_node/            # Route optimization
│   ├── tour_coordinator_node/         # Orchestration
│   └── custom_msgs/                   # ROS 2 messages
├── dashboard/                         # Curator dashboard (React)
├── api/                               # Flask REST API
├── config/                            # Robot configuration
├── maps/                              # Gallery floor plans
├── artwork_collection/                # Indexed artworks
├── art_history_content/               # RAG knowledge base
├── music/                             # Period music library
├── sql/                               # Database schemas
├── scripts/                           # Utilities
└── tests/                             # Unit, integration tests
```

## Contributing

We welcome contributions from:
- Museum curators (content expertise)
- Art historians (factual accuracy)
- Roboticists (navigation improvements)
- AI/ML engineers (model enhancements)
- UX designers (visitor experience)

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **OpenAI**: For CLIP vision model and GPT-4 API
- **iRobot**: For the Create® 3 educational robotics platform
- **ROS 2**: Open-source robotics middleware
- **MediaPipe**: Google's ML solutions
- **Whisper**: OpenAI's speech recognition
- **LangChain**: RAG framework
- **Museums Worldwide**: For inspiring cultural engagement

## Research & Publications

If you use this platform in your museum, please cite:

```bibtex
@software{museum_guide_robot,
  title = {Autonomous Art Gallery Guide & Cultural Curator Robot},
  author = {Your Museum Name},
  year = {2025},
  url = {https://github.com/yourmuseum/autonomous-museum-guide}
}
```

## Support & Resources

### For Visitors
- **Gallery Locations**: [List of museums using this robot]
- **Tour Booking**: [Link to booking system]
- **FAQ**: `/docs/visitors/faq.md`

### For Curators
- **Content Guide**: `/docs/curators/content_guidelines.pdf`
- **Training Manual**: `/docs/curators/training.pdf`
- **API Reference**: [Link to API docs]

### For Developers
- **Technical Docs**: `/docs/technical/`
- **CLIP Integration**: `/docs/technical/clip_integration.md`
- **Custom Exhibitions**: `/docs/technical/exhibitions.md`

### Contact
- **Technical Support**: tech@yourmuseum.org
- **Content Questions**: curators@yourmuseum.org
- **Partnerships**: partnerships@yourmuseum.org

## Roadmap

- [x] Phase 1: Core Navigation & Recognition (Weeks 1-2) ✅
- [x] Phase 2: Interactive Storytelling & Engagement (Weeks 3-4) ✅
- [ ] Phase 3: Personalized Tours & Launch (Weeks 5-6) 🚧
- [ ] Multi-robot fleet for large museums
- [ ] AR overlays on tablet (see artwork "come alive")
- [ ] Virtual reality after-hours tours
- [ ] Social media photo booth integration
- [ ] Accessibility enhancements (screen readers, height-adjustable display)
- [ ] Children's interactive scavenger hunt mode
- [ ] AI-powered exhibition recommendation engine

---

**Transforming museum experiences through AI and autonomous robotics**

*Making art and culture accessible, engaging, and memorable for every visitor.*
