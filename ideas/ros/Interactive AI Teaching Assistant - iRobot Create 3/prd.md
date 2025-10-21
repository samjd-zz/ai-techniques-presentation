# Product Requirements Document: Interactive AI Teaching Assistant Robot for Algonquin College

## Executive Summary
This PRD defines requirements for an autonomous AI-powered teaching assistant robot built on the iRobot Create® 3 platform that navigates Algonquin College campuses to provide personalized learning support. The system uses multimodal AI (MediaPipe for student recognition, Whisper for voice interaction, RAG for Q&A) to deliver 24/7 tutoring, interactive robotics demonstrations, and adaptive content delivery for STEM students.

**Target Users**: STEM students (Computer Science, IT, Robotics programs), faculty members, teaching assistants, academic support staff, campus administrators

**Business Value**: Scales teaching capacity by providing consistent, patient, 24/7 learning support; increases student engagement with technical subjects by 40%+; improves learning outcomes by 15%+ in pilot courses; reduces instructor workload for routine questions; makes abstract robotics/AI concepts tangible through hands-on demonstrations.

## Project Context
### Domain
Higher education technology and AI-assisted learning, focusing on STEM education at Algonquin College where students require hands-on support with programming, robotics, ROS 2, artificial intelligence, and technical concepts that benefit from interactive, visual demonstrations.

### Current Challenges
- Limited instructor availability during off-hours (evenings, weekends, exam periods)
- High student-to-instructor ratios in technical programs
- Difficulty providing personalized, one-on-one support at scale
- Abstract concepts in robotics and AI challenging to grasp without hands-on experience
- Students reluctant to ask "simple" questions in class or office hours
- Need for consistent, patient support for diverse learning paces
- Limited resources for continuous tutoring and demonstration support

### Technology Foundation
- **Platform**: iRobot Create® 3 with Raspberry Pi 4 or NVIDIA Jetson Nano
- **Framework**: ROS 2 (Humble) with Navigation2 stack
- **Language**: Python 3.10+
- **AI Models**: MediaPipe, Whisper, LangChain (RAG), local or cloud LLM
- **Hardware**: USB webcam, microphone array, 10" tablet display, battery extender
- **Infrastructure**: Campus WiFi, PostgreSQL for student data, course material database
- **Compliance**: FIPPA (Ontario privacy), accessibility standards (AODA)

## User Stories

### Epic 1: Autonomous Campus Navigation
**US-1.1**: As a student studying in the library, I want to request the robot to come help me via mobile app, so that I can get assistance without leaving my study spot.
- **AC1**: Mobile app allows students to request robot assistance with location sharing
- **AC2**: Robot plans path to student location using Navigation2
- **AC3**: Robot arrives within 5 minutes for requests within same building
- **AC4**: Robot notifies student when it's approaching

**US-1.2**: As a faculty member, I want the robot to autonomously patrol scheduled locations (labs, classrooms), so that it's available when and where students need help most.
- **AC1**: Robot follows schedule tied to course timetables
- **AC2**: Automatically navigates to CS lab during programming hours, robotics lab during ROS courses
- **AC3**: Pauses at high-traffic areas (study zones, outside classrooms) between scheduled stops
- **AC4**: Faculty can override schedule for special events/demonstrations

**US-1.3**: As a campus administrator, I want the robot to navigate safely in crowded hallways, so that it doesn't disrupt student traffic or cause safety concerns.
- **AC1**: Robot maintains 0.5m clearance from students using Create® 3 sensors
- **AC2**: Slows down in crowded areas, stops if path blocked
- **AC3**: Audible alert when approaching from behind ("Excuse me")
- **AC4**: Zero collision incidents during 100+ hours of operation

### Epic 2: Student Recognition & Personalization
**US-2.1**: As a registered student, I want the robot to recognize me and remember our previous interactions, so that I receive personalized support based on my learning history.
- **AC1**: Facial recognition identifies registered students with 90%+ accuracy
- **AC2**: Robot greets student by name and references previous topics discussed
- **AC3**: Adapts explanation complexity based on student's course level and past interactions
- **AC4**: Students can opt-in to recognition during first interaction

**US-2.2**: As a student struggling with a concept, I want the robot to detect my confusion and adjust its explanation, so that I can better understand difficult material.
- **AC1**: MediaPipe detects facial expressions indicating confusion (furrowed brow, head shaking)
- **AC2**: Robot asks "Would you like me to explain that differently?" when confusion detected
- **AC3**: Simplifies explanation or provides visual demonstration when student appears confused
- **AC4**: Engagement detection achieves 75%+ accuracy for confused vs. focused states

**US-2.3**: As an international student, I want interactions to be patient and respectful of language barriers, so that I feel comfortable asking for help.
- **AC1**: Robot speaks slowly and clearly by default
- **AC2**: Can repeat explanations without showing impatience
- **AC3**: Offers to display text explanations on tablet for reading
- **AC4**: Multi-language support for greetings and basic interactions

### Epic 3: Voice Q&A & Content Delivery
**US-3.1**: As a CS student, I want to ask the robot programming questions and get accurate answers from course materials, so that I can overcome learning obstacles immediately.
- **AC1**: Whisper recognizes student questions with 85%+ accuracy in typical campus noise
- **AC2**: RAG system retrieves relevant content from course materials, textbooks, documentation
- **AC3**: Answers are accurate 80%+ of the time (validated by faculty review)
- **AC4**: Robot cites sources (e.g., "According to your Python textbook, chapter 5...")

**US-3.2**: As a robotics student, I want to ask "How does SLAM work?" and see a live demonstration on the robot itself, so that I understand the concept practically.
- **AC1**: Voice command triggers relevant demonstration
- **AC2**: Tablet displays real-time sensor data, map building, localization
- **AC3**: Robot explains what's happening as it demonstrates
- **AC4**: Student can ask follow-up questions during demonstration

**US-3.3**: As a student with a complex multi-part question, I want the robot to have conversational context, so that I don't need to repeat background information.
- **AC1**: System maintains conversation history for current interaction session
- **AC2**: Robot references previous questions/answers in conversation
- **AC3**: Can handle "follow-up" questions like "Can you explain that part about recursion again?"
- **AC4**: Conversation context persists for 30 minutes or until student ends interaction

### Epic 4: Interactive Demonstrations
**US-4.1**: As a ROS 2 student, I want to see the robot's internal state (topics, nodes, sensor data), so that I understand how ROS systems actually work.
- **AC1**: Tablet displays live ROS 2 graph showing active nodes and topics
- **AC2**: Can visualize specific topic data (odometry, laser scan, camera feed)
- **AC3**: Student can request specific visualizations ("Show me the laser scan data")
- **AC4**: Demonstrations synchronized with verbal explanations

**US-4.2**: As a faculty member teaching path planning, I want to use the robot for in-class demonstrations, so that students see algorithms in action rather than just slides.
- **AC1**: Faculty can trigger specific demonstrations via tablet or voice
- **AC2**: Robot executes path planning algorithms with real-time visualization
- **AC3**: Can demonstrate A*, Dijkstra, obstacle avoidance visually
- **AC4**: Demonstration mode projects to classroom screen via HDMI/wireless

**US-4.3**: As an AI student, I want to see the robot's decision-making process, so that I understand how AI systems process information and make choices.
- **AC1**: Robot explains its reasoning ("I'm taking this path because...")
- **AC2**: Shows confidence scores for decisions (recognition confidence, path cost)
- **AC3**: Demonstrates machine learning concepts (classification, feature detection)
- **AC4**: Can "think aloud" about complex decisions for educational purposes

### Epic 5: Progress Tracking & Adaptive Learning
**US-5.1**: As a student, I want the robot to track my learning progress and identify topics I struggle with, so that I can focus my study efforts effectively.
- **AC1**: System logs topics discussed and questions asked per student
- **AC2**: Identifies patterns in repeated questions (knowledge gaps)
- **AC3**: Provides personalized study recommendations
- **AC4**: Students can view their interaction history and progress in mobile app

**US-5.2**: As a faculty member, I want to see aggregate data on common student questions, so that I can adjust my teaching to address widespread confusion.
- **AC1**: Faculty dashboard shows most frequently asked questions
- **AC2**: Identifies concepts students struggle with most (by topic/course)
- **AC3**: Provides anonymized engagement and confusion metrics
- **AC4**: Export reports for curriculum improvement

**US-5.3**: As an academic advisor, I want to identify students who haven't engaged with learning support, so that I can proactively reach out to at-risk students.
- **AC1**: Dashboard shows student engagement frequency
- **AC2**: Alerts for students with zero interactions despite poor course performance
- **AC3**: Respects student privacy (aggregated data only)
- **AC4**: Integration with student success tracking systems

## Functional Requirements

### FR-1: Autonomous Navigation Module
- **FR-1.1**: Integrate iRobot Create® 3 with ROS 2 Navigation2 stack
- **FR-1.2**: SLAM mapping of campus hallways, classrooms, labs, study areas
- **FR-1.3**: Dynamic path planning with real-time obstacle avoidance
- **FR-1.4**: Location-based scheduling (be at CS lab during programming hours)
- **FR-1.5**: Mobile app integration for student-initiated navigation requests
- **FR-1.6**: Publish robot location to `/robot/position` topic for tracking
- **FR-1.7**: Safety behaviors: slow in crowds, audible alerts, emergency stop

### FR-2: Student Recognition Module
- **FR-2.1**: MediaPipe face detection and tracking via USB webcam
- **FR-2.2**: Face recognition system with student profile database
- **FR-2.3**: Opt-in enrollment process with photo capture and consent
- **FR-2.4**: Personalized greetings based on student profile
- **FR-2.5**: Learning history retrieval (previous topics, knowledge level)
- **FR-2.6**: Privacy mode for unregistered users (generic interaction)
- **FR-2.7**: Encrypted storage of facial recognition data

### FR-3: Voice Q&A Module
- **FR-3.1**: Whisper speech-to-text for student questions
- **FR-3.2**: LangChain RAG pipeline connected to course material database
- **FR-3.3**: Course material ingestion (PDFs, markdown, HTML documentation)
- **FR-3.4**: Context-aware question answering with source citations
- **FR-3.5**: Text-to-speech response generation with natural prosody
- **FR-3.6**: Conversation history management (30-minute context window)
- **FR-3.7**: Fallback to "Let me connect you with a TA" for unknown topics

### FR-4: Engagement Detection Module
- **FR-4.1**: MediaPipe facial landmark detection for expression analysis
- **FR-4.2**: Confusion classifier (focused vs. confused vs. bored states)
- **FR-4.3**: Real-time engagement scoring during interactions
- **FR-4.4**: Adaptive explanation system (simplify when confusion detected)
- **FR-4.5**: Publish engagement metrics to `/student/engagement` topic
- **FR-4.6**: Alert faculty if persistent confusion across multiple students (concept issue)

### FR-5: Interactive Demonstration Module
- **FR-5.1**: Tablet UI displaying ROS 2 visualizations (rviz-like interface)
- **FR-5.2**: Real-time sensor data visualization (laser scan, odometry, images)
- **FR-5.3**: SLAM map display with robot position and path planning
- **FR-5.4**: Voice-triggered demonstrations ("Show me path planning")
- **FR-5.5**: Synchronized verbal explanations during demonstrations
- **FR-5.6**: Annotation overlays explaining what data means
- **FR-5.7**: Screen mirroring to classroom displays for group demonstrations

## Non-Functional Requirements

### NFR-1: Performance & Responsiveness
- **NFR-1.1**: Voice recognition latency <2 seconds from speech to response start
- **NFR-1.2**: Navigation requests processed and path planned within 5 seconds
- **NFR-1.3**: Student recognition completes within 1 second of face detection
- **NFR-1.4**: RAG Q&A response time <10 seconds for typical questions
- **NFR-1.5**: Tablet visualization refresh rate ≥15 FPS
- **NFR-1.6**: Battery life ≥6 hours continuous operation

### NFR-2: Educational Effectiveness
- **NFR-2.1**: Q&A accuracy ≥80% validated by faculty review (100 sample questions)
- **NFR-2.2**: Student usefulness rating ≥4/5 (survey after interaction)
- **NFR-2.3**: Learning outcomes improve 15%+ in pilot courses (test score comparison)
- **NFR-2.4**: Student engagement increase 40%+ (measured by interaction frequency)
- **NFR-2.5**: Concept comprehension improved for 70%+ of students (self-reported)

### NFR-3: Safety & Reliability
- **NFR-3.1**: Zero collisions with students or property during operation
- **NFR-3.2**: Create® 3 safety features (cliff detection, bumpers) always active
- **NFR-3.3**: Emergency stop button accessible on robot and via mobile app
- **NFR-3.4**: System uptime ≥95% during scheduled operation hours
- **NFR-3.5**: Graceful degradation if sensors fail (revert to teleoperation mode)
- **NFR-3.6**: Automatic return to charging station below 20% battery

### NFR-4: Privacy & Compliance
- **NFR-4.1**: FIPPA compliant for Ontario student data handling
- **NFR-4.2**: Facial recognition opt-in only with explicit consent
- **NFR-4.3**: Student data encrypted at rest (AES-256) and in transit (TLS)
- **NFR-4.4**: Right to data deletion upon student request
- **NFR-4.5**: Anonymized data for faculty dashboards (no personally identifiable information)
- **NFR-4.6**: Audit logging of all data access
- **NFR-4.7**: AODA accessibility compliance (screen reader compatible interfaces)

### NFR-5: Usability & Accessibility
- **NFR-5.1**: First-time users can interact successfully without training
- **NFR-5.2**: Voice interaction works in typical campus noise levels (50-70 dB)
- **NFR-5.3**: Tablet text readable at 1.5m distance (minimum 18pt font)
- **NFR-5.4**: Clear, friendly voice with adjustable volume
- **NFR-5.5**: Multi-language support (English, French) for basic interactions
- **NFR-5.6**: Inclusive design accommodating diverse student needs

### NFR-6: Integration & Maintenance
- **NFR-6.1**: Integration with Algonquin course management systems
- **NFR-6.2**: Mobile app works on iOS and Android
- **NFR-6.3**: Remote monitoring and diagnostics capability
- **NFR-6.4**: Software updates deployable without extended downtime
- **NFR-6.5**: Maintenance required ≤1 hour per week
- **NFR-6.6**: Component replacement without specialized tools

## System Integration

### Dependencies
**Hardware**:
- iRobot Create® 3 robot platform
- Raspberry Pi 4 (4GB+) or NVIDIA Jetson Nano
- USB webcam (1080p, wide angle)
- USB microphone array (4-channel minimum)
- 10" tablet display with HDMI input
- Battery extender for Create® 3 (6+ hour target)

**Software**:
- ROS 2 Humble
- Navigation2 stack
- MediaPipe 0.10+
- OpenAI Whisper
- LangChain for RAG pipeline
- PostgreSQL for student data
- Local LLM (Llama 2, Mistral) or cloud API (OpenAI GPT-4)

**Integration Points**:
- Algonquin student information system (for enrollment verification)
- Course management systems (Brightspace) for material access
- Campus WiFi infrastructure
- Charging station locations

### Data Flow
1. **Navigation**: Mobile app request → Coordinator → Navigation2 planner → Create® 3 motion → Position updates
2. **Recognition**: Camera → MediaPipe → Face recognition → Database lookup → Personalized profile
3. **Q&A**: Microphone → Whisper → LangChain RAG (course materials) → LLM → TTS → Tablet + speaker
4. **Engagement**: Camera → MediaPipe landmarks → Confusion classifier → Adaptive content selector
5. **Demonstration**: ROS 2 topics → Visualization engine → Tablet display + verbal explanation

### API Interfaces
**ROS 2 Topics**:
- `/robot/position` (geometry_msgs/PoseStamped): Current robot location
- `/student/recognized` (custom_msgs/StudentProfile): Identified student info
- `/student/engagement` (custom_msgs/EngagementLevel): Real-time engagement score
- `/demonstration/active` (custom_msgs/Demo): Current demonstration state

**REST API**:
- `POST /api/request`: Student requests robot assistance
- `GET /api/schedule`: Retrieve robot schedule
- `GET /api/faculty/dashboard`: Faculty analytics
- `DELETE /api/student/data`: Student data deletion (FIPPA compliance)

## Success Metrics

### Educational Impact
- **EI-1**: Student learning outcomes improve 15%+ in pilot courses (test scores)
- **EI-2**: Student engagement with technical subjects increases 40%+ (interaction frequency)
- **EI-3**: 70%+ of students report improved concept comprehension
- **EI-4**: Student usefulness rating ≥4/5 (post-interaction survey)
- **EI-5**: 50+ unique student interactions per week during pilot

### Technical Performance
- **TP-1**: Navigation success rate 95%+ (reaches destination without intervention)
- **TP-2**: Student recognition accuracy 90%+ for registered students
- **TP-3**: Q&A accuracy 80%+ validated by faculty (100 sample questions)
- **TP-4**: Engagement detection 75%+ accuracy (confused vs. focused)
- **TP-5**: Battery life ≥6 hours continuous operation

### Operational Success
- **OS-1**: System uptime ≥95% during scheduled hours
- **OS-2**: Zero collision incidents during 100+ operation hours
- **OS-3**: Faculty adoption: 5+ faculty use for demonstrations
- **OS-4**: Privacy compliance verified (FIPPA audit)
- **OS-5**: Maintenance time ≤1 hour per week

### Adoption & Scale
- **AS-1**: 200+ registered students enrolled in pilot semester
- **AS-2**: 80%+ of surveyed students recommend to peers
- **AS-3**: Faculty request expansion to additional courses/buildings
- **AS-4**: ROI: Cost per student interaction <$5 (vs. $50+ for human TA hour)

## Timeline & Milestones

### Phase 1: Core Navigation & Interaction (Weeks 1-2)
**Deliverables**:
- iRobot Create® 3 + ROS 2 integration
- Navigation2 with campus map (1 building)
- Autonomous navigation between 5 key locations
- MediaPipe student face detection
- Whisper voice recognition for basic commands
- Text-to-speech responses
- Tablet display showing text output

**Exit Criteria**: Robot navigates autonomously; responds to voice; displays text on tablet

### Phase 2: AI Q&A & Content Delivery (Weeks 3-4)
**Deliverables**:
- LangChain RAG system with first course materials (intro CS)
- Student profile database with learning history
- Engagement detection (confusion vs. focused)
- Adaptive explanation system (3 complexity levels)
- Interactive ROS 2 visualizations on tablet
- Mobile app for requesting assistance
- Multi-student queue management

**Exit Criteria**: Answers course questions accurately; adapts to student confusion; visualizations functional

### Phase 3: Advanced Features & Pilot Deployment (Weeks 5-6)
**Deliverables**:
- Schedule-based autonomous operation (course timetable integration)
- Faculty dashboard with analytics
- Voice-activated demonstrations for 5 robotics concepts
- Privacy features and FIPPA compliance verification
- Extended battery operation (6+ hours)
- Safety validation (hallway navigation, crowd handling)
- Pilot deployment in 1 building with 50+ students
- Student and faculty feedback collection

**Exit Criteria**: All success criteria met; faculty validate educational value; privacy compliance verified; ready for campus-wide deployment
