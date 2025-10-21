# Feature Idea: Interactive AI Teaching Assistant Robot for Algonquin College

## Overview
An autonomous AI-powered teaching assistant robot built on the iRobot Create® 3 platform that navigates Algonquin College campuses, provides personalized learning support to students, demonstrates robotics and AI concepts through interactive explanations, answers technical questions using voice interaction, and assists faculty with classroom demonstrations. The system combines autonomous navigation, computer vision for student recognition and engagement detection, voice interaction for Q&A, and adaptive learning content delivery.

## Problem Statement
Algonquin College students and faculty face several challenges in the learning environment:
- Limited access to instructors during off-hours (evenings, weekends)
- Difficulty getting immediate help with technical concepts in STEM courses
- Lack of hands-on robotics demonstrations for students learning ROS 2, AI, and robotics concepts
- Need for personalized learning support that adapts to individual student pace
- Limited resources for one-on-one tutoring in technical subjects
- Challenges in making abstract AI/robotics concepts tangible and interactive
- Desire for engaging, interactive learning experiences beyond traditional lectures

Traditional teaching assistants are limited by availability, scalability, and cost, while online resources lack the interactive, hands-on demonstration capabilities that make complex technical concepts accessible.

## Proposed Solution
Implement an iRobot Create® 3-based teaching assistant robot with five integrated AI modules:

- **Autonomous Campus Navigation Module**: ROS 2 Navigation2 stack enabling the robot to autonomously navigate hallways, classrooms, labs, and common areas, responding to student requests via mobile app to come provide assistance

- **Student Recognition & Engagement Module**: MediaPipe facial recognition to identify registered students, detect attention levels and confusion through facial expression analysis, and personalize interactions based on individual learning profiles

- **Voice Q&A Module**: Whisper-powered speech recognition with RAG (Retrieval-Augmented Generation) connected to course materials, textbooks, and documentation, enabling natural language Q&A about programming, robotics, AI, and STEM subjects

- **Interactive Demonstration Module**: Real-time visualization of robotics concepts (sensor data, path planning, SLAM mapping) on attached tablet display, allowing students to see "inside the robot" and understand how autonomous systems work

- **Adaptive Learning Module**: Track student interactions and learning progress, identify knowledge gaps, adjust explanation complexity, and recommend resources based on individual needs and course curriculum

## Expected Benefits
- **24/7 Learning Support**: Students can request assistance any time the robot is operational, including evenings and weekends
- **Scalable Tutoring**: One robot can assist multiple students throughout the day, multiplying teaching capacity
- **Hands-On Learning**: Students see real robotics and AI in action rather than just theory
- **Personalized Instruction**: Robot adapts explanations to student's knowledge level and learning pace
- **Faculty Support**: Assists professors with in-class demonstrations and lab sessions
- **Engagement Boost**: Interactive robot increases student engagement with technical subjects by 40%+
- **Concept Visualization**: Makes abstract concepts (SLAM, sensor fusion, path planning) tangible
- **Accessibility**: Provides consistent, patient support for students who may be intimidated asking instructors

## Technical Considerations
- **Platform**: iRobot Create® 3 with Raspberry Pi 4 or NVIDIA Jetson Nano companion computer
- **Technology Stack**: ROS 2 (Humble), Python 3.10+, MediaPipe, Whisper, LangChain (for RAG), Navigation2 stack
- **Hardware Additions**: 
  - 10" tablet display for visualizations (mounted on Create® 3)
  - USB webcam for student recognition and engagement detection
  - USB microphone array for voice interaction
  - Battery extender for 6+ hour operation
- **Content Integration**: RAG system connected to Algonquin course materials, textbooks, ROS 2 documentation, Python docs
- **Navigation**: Pre-mapped college hallways using SLAM, dynamic obstacle avoidance for students/furniture
- **Safety**: Create® 3's built-in cliff detection, bumper sensors, gentle collision avoidance for crowded hallways
- **Student Privacy**: Opt-in facial recognition, encrypted student data, GDPR/FIPPA compliance
- **Scheduling**: Integration with course timetables to be available in relevant locations (CS lab during programming hours, robotics lab during ROS courses)

## Project System Integration
- **Navigation Pipeline**: Create® 3 odometry + LIDAR (if available) → Navigation2 SLAM → Campus map database → Path planner → Create® 3 motion commands
- **Student Recognition Pipeline**: Webcam → MediaPipe face detection → Face recognition → Student profile lookup → Personalized greeting + learning history
- **Q&A Pipeline**: Mic array → Whisper STT → LangChain RAG (course materials) → GPT/Local LLM → Response generation → Text-to-speech → Tablet display
- **Engagement Pipeline**: Webcam → MediaPipe facial expression → Engagement classifier (focused/confused/bored) → Content adaptation → Explanation simplification/elaboration
- **Demonstration Pipeline**: ROS 2 topics (sensors, planning, etc.) → Data visualization → Tablet UI → Student learning → Interactive Q&A

## Initial Scope
### Phase 1: Core Navigation & Interaction (Weeks 1-2)
- ROS 2 integration with iRobot Create® 3 platform
- Navigation2 setup with campus hallway mapping
- Basic autonomous navigation between key locations (classrooms, labs, study areas)
- MediaPipe integration for student face detection
- Whisper integration for voice command recognition ("Robot, come help me", "Explain SLAM")
- Text-to-speech response system
- Tablet display for basic text output

### Phase 2: AI Q&A & Content Delivery (Weeks 3-4)
- RAG system integration with Algonquin course materials (CS, IT, robotics courses)
- LangChain pipeline for context-aware question answering
- Student profile database with learning history
- Engagement detection via facial expression analysis
- Adaptive explanation system (beginner/intermediate/advanced)
- Interactive ROS 2 visualization on tablet (sensor data, map, path planning)
- Mobile app for students to request robot assistance

### Phase 3: Advanced Features & Deployment (Weeks 5-6)
- Multi-student interaction capability (queue management)
- Scheduled locations based on course timetables
- Progress tracking and knowledge gap identification
- Faculty dashboard for monitoring robot usage and student engagement
- Voice-activated demonstrations ("Show me how path planning works")
- Integration with campus WiFi for continuous operation
- Safety compliance and student privacy features
- Pilot deployment in select buildings with student feedback

## Success Criteria
- [ ] Robot navigates autonomously between 5+ campus locations with 95%+ success rate
- [ ] Student recognition achieves 90%+ accuracy for registered students
- [ ] Voice Q&A answers technical questions correctly 80%+ of the time (validated by faculty)
- [ ] Engagement detection identifies confused students with 75%+ accuracy
- [ ] Students rate robot usefulness ≥4/5 for learning support
- [ ] 50+ unique student interactions per week during pilot
- [ ] Battery lasts ≥6 hours of continuous operation
- [ ] System operates safely in crowded hallways with zero collision incidents
- [ ] Faculty validate accuracy of technical explanations and demonstrations
- [ ] Student learning outcomes improve by 15%+ in pilot courses (measured by test scores)
- [ ] Privacy compliance verified (student consent, data encryption, FIPPA)
- [ ] 80%+ of students report robot made learning more engaging/accessible
