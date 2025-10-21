# Autonomous Robotics Platform with AI Integration

A comprehensive ROS 2-based robotics platform that combines computer vision, audio processing, motion control, and AI inference using MediaPipe and Whisper for autonomous robotic agents on Linux systems.

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Iron-blue)](https://docs.ros.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20LTS-orange)](https://ubuntu.com/)

## Overview

This platform provides a unified, Python-based framework for developing autonomous robots with multimodal perception capabilities. It integrates state-of-the-art AI models (MediaPipe for vision, Whisper for audio) with ROS 2's distributed architecture, enabling rapid prototyping and deployment of intelligent robotic systems.

**Key Features:**
- 🎥 **Real-time Vision Processing**: Pose detection, gesture recognition, and object tracking using MediaPipe
- 🎤 **Voice Command Recognition**: Speech-to-text transcription and intent parsing with Whisper
- 🤖 **Motion Control**: Trajectory planning and coordinated actuator control
- 🖥️ **CLI Orchestration**: Single-command launch with health monitoring and diagnostics
- 🔄 **Modular Architecture**: Distributed ROS 2 nodes with fault tolerance and auto-recovery

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Orchestrator                           │
│           (Process Management, Monitoring, Logging)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    Coordinator Node                             │
│          (Message Routing, Synchronization, State)              │
└─────┬────────┬────────┬────────┬─────────────────────────────┘
      │        │        │        │
┌─────▼──┐ ┌──▼────┐ ┌─▼──────┐ ┌▼─────────────────────────────┐
│ Vision │ │ Audio │ │ Motion │ │   ROS 2 Parameter Server     │
│  Node  │ │ Node  │ │  Node  │ │   Configuration Management   │
└────────┘ └───────┘ └────────┘ └──────────────────────────────┘
    │          │          │
    ▼          ▼          ▼
[Camera]   [Microphone] [Actuators]
```

## Requirements

### Hardware
- **CPU**: Quad-core processor (Intel i5/Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (NVIDIA CUDA 11.x/12.x or AMD ROCm 5.x for acceleration)
- **Camera**: USB webcam or V4L2-compatible camera
- **Microphone**: Standard audio input device or microphone array
- **Robot**: Compatible actuators (optional for motion control testing)

### Software
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill or Iron Irwini
- **Python**: 3.10, 3.11, or 3.12
- **System Tools**: systemd, v4l-utils, alsa-utils

## Installation

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Humble (or Iron)
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions

# Install system packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    v4l-utils \
    alsa-utils \
    libportaudio2 \
    ffmpeg

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Clone Repository

```bash
# Create workspace
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws/src

# Clone repository
git clone https://github.com/yourusername/autonomous-robotics-platform.git
cd autonomous-robotics-platform
```

### 3. Install Python Dependencies

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
opencv-python>=4.8.0
numpy>=1.24.0
sounddevice>=0.4.6
PyYAML>=6.0
```

### 4. Build ROS 2 Workspace

```bash
cd ~/robotics_ws
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

### 1. Launch the Complete System

```bash
# Activate environment
source ~/robotics_ws/install/setup.bash
source ~/robotics_ws/src/autonomous-robotics-platform/venv/bin/activate

# Launch all nodes
python3 src/autonomous-robotics-platform/src/cli_launcher/launch.py \
    --config src/autonomous-robotics-platform/config/system.yaml
```

### 2. Verify System Status

In another terminal:
```bash
# Check active nodes
ros2 node list

# Monitor vision output
ros2 topic echo /vision/pose

# Monitor audio transcription
ros2 topic echo /audio/transcript

# Check system diagnostics
ros2 topic echo /system/status
```

### 3. Test Vision Processing

```bash
# Wave at the camera to trigger gesture detection
# Pose landmarks will be published to /vision/pose
# Gesture events will appear on /vision/gestures
```

### 4. Test Voice Commands

```bash
# Speak commands like:
# "move forward"
# "stop"
# "turn left"
# Commands will be published to /audio/commands
```

## Configuration

### Vision Node Configuration

Edit `config/vision_node.yaml`:

```yaml
vision_node:
  ros__parameters:
    camera_index: 0              # Camera device ID
    fps_target: 30               # Target frame rate
    model_complexity: 1          # MediaPipe complexity (0-2)
    min_detection_confidence: 0.5
    enable_pose: true
    enable_hands: true
    enable_objects: false
```

### Audio Node Configuration

Edit `config/audio_node.yaml`:

```yaml
audio_node:
  ros__parameters:
    model_size: "base"           # Whisper model: tiny/base/small/medium
    sample_rate: 16000           # Audio sample rate
    vad_threshold: 0.5           # Voice activity threshold
    device_index: 0              # Microphone device ID
    command_patterns:
      move_forward: ["move forward", "go forward", "advance"]
      stop: ["stop", "halt", "freeze"]
      turn_left: ["turn left", "left turn", "go left"]
      turn_right: ["turn right", "right turn", "go right"]
```

### Motion Node Configuration

Edit `config/motion_node.yaml`:

```yaml
motion_node:
  ros__parameters:
    robot_description: "config/robot.urdf"
    max_velocity: 1.0
    max_acceleration: 0.5
    control_rate: 100            # Hz
    safety_limits:
      workspace_bounds: [-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]
```

## Usage Examples

### Example 1: Gesture-Controlled Robot

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_msgs.msg import GestureEvent, MotionCommand

class GestureController(Node):
    def __init__(self):
        super().__init__('gesture_controller')
        self.gesture_sub = self.create_subscription(
            GestureEvent, '/vision/gestures', self.gesture_callback, 10)
        self.motion_pub = self.create_publisher(
            MotionCommand, '/motion/commands', 10)
    
    def gesture_callback(self, msg):
        if msg.gesture_type == 'thumbs_up':
            # Move forward on thumbs up
            cmd = MotionCommand()
            cmd.command_type = 'velocity'
            cmd.joint_velocities = [1.0, 1.0]  # Forward
            self.motion_pub.publish(cmd)
        elif msg.gesture_type == 'stop':
            # Stop on stop gesture
            cmd = MotionCommand()
            cmd.command_type = 'velocity'
            cmd.joint_velocities = [0.0, 0.0]
            self.motion_pub.publish(cmd)

if __name__ == '__main__':
    rclpy.init()
    node = GestureController()
    rclpy.spin(node)
```

### Example 2: Voice-Controlled Navigation

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_msgs.msg import VoiceCommand, MotionCommand

class VoiceNavigator(Node):
    def __init__(self):
        super().__init__('voice_navigator')
        self.voice_sub = self.create_subscription(
            VoiceCommand, '/audio/commands', self.voice_callback, 10)
        self.motion_pub = self.create_publisher(
            MotionCommand, '/motion/commands', 10)
    
    def voice_callback(self, msg):
        cmd = MotionCommand()
        cmd.command_type = 'velocity'
        
        if msg.intent == 'move_forward':
            cmd.joint_velocities = [1.0, 1.0]
        elif msg.intent == 'turn_left':
            cmd.joint_velocities = [0.5, -0.5]
        elif msg.intent == 'turn_right':
            cmd.joint_velocities = [-0.5, 0.5]
        elif msg.intent == 'stop':
            cmd.joint_velocities = [0.0, 0.0]
        
        self.motion_pub.publish(cmd)

if __name__ == '__main__':
    rclpy.init()
    node = VoiceNavigator()
    rclpy.spin(node)
```

## Performance Tuning

### GPU Acceleration

Enable GPU acceleration for MediaPipe:
```yaml
vision_node:
  ros__parameters:
    use_gpu: true
    gpu_device: 0
```

### Whisper Model Selection

Choose model based on performance needs:
- **tiny**: Fastest, lower accuracy (~1GB RAM)
- **base**: Balanced, good accuracy (~1.5GB RAM)
- **small**: High accuracy (~2GB RAM)
- **medium**: Highest accuracy (~5GB RAM)

### CPU Optimization

Pin nodes to specific CPU cores for real-time performance:
```bash
# Pin motion control to CPU 0-1
taskset -c 0-1 ros2 run motion_control motion_node
```

## Troubleshooting

### Camera Not Detected
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
ffplay /dev/video0

# Update camera_index in config/vision_node.yaml
```

### Microphone Issues
```bash
# List audio devices
arecord -l

# Test microphone
arecord -d 5 test.wav && aplay test.wav

# Update device_index in config/audio_node.yaml
```

### Node Crashes
```bash
# Check logs
ros2 node info /vision_node
ros2 topic hz /vision/pose

# Enable debug logging
export RCUTILS_LOGGING_SEVERITY=DEBUG
```

### Performance Issues
- Reduce vision FPS target
- Use smaller Whisper model (tiny/base)
- Enable GPU acceleration
- Close unnecessary applications

## Development

### Running Tests

```bash
# Unit tests
cd ~/robotics_ws
colcon test --packages-select vision_node audio_node motion_node

# Integration tests
python3 -m pytest tests/integration/

# Test coverage
coverage run -m pytest tests/
coverage report
```

### Code Style

This project follows PEP 8 style guidelines:
```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

### Building Documentation

```bash
cd docs/
make html
firefox _build/html/index.html
```

## Project Structure

```
autonomous-robotics-platform/
├── src/
│   ├── vision_node/          # Vision processing node
│   ├── audio_node/           # Audio processing node
│   ├── motion_node/          # Motion control node
│   ├── coordinator_node/     # System coordinator
│   ├── custom_msgs/          # Custom ROS 2 messages
│   └── cli_launcher/         # CLI orchestration tool
├── config/                   # Configuration files
│   ├── vision_node.yaml
│   ├── audio_node.yaml
│   ├── motion_node.yaml
│   └── system.yaml
├── launch/                   # ROS 2 launch files
├── tests/                    # Unit and integration tests
├── docs/                     # Sphinx documentation
├── examples/                 # Usage examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass (`colcon test`)
- Documentation is updated
- Commit messages are descriptive

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ROS 2**: Open-source robotics middleware
- **MediaPipe**: Google's cross-platform ML solutions
- **OpenAI Whisper**: Robust speech recognition system
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing library

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{autonomous_robotics_platform,
  title = {Autonomous Robotics Platform with AI Integration},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/autonomous-robotics-platform}
}
```

## Support

- **Documentation**: [https://autonomous-robotics-platform.readthedocs.io](https://autonomous-robotics-platform.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/autonomous-robotics-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autonomous-robotics-platform/discussions)
- **Email**: support@yourproject.com

## Roadmap

- [ ] Phase 1: Foundation (Weeks 1-3) ✅
- [ ] Phase 2: Integration (Weeks 4-6) 🚧
- [ ] Phase 3: Enhancement (Weeks 7-8)
- [ ] Additional sensor support (LIDAR, depth cameras)
- [ ] Web-based monitoring dashboard
- [ ] Docker containerization
- [ ] Cloud deployment support
- [ ] Multi-robot coordination

## Community

Join our community:
- Discord: [Join Server](https://discord.gg/yourserver)
- Slack: [Join Workspace](https://yourworkspace.slack.com)
- Twitter: [@YourProject](https://twitter.com/yourproject)

---

**Built with ❤️ for the robotics community**
