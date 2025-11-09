# Drivable 🚗✋

**Transform your hands into a virtual steering wheel for Roblox driving games!**

Drivable is a real-time hand-tracking controller that uses computer vision to let you drive in Roblox games such Driving Empire using natural hand gestures. Hold your hands like you're gripping a steering wheel, tilt to steer, and show an open palm to brake—all captured through your webcam.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.0+-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## ✨ Features

- **Intuitive Hand Steering**: Raise your left hand to steer left, right hand to steer right, keep level to go straight
- **Proportional Control**: Steeper hand angles = sharper turns
- **Gesture-Based Braking**: Show an open palm with either hand to activate the brake
- **Auto-Acceleration**: Automatically accelerates when both hands are detected (like holding the wheel)
- **Real-Time Visual Feedback**: See hand landmarks, steering angle, and gesture classifications on screen
- **Start/Stop Control**: Toggle detection on/off without closing the app
- **Smooth Steering**: Moving average filter prevents jittery movements
- **Privacy-First**: All processing happens locally—no data leaves your computer

---

## 🎮 How It Works

1. **Position your hands** in front of your webcam as if holding a steering wheel
2. **Tilt your hands** to steer:
   - Left hand higher = Steer LEFT
   - Right hand higher = Steer RIGHT
   - Both hands level = Drive STRAIGHT
3. **Show an open palm** (either hand) to brake
4. **Auto-accelerate** when both hands are detected

The system calculates the angle between your hands and translates it into keyboard inputs (WASD keys) that Roblox recognizes.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drivable.git
   cd drivable
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Usage

1. **Launch the application** using the command above
2. **Position yourself** so your webcam can see both hands clearly
3. **Press 's'** to START detection and keyboard input
4. **Press 'x'** to STOP detection (releases all keys)
5. **Press 'q'** to QUIT the application

**Controls:**
- `s` - Start detection and keyboard input
- `x` - Stop detection and release all keys
- `q` - Quit application

**Hand Gestures:**
- Two hands detected → Steering control active
- Left hand higher → Steer LEFT (A key)
- Right hand higher → Steer RIGHT (D key)
- Hands level → Drive STRAIGHT
- Open palm → BRAKE (S key)
- Two hands present (not braking) → AUTO-ACCELERATE (W key)

---

## 🛠️ Technical Details

### Architecture

The project consists of three main modules:

#### 1. **HandDetector** (`hand_detector.py`)
- MediaPipe Hands integration for real-time tracking
- Gesture recognition (open palm, fist detection)
- Hand center calculation for steering
- Angle computation between hand positions
- Visual overlay rendering

#### 2. **KeyboardController** (`keyboard_controller.py`)
- Keyboard input simulation via pynput
- State management for key press/release
- WASD key mapping for Roblox games
- Conflict prevention and cleanup

#### 3. **HandSteeringApp** (`hand_steering_app.py`)
- Main application coordinator
- Camera feed management
- Steering angle smoothing (10-frame moving average)
- Start/stop state control
- Real-time frame processing pipeline

### Steering Algorithm

```
1. Detect both hands and extract landmark positions
2. Identify left/right hands based on x-coordinate
3. Calculate vertical height difference (y-coordinates)
4. Compute angle: atan2(height_diff, horizontal_distance)
5. Apply threshold (12.5°) for "straight" dead zone
6. Calculate excess angle for proportional steering
7. Apply moving average filter for smoothing
8. Map to keyboard input (A/D keys)
```

### Technologies Used

- **Python 3.x** - Core programming language
- **MediaPipe 0.10.0+** - Hand tracking and landmark detection
- **OpenCV 4.8.0+** - Camera access and image processing
- **pynput 1.7.6+** - Keyboard input simulation
- **NumPy 1.24.0+** - Numerical computations

---

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for excellent hand tracking
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [pynput](https://pynput.readthedocs.io/) for cross-platform input control
- The Roblox community for inspiration

---

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Reach out to the maintainers

---

## ⚠️ Disclaimer

This project is for educational and entertainment purposes. Use responsibly and in accordance with Roblox's Terms of Service. The developers are not responsible for any account actions taken by Roblox as a result of using this tool.

---

**Made with ❤️ and Python**

*Drive safely, even virtually!* 🚗💨
