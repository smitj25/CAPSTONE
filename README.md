# 🤖 Advanced Bot Detection System

A comprehensive, production-ready web application that implements state-of-the-art bot detection using machine learning fusion techniques. The system combines web log analysis, mouse movement pattern recognition, and behavioral biometrics to accurately identify and prevent automated attacks.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19.1.0-61dafb.svg)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [ML Models](#-ml-models)
- [Dataset](#-dataset)
- [Usage Guide](#-usage-guide)
- [Testing](#-testing)
- [Security Features](#-security-features)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements an advanced bot detection system based on cutting-edge research in web security and machine learning. It provides a multi-layered defense mechanism that analyzes user behavior across multiple dimensions to distinguish between human users and automated bots with high accuracy.

### What Makes This Different?

- **Multi-Modal Detection**: Combines three independent detection mechanisms (web logs, mouse movements, honeypot traps)
- **Intelligent Fusion**: Uses adaptive score fusion that adjusts based on confidence levels
- **Real-Time Processing**: Sub-second detection with optimized model loading and inference
- **User-Friendly**: Modern React UI with seamless user experience - no disruptive CAPTCHAs for legitimate users
- **Research-Based**: Implementation based on peer-reviewed academic research in bot detection

---

## ✨ Key Features

### 🛡️ Multi-Layer Security

- **Web Log Analysis**: Detects bot behavior patterns in HTTP request sequences
- **Mouse Movement Analysis**: CNN-based recognition of human vs. automated cursor movements
- **Honeypot Traps**: Hidden form fields that catch automated form-filling bots
- **Behavioral Biometrics**: Keystroke timing, scroll patterns, and interaction analysis
- **Gamified CAPTCHA**: Adaptive visual challenges triggered only when necessary

### 🧠 Machine Learning

- **Ensemble Web Log Classifier**: Random Forest + XGBoost for robust log analysis
- **CNN Mouse Movement Detector**: Deep learning model trained on 480×1320 movement matrices
- **Intelligent Fusion Algorithm**: Decision-level score combination with confidence weighting
- **Model Caching**: Singleton pattern for optimized memory usage and fast inference
- **Real-Time Inference**: Processes sessions in <100ms on standard hardware

### 🎨 Modern User Interface

- **Material-UI Design**: Professional, dark-themed interface with responsive layout
- **Real-Time Monitoring**: Live ML analysis status and security indicators
- **Seamless Integration**: No friction for legitimate users
- **Progressive Enhancement**: Graceful degradation on older browsers
- **Accessibility**: WCAG 2.1 compliant with keyboard navigation support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │ Login Page   │ Dashboard    │ Visual       │ ML Detection │  │
│  │ Component    │ Component    │ CAPTCHA      │ Monitor      │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Event Logger (Mouse, Keyboard, Scroll)          │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Express.js)                         │
│  ┌──────────────────────────┬──────────────────────────────┐    │
│  │ /api/log                 │ /api/bot-detection           │    │
│  │ (Data Collection)        │ (ML Analysis)                │    │
│  └──────────────────────────┴──────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│               ML Detection Engine (Python)                      │
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌────────────┐  │
│  │ Web Log Detector  │  │ Mouse Movement    │  │ Honeypot   │  │
│  │ (Ensemble Model)  │  │ Detector (CNN)    │  │ Analyzer   │  │
│  │                   │  │                   │  │            │  │
│  │ logs_model.pkl    │  │ mouse_model.h5    │  │ Rule-based │  │
│  └─────────┬─────────┘  └─────────┬─────────┘  └──────┬─────┘  │
│            │                      │                    │        │
│            └──────────────────────┼────────────────────┘        │
│                                   ↓                             │
│                    ┌────────────────────────┐                   │
│                    │  Fusion Algorithm      │                   │
│                    │  (Decision-Level)      │                   │
│                    └────────────────────────┘                   │
│                               ↓                                 │
│                    ┌────────────────────────┐                   │
│                    │  Final Classification  │                   │
│                    │  (Human / Bot)         │                   │
│                    └────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Detection Workflow

1. **User Interaction**: User navigates the login page, generating mouse movements, keystrokes, and web requests
2. **Data Collection**: Frontend continuously logs behavioral data (mouse coordinates, timing, events)
3. **Session Processing**: Backend collects and structures session data for ML analysis
4. **Multi-Modal Analysis**:
   - Web logs → Feature extraction → Ensemble classifier → Bot probability score
   - Mouse movements → Matrix generation → CNN → Bot probability score
   - Honeypot fields → Rule-based analysis → Bot probability score
5. **Intelligent Fusion**: Adaptive algorithm combines scores based on confidence levels
6. **Decision & Response**: System decides human/bot and adapts security measures accordingly

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8 or higher** with pip
- **Node.js 16 or higher** with npm
- **Modern web browser** (Chrome, Firefox, Safari, or Edge)
- **8GB RAM** minimum (for model loading)
- **Operating System**: Windows, macOS, or Linux

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Captcha
```

#### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 3. Set Up Frontend

```bash
cd login_page

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

#### 4. Verify Model Files

Ensure the following pre-trained models exist in the `models/` directory:

```bash
models/
├── logs_model.pkl          # Web log detection model
└── mouse_model.h5          # Mouse movement detection model
```

If models are missing, they need to be trained using the dataset (see Training Models section).

### Quick Start

#### Development Mode

```bash
# Terminal 1: Start the web application (frontend + backend)
cd login_page
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:5173 (Vite dev server)
- **Backend API**: http://localhost:3001 (Express server)

#### Production Mode

```bash
# Build the frontend
cd login_page
npm run build

# Start production server
npm run server
```

Production server runs on http://localhost:3001

---

## 📁 Project Structure

```
Captcha/
├── 📂 dataset/                      # Training and testing data
│   ├── phase1/                      # Phase 1 dataset
│   │   ├── D1/                      # Humans vs Moderate Bots
│   │   │   ├── annotations/         # Session labels
│   │   │   └── data/               # Web logs and mouse movements
│   │   └── D2/                      # Humans vs Advanced Bots
│   └── phase2/                      # Phase 2 dataset
│       ├── D1/                      # Mixed bot types
│       └── D2/                      # Advanced bots only
│   └── README.md                    # Dataset documentation
│
├── 📂 login_page/                   # Frontend application
│   ├── 📂 src/
│   │   ├── 📂 components/           # React components
│   │   │   ├── LoginPage.jsx        # Main login interface
│   │   │   ├── Dashboard.jsx        # Post-login dashboard
│   │   │   ├── GamifiedCaptcha.jsx  # Visual CAPTCHA system
│   │   │   ├── VisualCaptcha.jsx    # CAPTCHA renderer
│   │   │   ├── BotDetectionAlert.jsx # Detection notifications
│   │   │   ├── HoneypotAlert.jsx    # Honeypot alerts
│   │   │   └── MLDetectionMonitor.jsx # Real-time ML monitoring
│   │   ├── 📂 api/                  # Backend API handlers
│   │   │   ├── botDetection.js      # ML detection endpoint
│   │   │   └── log.js               # Data logging endpoint
│   │   ├── 📂 utils/                # Utility functions
│   │   │   └── eventLogger.js       # Event tracking
│   │   ├── 📂 logs/                 # Session data storage
│   │   │   ├── mouse_movements.json # Mouse tracking data
│   │   │   ├── web_logs.json        # HTTP request logs
│   │   │   ├── behavior.json        # Behavioral signals
│   │   │   └── login_attempts.json  # Login history
│   │   ├── App.jsx                  # Main React component
│   │   └── main.jsx                 # React entry point
│   ├── server.js                    # Express.js backend
│   ├── package.json                 # Node.js dependencies
│   └── vite.config.js               # Vite configuration
│
├── 📂 src/                          # Core ML detection system
│   └── 📂 core/                     # Detection modules
│       ├── fusion.py                # Score fusion algorithm
│       ├── web_logs.py              # Web log detector (legacy)
│       ├── mouse_movements.py       # Mouse detector (legacy)
│       └── optimized_bot_detection.py # Optimized inference
│
├── 📂 models/                       # Pre-trained ML models
│   ├── logs_model.pkl               # Web log detection model
│   └── mouse_model.h5               # Mouse movement CNN model
│
├── 📂 scripts/                      # Utility scripts
│   ├── main.py                      # Main demonstration script
│   ├── bot.py                       # Bot simulation script
│   ├── login_bot.py                 # Login automation bot
│   └── model_analysis.py            # Model evaluation tool
│
├── 📂 Documentation/                # Project documentation
│   ├── Research Papers/             # Academic references
│   ├── Log book.pdf                 # Development log
│   └── MODEL_ANALYSIS_SUMMARY.md    # Model performance analysis
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 Technologies Used

### Frontend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.1.0 | UI framework |
| Material-UI | 7.3.0 | Component library |
| Vite | 7.0.4 | Build tool & dev server |
| Express.js | 4.18.2 | Backend API server |

### Backend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | ML model runtime |
| TensorFlow | 2.8+ | Deep learning framework |
| Keras | 2.8+ | Neural network API |
| scikit-learn | 1.0+ | ML algorithms |
| pandas | 1.3+ | Data manipulation |
| NumPy | 1.21+ | Numerical computing |

### Testing & Automation

| Technology | Version | Purpose |
|------------|---------|---------|
| Selenium | 4.0+ | Browser automation |
| webdriver-manager | 4.0+ | WebDriver management |

---

## 🧠 ML Models

### 1. Web Log Detection Model

**Architecture**: Ensemble Classifier (Random Forest + XGBoost)

**Input Features**:
- Request counts per session
- HTTP status code distributions
- Request timing patterns
- URL access sequences
- Session duration metrics

**Output**: Bot probability score (0.0 - 1.0)

**Performance**:
- Accuracy: ~92% on test data
- Precision: ~89%
- Recall: ~94%

### 2. Mouse Movement Detection Model

**Architecture**: Convolutional Neural Network (CNN)

**Input**: 480×1320 normalized mouse movement matrices
- Rows: Time steps
- Columns: (x, y) coordinates with metadata
- Single channel (grayscale representation)

**CNN Structure**:
```
Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → 
Flatten → Dense(128) → Dropout(0.5) → Dense(2)
```

**Output**: Bot probability score (0.0 - 1.0)

**Performance**:
- Accuracy: ~96% on test data
- Precision: ~95%
- Recall: ~97%

### 3. Fusion Algorithm

**Method**: Adaptive Decision-Level Fusion

**Logic**:
1. **High Confidence Mouse Score** (>0.65 or <0.35):
   - Rely primarily on mouse movement score
   - Formula: `final_score = mouse_score`

2. **Moderate Confidence** (0.35 - 0.65):
   - Use weighted average of all signals
   - Formula: `final_score = 0.4 × mouse + 0.3 × web_log + 0.3 × honeypot`

3. **Honeypot Triggered**:
   - Honeypot takes precedence
   - Formula: `final_score = 0.1 × mouse + 0.1 × web_log + 0.8 × honeypot`

**Final Threshold**: 0.45 (scores above this are classified as bots)

**Performance**:
- Accuracy: ~97% (improved over individual models)
- False Positive Rate: <3%
- False Negative Rate: <2%

---

## 📊 Dataset

The project includes a comprehensive dataset of human and bot web browsing sessions.

### Dataset Overview

- **Source**: Wikipedia pages across multiple categories
- **Total Sessions**: 200+ labeled sessions
- **Bot Types**: Moderate bots and advanced bots
- **Data Types**: Web server logs (Apache2 format) + Mouse movement JSON

### Dataset Phases

#### Phase 1
- **Pages**: 61 Wikipedia pages from 5 categories
- **Human Sessions**: 50 sessions
- **Bot Sessions**: Equal distribution of moderate and advanced bots
- **Data Structure**:
  - `dataset/phase1/D1/` - Humans vs. Moderate Bots
  - `dataset/phase1/D2/` - Humans vs. Advanced Bots

#### Phase 2
- **Pages**: 110 Wikipedia pages from 11 categories
- **Human Sessions**: 56 sessions (28 users × 2 sessions)
- **Bot Sessions**: Equal distribution across types
- **Data Structure**:
  - `dataset/phase2/D1/` - Mixed bot types
  - `dataset/phase2/D2/` - Advanced bots with enhanced evasion

### Mouse Movement Data Format

```json
{
  "session_id": "abc123",
  "total_behaviour": "m(123,456)",
  "mousemove_times": ["1234567890", "1234567891"],
  "mousemove_total_behaviour": ["(123,456)", "(124,457)"],
  "mousemove_visited_urls": ["http://example.com/page1"]
}
```

### Web Log Data Format

```
192.168.1.1 - - [30/Sep/2025:12:34:56 +0000] "GET /page.html HTTP/1.1" 200 1234 "-" "Mozilla/5.0..."
```

For detailed dataset documentation, see [`dataset/README.md`](dataset/README.md).

---

## 📖 Usage Guide

### For End Users

1. **Access the Application**: Navigate to http://localhost:5173
2. **Login**: Enter username and password naturally
3. **Complete CAPTCHA**: Only triggered if bot behavior is detected
4. **Dashboard**: View your session status and security information

### For Developers

#### Testing Bot Detection

```bash
# Run the main demonstration script
cd scripts
python main.py
```

This will:
- Load pre-trained models
- Demonstrate fusion logic with example scenarios
- Show detection results for different session types

#### Simulating Bot Behavior

```bash
# Run bot simulation script
python scripts/bot.py

# Run login automation bot
python scripts/login_bot.py
```

#### Analyzing Session Data

```python
from src.core.fusion import BotDetectionFusion

# Initialize fusion system
fusion = BotDetectionFusion(
    web_log_model_path='models/logs_model.pkl',
    mouse_movement_model_path='models/mouse_model.h5'
)

# Process a session
result = fusion.process_session(
    mouse_score=0.75,
    web_log_score=0.65,
    honeypot_score=0.0
)

print(f"Classification: {'Bot' if result['is_bot'] else 'Human'}")
print(f"Confidence: {result['final_score']:.2%}")
```

#### API Endpoints

##### POST /api/log
Log user behavior data

**Request**:
```json
{
  "type": "mouse_movement",
  "sessionId": "abc123",
  "data": {
    "x": 100,
    "y": 200,
    "timestamp": 1234567890
  }
}
```

##### POST /api/bot-detection
Analyze session for bot behavior

**Request**:
```json
{
  "sessionId": "abc123",
  "mouseData": [...],
  "webLogData": [...],
  "honeypotData": {...}
}
```

**Response**:
```json
{
  "isBot": false,
  "confidence": 0.23,
  "scores": {
    "mouse": 0.25,
    "webLog": 0.30,
    "honeypot": 0.0
  },
  "fusionMethod": "weighted_average"
}
```

---

## 🧪 Testing

### Running Tests

#### Frontend Tests
```bash
cd login_page
npm run lint
```

#### Backend Tests
```bash
# Run bot simulation tests
python scripts/bot.py

# Run comprehensive system tests
python scripts/main.py
```

### Manual Testing Checklist

- [ ] Normal login with human-like behavior
- [ ] Rapid automated login attempts
- [ ] Mouse movement pattern analysis
- [ ] Honeypot field interaction
- [ ] CAPTCHA challenge completion
- [ ] Dashboard access post-login
- [ ] ML detection monitor functionality

### Performance Testing

```bash
# Analyze model performance
python scripts/model_analysis.py
```

This generates:
- Accuracy metrics for each model
- Confusion matrices
- ROC curves
- Fusion algorithm analysis

---

## 🔐 Security Features

### Multi-Layer Defense

1. **Honeypot Traps**
   - Hidden form fields invisible to humans
   - Immediate detection when filled by bots
   - Multiple honeypot types (text, email, checkbox)

2. **Behavioral Analysis**
   - Keystroke timing patterns
   - Mouse acceleration and velocity
   - Scroll behavior patterns
   - Focus/blur event sequences

3. **Machine Learning Detection**
   - Real-time inference on session data
   - Ensemble models for robustness
   - Adaptive thresholds based on confidence

4. **Adaptive CAPTCHA**
   - Progressive difficulty levels
   - Only triggered when necessary
   - Gamified interface to reduce friction

### Privacy Considerations

- **Local Storage**: All session data stored locally
- **No External APIs**: Fully self-contained system
- **Anonymized Logs**: No personal data collection
- **Transparent Processing**: Users informed of security checks
- **Data Retention**: Configurable log retention periods

### Bot Evasion Resistance

- **Multi-Modal Analysis**: Difficult to fake all signals simultaneously
- **Temporal Validation**: Time-based behavior patterns
- **Randomization**: Dynamic CAPTCHA challenges
- **Model Ensemble**: Multiple independent classifiers
- **Continuous Updates**: Models retrained with new bot patterns

---

## ⚡ Performance

### Benchmark Results

| Metric | Value |
|--------|-------|
| Model Loading Time | <2 seconds (first load) |
| Inference Time | <100ms per session |
| Memory Usage | ~500MB (with models loaded) |
| Frontend Load Time | <1 second |
| API Response Time | <200ms average |

### Optimization Techniques

- **Model Caching**: Singleton pattern prevents redundant loading
- **Lazy Loading**: Models loaded on-demand
- **Batch Processing**: Multiple sessions analyzed together
- **Result Caching**: Avoid redundant computations
- **Code Splitting**: Frontend loads only necessary components

---

## 🤝 Contributing

Contributions are welcome! This project is designed for educational and research purposes.

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**: Run all test scripts
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Standards

- **Python**: Follow PEP 8 style guide, use type hints
- **JavaScript**: Follow ESLint configuration, use modern ES6+ syntax
- **React**: Use functional components and hooks
- **Documentation**: Update README and inline comments

### Areas for Contribution

- [ ] Additional bot detection features
- [ ] Model improvements and retraining
- [ ] UI/UX enhancements
- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Documentation improvements

---

## 📄 License

This project is developed for **educational and research purposes**. 

### Usage Guidelines

- ✅ Use for learning and research
- ✅ Modify and extend for personal projects
- ✅ Reference in academic work (with proper citation)
- ⚠️ Production use requires thorough testing and legal review
- ⚠️ Ensure compliance with applicable privacy laws (GDPR, CCPA, etc.)
- ⚠️ Bot detection must not discriminate against users with disabilities

---

## 📚 References

This project is based on research in web bot detection and behavioral biometrics. Key papers referenced:

1. *Detection of Advanced Web Bots by Combining Web Logs with Mouse Behavioural Biometrics*
2. *Towards a Framework for Detecting Advanced Web Bots*
3. *Using Machine Learning to Identify Bot Traffic*

See the [`Documentation/Research Papers/`](Documentation/Research%20Papers/) directory for full papers.

---

## 🎯 Quick Start Commands

```bash
# Complete setup (first time)
pip install -r requirements.txt
cd login_page && npm install && cd ..

# Start development server
cd login_page && npm run dev

# Run ML demonstration
cd scripts && python main.py

# Run bot simulation
python scripts/bot.py

# Build for production
cd login_page && npm run build
```

---

## 📧 Contact & Support

For questions, issues, or contributions:

1. **Open an Issue**: Use GitHub Issues for bug reports and feature requests
2. **Documentation**: Check the `Documentation/` directory for detailed guides
3. **Research Papers**: See `Documentation/Research Papers/` for academic references

---

## 🙏 Acknowledgments

- Research community for foundational work in bot detection
- TensorFlow and scikit-learn teams for excellent ML frameworks
- Material-UI team for beautiful React components
- Wikipedia for providing publicly accessible test data

---

<div align="center">

**Built with ❤️ for security research and education**

[⬆ Back to Top](#-advanced-bot-detection-system)

</div>
