# Advanced Bot Detection System with ML Fusion

A comprehensive web application that implements advanced bot detection using machine learning models, combining web log analysis and mouse movement patterns to identify automated behavior.

## 🏗️ Project Structure

```
CAPSTONE-main/
├── 📁 login_page/                 # Frontend React Application
│   ├── 📁 src/
│   │   ├── 📁 components/         # React Components
│   │   │   ├── LoginPage.jsx      # Main login interface
│   │   │   ├── Dashboard.jsx      # Post-login dashboard
│   │   │   ├── VisualCaptcha.jsx  # Gamified CAPTCHA system
│   │   │   ├── BotDetectionAlert.jsx # Bot detection notifications
│   │   │   ├── HoneypotAlert.jsx  # Honeypot trap alerts
│   │   │   └── MLDetectionMonitor.jsx # Real-time ML monitoring
│   │   ├── 📁 api/                # Backend API handlers
│   │   │   ├── botDetection.js    # ML bot detection endpoint
│   │   │   └── log.js             # Data logging endpoint
│   │   ├── 📁 logs/               # Session data storage
│   │   │   ├── mouse_movements.json
│   │   │   ├── web_logs.json
│   │   │   ├── behavior.json
│   │   │   └── login_attempts.json
│   │   └── 📁 utils/              # Utility functions
│   │       └── eventLogger.js     # Event tracking utilities
│   ├── server.js                  # Express.js backend server
│   ├── package.json               # Node.js dependencies
│   └── vite.config.js             # Vite build configuration
├── 📁 src/                        # Core ML Detection System
│   ├── 📁 core/                   # ML Detection Modules
│   │   ├── optimized_bot_detection.py # Fast ML processing
│   │   ├── web_log_detection_bot.py  # Web log analysis
│   │   ├── mouse_movements_detection_bot.py # Mouse pattern analysis
│   │   └── fusion.py              # Score fusion algorithm
│   └── 📁 utils/                  # ML utilities
│       └── session_processor.py   # Session data processing
├── 📁 models/                     # Pre-trained ML Models
│   ├── web_log_detector_comprehensive.pkl
│   └── mouse_movement_detector_comprehensive.h5
├── 📁 scripts/                    # Test and Demo Scripts
│   ├── main.py                    # Main demonstration script
│   ├── bot.py                     # Bot simulation script
│   ├── login_bot.py               # Login automation bot
│   └── run_demo.py                # Demo runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔄 System Workflow

### 1. **Frontend Data Collection**
The React frontend (`login_page/`) collects comprehensive user behavior data:

- **Mouse Movements**: Real-time tracking of cursor coordinates
- **Web Logs**: HTTP requests, page interactions, and navigation patterns
- **Behavior Signals**: Keystroke timing, scroll patterns, focus/blur events
- **Honeypot Traps**: Hidden form fields to catch automated tools

### 2. **Data Processing Pipeline**
```
User Interaction → Event Logging → Data Storage → ML Analysis → Decision
```

#### Data Collection Flow:
1. **Mouse Tracking**: Continuous coordinate logging with session IDs
2. **Event Logging**: All user interactions captured via `eventLogger.js`
3. **Behavior Analysis**: Keystroke intervals, click trustworthiness, scroll variance
4. **Storage**: JSON files in `login_page/src/logs/` for ML processing

### 3. **Multi-Layer Security Detection**

#### Core Components:

**A. reCAPTCHA v3 Integration**
- Google's invisible bot detection
- Score-based analysis (0.0 - 1.0)
- No user friction or challenges
- Real-time risk assessment
- Thresholds: High (0.7+), Medium (0.5+), Low (0.3+), Critical (0.1+)

**B. Web Log Detection** (`web_log_detection_bot.py`)
- Analyzes HTTP request patterns
- Features: request counts, status codes, timing patterns
- Model: Ensemble classifier (Random Forest, XGBoost)
- Output: Bot probability score (0-1)

**C. Mouse Movement Detection** (`mouse_movements_detection_bot.py`)
- CNN-based pattern recognition
- Input: 480x1320 normalized mouse movement matrices
- Features: Movement trajectories, acceleration, click patterns
- Output: Bot probability score (0-1)

**D. Intelligent Fusion** (`fusion.py`)
- Multi-layer decision fusion
- Combines reCAPTCHA + ML scores
- Logic:
  - If mouse score > 0.65 or < 0.35: Use mouse score only
  - Otherwise: Weighted average (60% mouse + 40% web log)
- Final threshold: 0.45 for bot classification
- Combined risk assessment with reCAPTCHA validation

### 4. **Real-time Processing**

#### Fast Detection Pipeline (`optimized_bot_detection.py`):
1. **Model Caching**: Singleton pattern for fast model loading
2. **Preprocessing**: Optimized feature extraction
3. **Parallel Processing**: Concurrent analysis of multiple data streams
4. **Result Fusion**: Intelligent score combination

### 5. **Security Response System**

#### Multi-layered Defense:
1. **reCAPTCHA v3**: Invisible Google bot detection
2. **Honeypot Detection**: Immediate CAPTCHA trigger
3. **ML Analysis**: Comprehensive behavior scoring
4. **Combined Analysis**: reCAPTCHA + ML fusion
5. **Adaptive CAPTCHA**: Difficulty based on combined risk
6. **Visual Indicators**: Real-time security status display

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** with ML libraries (TensorFlow, scikit-learn, pandas)
- **Node.js 16+** with npm
- **Modern web browser** with JavaScript enabled
- **Google reCAPTCHA v3** API keys (optional but recommended)

### Installation

#### 1. Backend Setup (Python ML System)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify model files exist
ls models/
# Should show:
# - logs_model.pkl
# - mouse_model.h5
```

#### 2. Frontend Setup (React Application)
```bash
cd login_page/

# Install Node.js dependencies
npm install

# Configure reCAPTCHA v3 (optional)
# Copy login_page/src/config/recaptcha.js and update with your keys
# Get keys from: https://www.google.com/recaptcha/admin

# Start development server
npm run dev
# This runs both Vite dev server and Express backend concurrently
```

#### 3. reCAPTCHA Enterprise Setup (Optional but Recommended)
```bash
# 1. Visit https://www.google.com/recaptcha/admin
# 2. Create a new site with reCAPTCHA Enterprise
# 3. Add your domain (localhost for development)
# 4. Copy the Site Key and API Key
# 5. Update login_page/src/config/recaptcha.js with your keys
# 6. Set environment variables:
#    REACT_APP_RECAPTCHA_SITE_KEY=your_site_key
#    REACT_APP_RECAPTCHA_API_KEY=your_api_key

# Enterprise API Configuration:
# - Project ID: endless-gamma-457506-a0
# - Site Key: 6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO
# - API Endpoint: https://recaptchaenterprise.googleapis.com/v1/projects/endless-gamma-457506-a0/assessments
```

### Running the System

#### Development Mode:
```bash
# Terminal 1: Start ML backend
cd scripts/
python main.py

# Terminal 2: Start web application
cd login_page/
npm run dev
```

#### Production Mode:
```bash
# Build and start production server
cd login_page/
npm run build
npm run server
```

## 🔧 Configuration

### ML Model Parameters
Edit `src/core/fusion.py` to adjust detection thresholds:
```python
high_threshold: float = 0.65    # High confidence mouse threshold
low_threshold: float = 0.35     # Low confidence mouse threshold
final_threshold: float = 0.45   # Final bot classification threshold
```

### reCAPTCHA v3 Configuration
Edit `login_page/src/config/recaptcha.js` to customize:
```javascript
scoreThresholds: {
  high: 0.7,      // High confidence human
  medium: 0.5,    // Medium confidence  
  low: 0.3,       // Low confidence - likely bot
  critical: 0.1   // Very likely bot
}
```

### Frontend Settings
Modify `login_page/src/components/LoginPage.jsx`:
- CAPTCHA difficulty levels
- Honeypot field configuration
- ML analysis triggers
- reCAPTCHA execution timing

## 📊 Testing and Validation

### Bot Simulation
```bash
# Run automated bot tests
python scripts/bot.py

# Test login automation
python scripts/login_bot.py

# Run comprehensive demo
python scripts/run_demo.py
```

### Manual Testing
1. **Human Behavior**: Normal mouse movements, realistic timing
2. **Bot Simulation**: Automated clicks, rapid movements
3. **Edge Cases**: Mixed behavior patterns

## 🔍 Key Features

### Advanced ML Detection
- **Triple Layer Architecture**: reCAPTCHA v3 + Web logs + Mouse movements
- **Intelligent Fusion**: Adaptive score combination with Google validation
- **Real-time Processing**: Sub-second detection
- **Model Caching**: Optimized performance

### Security Mechanisms
- **reCAPTCHA v3**: Google's invisible bot detection
- **Honeypot Traps**: Hidden form fields
- **Visual CAPTCHA**: Gamified verification
- **Behavior Analysis**: Keystroke timing, scroll patterns
- **Adaptive Responses**: Dynamic security levels

### User Experience
- **Modern UI**: Material-UI with dark theme
- **Real-time Feedback**: Live ML analysis status
- **Progressive Enhancement**: Graceful degradation
- **Responsive Design**: Mobile-friendly interface

## 🛡️ Security Considerations

### Data Privacy
- **Local Storage**: Session data stored locally
- **No External APIs**: Fully self-contained system
- **Anonymized Logs**: No personal data collection
- **Secure Transmission**: HTTPS in production

### Bot Evasion Resistance
- **Multiple Signals**: reCAPTCHA + ML patterns make evasion extremely difficult
- **Google Validation**: Leverages Google's massive bot detection database
- **Temporal Analysis**: Time-based behavior validation
- **Adaptive Thresholds**: Dynamic detection sensitivity
- **Triple Fusion Logic**: reCAPTCHA + ML + Behavioral analysis

## 🔬 Technical Details

### ML Model Architecture
- **reCAPTCHA v3**: Google's neural network with 0.0-1.0 scoring
- **Web Log Model**: Ensemble of Random Forest + XGBoost
- **Mouse Model**: CNN with 480x1320x1 input shape
- **Triple Fusion**: reCAPTCHA + ML decision-level combination with confidence weighting

### Performance Optimization
- **Model Caching**: Singleton pattern for memory efficiency
- **Batch Processing**: Parallel data analysis
- **Lazy Loading**: On-demand model initialization
- **Result Caching**: Avoid redundant computations

## 📈 Monitoring and Analytics

### Real-time Metrics
- **Detection Accuracy**: Bot vs Human classification rates
- **Processing Speed**: ML analysis timing
- **User Behavior**: Interaction patterns and trends
- **System Performance**: Resource utilization

### Log Analysis
- **Session Tracking**: Complete user journey mapping
- **Behavior Profiling**: Detailed interaction analysis
- **Security Events**: Honeypot triggers and CAPTCHA challenges
- **System Health**: Error rates and performance metrics

## 🤝 Contributing

### Development Workflow
1. **Feature Development**: Create feature branches
2. **Testing**: Comprehensive bot simulation tests
3. **Code Review**: ML model validation
4. **Documentation**: Update README and inline comments

### Code Standards
- **Python**: PEP 8 compliance, type hints
- **JavaScript**: ESLint configuration, modern ES6+
- **React**: Functional components, hooks
- **Documentation**: Comprehensive inline comments

## 📝 License

This project is developed for educational and research purposes. Please ensure compliance with applicable laws and regulations when implementing bot detection systems in production environments.

---

## 🎯 Quick Start Summary

1. **Install Dependencies**: `pip install -r requirements.txt && cd login_page && npm install`
2. **Start System**: `cd login_page && npm run dev`
3. **Access Application**: Open `http://localhost:3001`
4. **Test Detection**: Try both human and bot-like behavior patterns
5. **Monitor Results**: Check console logs and ML analysis results

The system provides a complete end-to-end bot detection solution with modern web interface and advanced machine learning capabilities.
