# 📁 Bot Detection System - Project Structure

## 🏗️ Directory Organization

```
CAPSTONE-main/
├── 📁 src/                          # Source code directory
│   ├── 📁 core/                     # Core bot detection modules
│   │   ├── web_log_detection_bot.py      # Web log analysis module
│   │   ├── mouse_movements_detection_bot.py  # Mouse movement analysis
│   │   ├── fusion.py                     # Score fusion module
│   │   └── optimized_bot_detection.py    # Optimized detection system
│   │
│   ├── 📁 utils/                    # Utility functions
│   │   └── session_processor.py          # Session data processing
│   │
│   ├── 📁 tests/                    # Test files and test data
│   │   ├── test.py                       # Main test suite
│   │   ├── simple_log_test.py            # Simple log testing
│   │   └── test.html                     # Test HTML page
│   │
│   ├── 📁 config/                   # Configuration files
│   │   └── (future config files)
│   │
│   └── 📁 docs/                     # Documentation
│       ├── TESTING_GUIDE.md              # Testing instructions
│       └── PERFORMANCE_OPTIMIZATION.md   # Performance guidelines
│
├── 📁 scripts/                      # Executable scripts
│   ├── run_demo.py                      # Demo script
│   ├── bot.py                           # Bot simulator
│   └── main.py                          # Main system runner
│
├── 📁 data/                         # Data storage
│   └── (future data files)
│
├── 📁 logs/                         # Log files
│   └── (runtime logs)
│
├── 📁 models/                       # Trained ML models
│   ├── web_log_detector_comprehensive.pkl
│   └── mouse_movement_detector_comprehensive.h5
│
├── 📁 login_page/                   # Web application
│   ├── 📁 src/
│   │   ├── 📁 api/                      # API endpoints
│   │   │   ├── botDetection.js              # Bot detection API
│   │   │   └── log.js                       # Logging API
│   │   ├── 📁 components/                  # React components
│   │   ├── 📁 logs/                        # Application logs
│   │   └── 📁 utils/                       # Frontend utilities
│   ├── server.js                        # Express server
│   ├── package.json                     # Node.js dependencies
│   └── vite.config.js                   # Vite configuration
│
├── 📁 .venv/                        # Python virtual environment
├── requirements.txt                  # Python dependencies
├── README.md                        # Main project documentation
├── .gitignore                       # Git ignore rules
└── chromedriver.exe                 # Chrome WebDriver
```

## 🔧 Module Dependencies

### Core Modules (`src/core/`)
- **web_log_detection_bot.py**: Analyzes web server logs for bot patterns
- **mouse_movements_detection_bot.py**: Analyzes mouse movement patterns using CNN
- **fusion.py**: Combines both detection signals using intelligent fusion
- **optimized_bot_detection.py**: High-performance detection system

### Utility Modules (`src/utils/`)
- **session_processor.py**: Processes session data and extracts features

### Test Modules (`src/tests/`)
- **test.py**: Comprehensive test suite for all detection modules
- **simple_log_test.py**: Simple log analysis testing
- **test.html**: Test page for bot simulation

### Scripts (`scripts/`)
- **run_demo.py**: Demonstrates system capabilities
- **bot.py**: Humanoid bot simulator for testing
- **main.py**: Main system integration and training

## 🚀 Quick Start Commands

### Development
```bash
# Run demo
python scripts/run_demo.py

# Run tests
python src/tests/test.py

# Start web application
cd login_page && npm run dev

# Run bot simulator
python scripts/bot.py
```

### Production
```bash
# Start optimized server
python scripts/start_optimized_server.py

# Run main system
python scripts/main.py
```

## 📊 Data Flow

1. **Input**: Web logs + Mouse movements
2. **Processing**: Feature extraction in utils/
3. **Detection**: Analysis in core/ modules
4. **Fusion**: Score combination in fusion.py
5. **Output**: Bot classification and confidence

## 🔍 Testing Strategy

- **Unit Tests**: Individual module testing
- **Integration Tests**: End-to-end system testing
- **Performance Tests**: Load and stress testing
- **Bot Simulation**: Realistic bot behavior testing

## 📈 Performance Monitoring

- **Processing Time**: Track detection speed
- **Accuracy Metrics**: Monitor detection accuracy
- **Resource Usage**: CPU and memory monitoring
- **Error Rates**: Track false positives/negatives

## 🛠️ Maintenance

- **Regular Updates**: Keep dependencies current
- **Model Retraining**: Update ML models with new data
- **Performance Optimization**: Continuous improvement
- **Security Updates**: Regular security patches

---

**This structure provides:**
- ✅ Clear separation of concerns
- ✅ Easy maintenance and updates
- ✅ Scalable architecture
- ✅ Comprehensive testing
- ✅ Professional organization


