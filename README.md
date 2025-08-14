# Comprehensive Bot Detection System

This project implements a robust bot detection system based on the research paper's approach, combining web log analysis and mouse movement patterns with intelligent fusion for enhanced detection accuracy.

## 🎯 Overview

The system consists of three main components that work together to provide comprehensive bot detection:

1. **Web Log Detection** (`web_log_detection_bot.py`) - Analyzes server logs for bot patterns
2. **Mouse Movement Detection** (`mouse_movements_detection_bot.py`) - Analyzes mouse movement patterns using CNN
3. **Fusion Module** (`fusion.py`) - Combines both signals using decision-level fusion

## 🏗️ Architecture

### Component 1: Web Log Detection
- **Purpose**: Analyzes Apache web server logs to extract session-based features
- **Method**: Ensemble classifier (SVM, Random Forest, AdaBoost, MLP)
- **Features**: HTTP requests, status codes, content types, browsing behavior
- **Output**: Bot probability score (0-1)

### Component 2: Mouse Movement Detection  
- **Purpose**: Analyzes mouse movement patterns to detect non-human behavior
- **Method**: Convolutional Neural Network (CNN)
- **Input**: Mouse movement matrices (spatial-temporal data)
- **Output**: Bot probability score (0-1)

### Component 3: Fusion Module
- **Purpose**: Combines scores from both detection modules
- **Method**: Decision-level fusion with intelligent thresholds
- **Logic**: 
  - If mouse score > 0.7 or < 0.3: Use mouse score only
  - Otherwise: Weighted average (0.5 × mouse + 0.5 × web_log)
- **Output**: Final bot classification and confidence score

## 📁 Project Structure

```
CAPSTONE/
├── web_log_detection_bot.py          # Web log analysis module
├── mouse_movements_detection_bot.py  # Mouse movement analysis module  
├── fusion.py                         # Score fusion module
├── main_bot_detection_system.py      # Main system integration
├── bot.py                           # Humanoid bot simulator (for testing)
├── dataset/                         # Training and test data
│   ├── phase1/                      # Phase 1 datasets
│   │   ├── D1/                      # Humans vs Moderate Bots
│   │   └── D2/                      # Humans vs Advanced Bots
│   └── phase2/                      # Phase 2 datasets
│       ├── D1/                      # Humans vs Moderate & Advanced Bots
│       └── D2/                      # Humans vs Advanced Bots
├── web_log_detector_comprehensive.pkl    # Trained web log model
├── mouse_movement_detector_comprehensive.h5  # Trained mouse movement model
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn tensorflow selenium
```

### Training the Models
```bash
# Train web log detection model
python3 web_log_detection_bot.py

# Train mouse movement detection model  
python3 mouse_movements_detection_bot.py
```

### Running the Complete System
```bash
# Run the main system (trains models and demonstrates fusion)
python3 main_bot_detection_system.py

# Test fusion logic only
python3 fusion.py
```

## 🔧 Usage Examples

### 1. Individual Module Usage

**Web Log Detection:**
```python
from web_log_detection_bot import WebLogDetectionBot

# Initialize and train
detector = WebLogDetectionBot()
detector.train_sequentially()  # Trains on all phases

# Predict on new data
score = detector.predict(web_log_features)
```

**Mouse Movement Detection:**
```python
from mouse_movements_detection_bot import MouseMovementDetectionBot

# Initialize and train
detector = MouseMovementDetectionBot()
detector.train_sequentially()  # Trains on all phases

# Predict on new data
score = detector.predict_session(mouse_matrices)
```

### 2. Fusion Usage

```python
from fusion import BotDetectionFusion

# Initialize fusion with trained models
fusion = BotDetectionFusion(
    web_log_model_path='web_log_detector_comprehensive.pkl',
    mouse_movement_model_path='mouse_movement_detector_comprehensive.h5'
)

# Process a session
result = fusion.process_session(
    mouse_score=0.85,      # High bot probability from mouse movements
    web_log_score=0.45     # Moderate bot probability from web logs
)

print(f"Final Classification: {'BOT' if result['is_bot'] else 'HUMAN'}")
print(f"Confidence Score: {result['final_score']:.3f}")
```

### 3. Complete System Usage

```python
from main_bot_detection_system import ComprehensiveBotDetectionSystem

# Initialize the complete system
system = ComprehensiveBotDetectionSystem()

# Train all models
system.train_all_models()

# Detect bot in a session
result = system.detect_bot_session(
    session_id="session_001",
    mouse_score=0.75,
    web_log_score=0.60
)
```

## 📊 Training Data

The system uses a comprehensive dataset with two phases:

### Phase 1 (Initial Training)
- **D1**: Humans vs Moderate Bots
- **D2**: Humans vs Advanced Bots

### Phase 2 (Incremental Training)  
- **D1**: Humans vs Moderate & Advanced Bots
- **D2**: Humans vs Advanced Bots

### Data Sources
- **Web Logs**: Apache server logs with session IDs
- **Mouse Movements**: JavaScript-collected mouse movement sequences
- **Annotations**: Ground truth labels for training and evaluation

## 🎯 Fusion Logic

The fusion module implements intelligent decision-making:

```
IF mouse_score > 0.7 OR mouse_score < 0.3:
    final_score = mouse_score  # High confidence mouse movement
ELSE:
    final_score = 0.5 × mouse_score + 0.5 × web_log_score  # Weighted average

IF final_score > 0.5:
    classification = "BOT"
ELSE:
    classification = "HUMAN"
```

## 📈 Performance

The system achieves robust performance through:

1. **Sequential Training**: Models train on Phase 1, then incrementally on Phase 2
2. **Ensemble Methods**: Multiple classifiers for web log analysis
3. **CNN Architecture**: Deep learning for mouse movement patterns
4. **Intelligent Fusion**: Decision-level combination of signals
5. **Majority Voting**: For mouse movement matrices within sessions

## 🔍 Key Features

- **Robust Detection**: Harder for advanced bots to evade
- **Sequential Learning**: Incremental training across phases
- **Modular Design**: Each component can be used independently
- **Comprehensive Evaluation**: Performance metrics for all components
- **Production Ready**: Single model files for deployment

## 🛠️ Development

### Adding New Features
1. **Web Log Features**: Modify `extract_features()` in `web_log_detection_bot.py`
2. **Mouse Movement Features**: Modify matrix generation in `mouse_movements_detection_bot.py`
3. **Fusion Logic**: Adjust thresholds and weights in `fusion.py`

### Testing
```bash
# Test individual components
python3 web_log_detection_bot.py
python3 mouse_movements_detection_bot.py
python3 fusion.py

# Test complete system
python3 main_bot_detection_system.py
```

## 📝 Research Paper Implementation

This system implements the approach described in the research paper:

1. **Session Extraction**: PHP session IDs from web logs
2. **Feature Engineering**: 19 web log features + mouse movement matrices
3. **Model Training**: Ensemble + CNN with sequential learning
4. **Decision Fusion**: Intelligent combination of detection signals
5. **Evaluation**: Comprehensive metrics across all datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is for academic research purposes. Please cite the original research paper if using this implementation.

## 🆘 Support

For issues and questions:
1. Check the documentation in each module
2. Review the example usage in `main_bot_detection_system.py`
3. Examine the test outputs for debugging information

---

**Note**: This system is designed for research and educational purposes. The models should be retrained with your specific data for production use.
