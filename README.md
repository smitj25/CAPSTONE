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
├── main.py                           # Main system integration
├── bot.py                            # Humanoid bot simulator (for testing)
├── dataset/                          # Training and test data
│   ├── phase1/                       # Phase 1 datasets
│   │   ├── D1/                       # Humans vs Moderate Bots
│   │   └── D2/                       # Humans vs Advanced Bots
│   └── phase2/                       # Phase 2 datasets
│       ├── D1/                       #  Humans vs Moderate & Advanced Bots
│       └── D2/                       # Humans vs Advanced Bots
├── web_log_detector_comprehensive.pkl    # Trained web log model
├── mouse_movement_detector_comprehensive.h5  # Trained mouse movement model
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip3 install -r requirements.txt
```

### Training the Models
```bash
# Train web log detection model (sequential training on all phases)
python3 web_log_detection_bot.py

# Train mouse movement detection model (sequential training on all phases)
python3 mouse_movements_detection_bot.py
```

### Running the Complete System
```bash
# Run the main system (trains models and demonstrates fusion)
python3 main.py

# Test fusion logic with example scenarios
python3 fusion.py
```

## 🔧 Usage Examples

### 1. Complete System Usage (Recommended)

**Using the Main System:**
```python
# Run the complete system with all components
python3 main.py
```

The main system will:
- Train web log detection model on all phases
- Train mouse movement detection model on all phases
- Demonstrate fusion logic with example scenarios
- Show example session detection results

### 2. Individual Module Usage

**Web Log Detection:**
```python
from web_log_detection_bot import WebLogDetectionBot

# Initialize and train
detector = WebLogDetectionBot()
detector.train_sequentially()  # Trains on all phases sequentially

# Predict on new data
score = detector.predict(web_log_features)
```

**Mouse Movement Detection:**
```python
from mouse_movements_detection_bot import MouseMovementDetectionBot

# Initialize and train
detector = MouseMovementDetectionBot()
detector.train_sequentially()  # Trains on all phases sequentially

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
print(f"Fusion Method: {result['fusion_method']}")
```

### 3. Complete Workflow Example

**Option 1: Using Main System (Simplest)**
```bash
# Run everything with one command
python3 main.py
```

**Option 2: Manual Step-by-Step**
```python
# Step 1: Train web log detection model
from web_log_detection_bot import WebLogDetectionBot
web_log_detector = WebLogDetectionBot()
web_log_detector.train_sequentially()

# Step 2: Train mouse movement detection model
from mouse_movements_detection_bot import MouseMovementDetectionBot
mouse_detector = MouseMovementDetectionBot()
mouse_detector.train_sequentially()

# Step 3: Use fusion for final classification
from fusion import BotDetectionFusion
fusion = BotDetectionFusion(
    web_log_model_path='web_log_detector_comprehensive.pkl',
    mouse_movement_model_path='mouse_movement_detector_comprehensive.h5'
)

# Step 4: Detect bot in a session
result = fusion.process_session(
    mouse_score=0.75,
    web_log_score=0.60
)
```

## 📊 Training Data

The system uses a comprehensive dataset with two phases:

### Phase 1 (Initial Training)
- **D1**: Humans vs Moderate Bots
  - Web logs: `access_1.log` to `access_5.log` (humans) + `access_moderate_bots.log`
  - Mouse movements: JSON files with session data
- **D2**: Humans vs Advanced Bots
  - Web logs: `access_1.log` to `access_5.log` (humans) + `access_advanced_bots.log`
  - Mouse movements: JSON files with session data

### Phase 2 (Incremental Training)  
- **D1**: Humans vs Moderate & Advanced Bots
  - Web logs: Multiple human files + `access_moderate_and_advanced_bots.log`
  - Mouse movements: URL sequence data
- **D2**: Humans vs Advanced Bots
  - Web logs: Multiple human files + `access_moderate_and_advanced_bots.log`
  - Mouse movements: URL sequence data

### Data Sources
- **Web Logs**: Apache server logs with session IDs and comprehensive request data
- **Mouse Movements**: JavaScript-collected mouse movement sequences and URL patterns
- **Annotations**: Ground truth labels for training and evaluation

## 🎯 Fusion Logic

The fusion module implements intelligent decision-making:

```
IF mouse_score > 0.7 OR mouse_score < 0.3:
    final_score = mouse_score  # High confidence mouse movement
    fusion_method = "mouse_only"
ELSE:
    final_score = 0.5 × mouse_score + 0.5 × web_log_score  # Weighted average
    fusion_method = "weighted_average"

IF final_score > 0.5:
    classification = "BOT"
ELSE:
    classification = "HUMAN"
```

## 📈 Performance

The system achieves robust performance through:

1. **Sequential Training**: Models train on Phase 1 first, then incrementally on Phase 2
2. **Ensemble Methods**: Multiple classifiers (SVM, Random Forest, AdaBoost, MLP) for web log analysis
3. **CNN Architecture**: Deep learning for mouse movement pattern recognition
4. **Intelligent Fusion**: Decision-level combination of signals with adaptive thresholds
5. **Majority Voting**: For mouse movement matrices within sessions
6. **Single Model Files**: Each component saves one comprehensive model after all training phases

## 🔍 Key Features

- **Robust Detection**: Harder for advanced bots to evade due to dual-signal approach
- **Sequential Learning**: Incremental training across phases preserves learned features
- **Modular Design**: Each component can be used independently or together
- **Comprehensive Evaluation**: Performance metrics for all components and fusion
- **Production Ready**: Single model files for easy deployment
- **Research Paper Implementation**: Follows the exact approach described in the paper

## 🛠️ Development

### Adding New Features
1. **Web Log Features**: Modify `extract_features()` in `web_log_detection_bot.py`
2. **Mouse Movement Features**: Modify matrix generation in `mouse_movements_detection_bot.py`
3. **Fusion Logic**: Adjust thresholds and weights in `fusion.py`

### Testing
```bash
# Test the complete system (recommended)
python3 main.py

# Test individual components
python3 web_log_detection_bot.py
python3 mouse_movements_detection_bot.py
python3 fusion.py
```

### Model Files
- **Web Log Model**: `web_log_detector_comprehensive.pkl` (contains model, scaler, and selected features)
- **Mouse Movement Model**: `mouse_movement_detector_comprehensive.h5` (Keras CNN model)

## 📝 Research Paper Implementation

This system implements the approach described in the research paper:

1. **Session Extraction**: PHP session IDs from Apache web logs
2. **Feature Engineering**: 19 comprehensive web log features + mouse movement matrices
3. **Model Training**: Ensemble classifier + CNN with sequential learning across phases
4. **Decision Fusion**: Intelligent combination of detection signals with adaptive thresholds
5. **Evaluation**: Comprehensive metrics across all datasets (D1, D2, Phase 1, Phase 2)

## 🎯 System Status

✅ **Web Log Detection**: Fully implemented and tested
✅ **Mouse Movement Detection**: Fully implemented and tested  
✅ **Fusion Module**: Fully implemented and tested
✅ **Sequential Training**: Working across all phases
✅ **Model Persistence**: Single comprehensive model files
✅ **Documentation**: Complete with examples

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
2. Review the example usage in the README
3. Examine the test outputs for debugging information
4. Check the `.gitignore` file for excluded files

---

**Note**: This system is designed for research and educational purposes. The models should be retrained with your specific data for production use.
