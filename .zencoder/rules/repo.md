---
description: Repository Information Overview
alwaysApply: true
---

# Bot Detection System Information

## Summary
A comprehensive web bot detection system that combines mouse movement analysis and web log analysis to identify bots with high accuracy. The system uses a fusion approach that prioritizes mouse movement detection when confidence is high, and combines both detection methods when confidence is moderate.

## Structure
- **Root Directory**: Contains main Python modules and test files
- **Dataset Directory**: Referenced in code for storing training and testing data
- **.zencoder**: Configuration directory for Zencoder

## Language & Runtime
**Language**: Python
**Version**: Python 3.x
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0
- keras >= 2.8.0
- selenium (for bot simulation)

## Main Components
**Bot Detection Modules**:
- `bot.py`: Humanoid bot simulator for testing
- `web_log_detection_bot.py`: Analyzes web server logs for bot patterns
- `mouse_movements_detection_bot.py`: Analyzes mouse movement data using CNN
- `fusion.py`: Combines both detection methods with adaptive weighting

**Testing & Evaluation**:
- `test_bot_detection.py`: Test script for the detection system
- `train_and_evaluate.py`: Comprehensive training and evaluation system

## Build & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_bot_detection.py

# Train and evaluate models
python train_and_evaluate.py
```

## Testing
**Framework**: Built-in testing module
**Test Files**: 
- `test_bot_detection.py`: Tests individual components and integration
**Run Command**:
```bash
python test_bot_detection.py
```

## Model Architecture
**Mouse Movement Detection**:
- CNN with multiple convolutional and pooling layers
- Input shape: (480, 1320, 1) - cropped from 1080x1920 with 300px offset
- Binary classification (human vs bot)
- Currently supports training on either Phase 1 or Phase 2 datasets separately, not sequentially

**Web Log Detection**:
- Ensemble of SVC, MLP, Random Forest, and AdaBoost classifiers
- Features: request patterns, HTTP methods, response codes, file types, browsing behavior

**Fusion Mechanism**:
- Prioritizes mouse movement score when confidence is high (≥0.7) or low (≤0.3)
- Uses weighted average of both scores when confidence is moderate
- Classification threshold: 0.5 (above = bot, below = human)

## Dataset Structure
The code references a dataset with the following structure:
- Phase1/D1: Humans vs moderate bots
- Phase1/D2: Humans vs advanced bots
- Contains mouse movement data and web logs
- Includes annotations for training and testing

## Implementation Notes
- The mouse movement detection module (`mouse_movements_detection_bot.py`) currently allows training on either Phase 1 or Phase 2 datasets, but not both sequentially
- The `process_session_data` method accepts a `phase` parameter ('phase1' or 'phase2') to specify which dataset format to use
- Future improvements could include sequential training on both phases to improve model robustness
