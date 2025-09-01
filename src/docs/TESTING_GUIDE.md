# 🧪 Bot Detection System Testing Guide

This guide explains how to test your bot detection system against various types of bots and scenarios.

## 🚀 Quick Start Testing

### 1. **Demo Test** (Recommended for beginners)
```bash
python run_demo.py
```
This runs a quick demonstration showing:
- Fusion logic with different scenarios
- Sample data testing
- System readiness verification

### 2. **Log File Testing** (Test with existing data)
```bash
python test.py
```
This tests the system against real log files in `login_page/src/logs/`:
- `mouse_movements.json` - Mouse movement patterns
- `web_logs.json` - Web server logs
- `login_attempts.json` - Login session data
- `events.json` - User interaction events

## 🕷️ Bot Simulation Testing

### 3. **Built-in Bot Simulator**
```bash
python bot.py
```
This runs a humanoid bot that:
- Simulates human-like mouse movements
- Generates realistic web interactions
- Creates test data for analysis

### 4. **Live Web Application Testing**

#### Start the Server:
```bash
cd login_page
npm install  # First time only
npm start
```

#### Start the Backend:
```bash
node server.js
```

#### Test with Real Browser:
1. Open `http://localhost:3000` in your browser
2. Interact with the login form
3. Check the logs in `login_page/src/logs/`
4. Run bot detection analysis

## 🔬 Advanced Testing Methods

### 5. **Custom Bot Testing**

Create a custom test script:
```python
from fusion import BotDetectionFusion
import numpy as np
import pandas as pd

# Initialize fusion
fusion = BotDetectionFusion(
    web_log_model_path='models/web_log_detector_comprehensive.pkl',
    mouse_movement_model_path='models/mouse_movement_detector_comprehensive.h5'
)

# Test with custom data
web_log_features = pd.DataFrame([{
    'total_requests': 50,
    'unique_pages': 10,
    'avg_session_duration': 300,
    # ... more features
}])

mouse_movements = [np.random.rand(50, 50)]  # 50x50 movement matrix

# Run detection
result = fusion.process_session(web_log_features, mouse_movements)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")
```

### 6. **Performance Testing**

Test system performance:
```bash
python start_optimized_server.py
```

This runs optimized detection with:
- Caching for faster responses
- Rate limiting
- Performance monitoring

## 📊 Understanding Test Results

### Bot Detection Scores:
- **0.0 - 0.3**: Likely Human
- **0.3 - 0.7**: Uncertain (uses weighted fusion)
- **0.7 - 1.0**: Likely Bot

### Fusion Methods:
- **mouse_only**: High confidence mouse movement
- **web_log_only**: High confidence web log pattern
- **weighted_average**: Balanced approach

## 🎯 Testing Scenarios

### Scenario 1: Human vs Simple Bot
```bash
# Run demo to see basic scenarios
python run_demo.py
```

### Scenario 2: Advanced Bot Detection
```bash
# Test with real log data
python test.py
```

### Scenario 3: Live Interaction
```bash
# Start web app and interact manually
cd login_page && npm start
```

### Scenario 4: Performance Under Load
```bash
# Test optimized server
python start_optimized_server.py
```

## 🔍 Monitoring and Debugging

### Check Log Files:
- `login_page/src/logs/mouse_movements.json`
- `login_page/src/logs/web_logs.json`
- `login_page/src/logs/login_attempts.json`
- `login_page/src/logs/events.json`

### Debug Output:
- Look for `[SUCCESS]` and `[ERROR]` messages
- Check processing times
- Monitor fusion method selection

## 🛠️ Troubleshooting

### Common Issues:

1. **Model Loading Errors**:
   - Ensure models exist in `models/` directory
   - Check file permissions

2. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Node.js Dependencies**:
   ```bash
   cd login_page && npm install
   ```

4. **ChromeDriver Issues**:
   - Ensure `chromedriver.exe` is in project root
   - Update ChromeDriver if needed

## 📈 Performance Metrics

Monitor these metrics during testing:
- **Detection Accuracy**: % of correct classifications
- **Processing Time**: Time to analyze session
- **False Positives**: Humans classified as bots
- **False Negatives**: Bots classified as humans
- **Fusion Method Distribution**: Which method is used most

## 🎯 Best Practices

1. **Test Regularly**: Run tests after any code changes
2. **Use Multiple Scenarios**: Test different bot types
3. **Monitor Performance**: Watch for degradation
4. **Validate Results**: Check against known data
5. **Document Findings**: Keep track of test results

## 🚀 Next Steps

After testing:
1. Analyze results and identify areas for improvement
2. Adjust fusion thresholds if needed
3. Retrain models with new data
4. Deploy to production environment
5. Set up continuous monitoring

---

**Happy Testing! 🎉**
