import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  IconButton,
  LinearProgress,
  Fade,
  Zoom,
  Chip,
  Alert
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  TouchApp as TouchIcon,
  Visibility as EyeIcon,
  VisibilityOff as EyeOffIcon
} from '@mui/icons-material';

const GamifiedCaptcha = ({ onSuccess, onClose, difficulty = 'medium' }) => {
  const [currentGame, setCurrentGame] = useState(null);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(30);
  const [gameStatus, setGameStatus] = useState('playing'); // playing, success, failed
  const [selectedItems, setSelectedItems] = useState([]);
  const [showHint, setShowHint] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const [streak, setStreak] = useState(0);
  const [gameHistory, setGameHistory] = useState([]);
  
  const canvasRef = useRef(null);
  const timerRef = useRef(null);

  // Game configurations based on difficulty - Made much easier
  const gameConfigs = {
    easy: {
      timeLimit: 60,  // Increased from 45
      requiredScore: 2,  // Reduced from 3
      games: ['colorMatch', 'shapeCount']  // Only easiest games
    },
    medium: {
      timeLimit: 45,  // Increased from 30
      requiredScore: 3,  // Reduced from 4
      games: ['colorMatch', 'shapeCount', 'patternComplete']  // Removed memory sequence
    },
    hard: {
      timeLimit: 30,  // Increased from 20
      requiredScore: 4,  // Reduced from 5
      games: ['colorMatch', 'shapeCount', 'patternComplete', 'mathPuzzle']  // Removed memory sequence
    }
  };

  const config = gameConfigs[difficulty] || gameConfigs.medium;

  // Game data
  const gameData = {
    colorMatch: {
      name: "Color Match Challenge",
      description: "Click all items that match the target color",
      icon: "🎨",
      generate: () => {
        const colors = ['red', 'blue', 'green', 'yellow'];  // Reduced to 4 colors
        const targetColor = colors[Math.floor(Math.random() * colors.length)];
        const items = [];
        
        // Reduced to 8 items instead of 12
        for (let i = 0; i < 8; i++) {
          const color = colors[Math.floor(Math.random() * colors.length)];
          items.push({
            id: i,
            color,
            isTarget: color === targetColor,
            shape: 'circle'  // Only circles for simplicity
          });
        }
        
        // Ensure at least 2-3 target items
        const targetCount = Math.max(2, Math.min(3, items.filter(item => item.isTarget).length));
        const nonTargetItems = items.filter(item => !item.isTarget);
        const targetItems = items.filter(item => item.isTarget);
        
        // If not enough targets, convert some non-targets
        while (targetItems.length < targetCount && nonTargetItems.length > 0) {
          const item = nonTargetItems.pop();
          item.isTarget = true;
          item.color = targetColor;
          targetItems.push(item);
        }
        
        return { targetColor, items, correctAnswers: items.filter(item => item.isTarget).length };
      }
    },
    
    shapeCount: {
      name: "Shape Counter",
      description: "Count how many of the specified shape you see",
      icon: "🔢",
      generate: () => {
        const shapes = ['circle', 'square'];  // Only 2 shapes for simplicity
        const targetShape = shapes[Math.floor(Math.random() * shapes.length)];
        const items = [];
        let count = 0;
        
        // Reduced to 9 items instead of 16
        for (let i = 0; i < 9; i++) {
          const shape = shapes[Math.floor(Math.random() * shapes.length)];
          if (shape === targetShape) count++;
          items.push({
            id: i,
            shape,
            color: ['red', 'blue', 'green'][Math.floor(Math.random() * 3)]  // Only 3 colors
          });
        }
        
        // Ensure reasonable count (2-4)
        const targetCount = Math.max(2, Math.min(4, count));
        
        return { targetShape, items, correctAnswer: targetCount };
      }
    },
    
    patternComplete: {
      name: "Pattern Master",
      description: "Complete the pattern by selecting the missing piece",
      icon: "🧩",
      generate: () => {
        const patterns = [
          { sequence: ['red', 'blue', 'red', '?'], answer: 'blue' },  // Simpler patterns
          { sequence: ['circle', 'square', 'circle', '?'], answer: 'square' },
          { sequence: ['1', '2', '1', '?'], answer: '2' },
          { sequence: ['▲', '●', '▲', '?'], answer: '●' }
        ];
        
        const pattern = patterns[Math.floor(Math.random() * patterns.length)];
        // Simplified options - only 3 choices instead of 4
        const options = ['red', 'blue', 'circle', 'square', '1', '2', '▲', '●'];
        const wrongOptions = options.filter(opt => opt !== pattern.answer).slice(0, 2);
        const allOptions = [pattern.answer, ...wrongOptions].sort(() => Math.random() - 0.5);
        
        return { ...pattern, options: allOptions };
      }
    },
    
    memorySequence: {
      name: "Memory Sequence",
      description: "Remember and repeat the sequence",
      icon: "🧠",
      generate: () => {
        const colors = ['red', 'blue', 'green', 'yellow'];
        const sequence = [];
        const length = difficulty === 'easy' ? 3 : difficulty === 'medium' ? 4 : 5;
        
        for (let i = 0; i < length; i++) {
          sequence.push(colors[Math.floor(Math.random() * colors.length)]);
        }
        
        return { sequence, userSequence: [] };
      }
    },
    
    mathPuzzle: {
      name: "Math Puzzle",
      description: "Solve the simple math problem",
      icon: "🧮",
      generate: () => {
        const operations = ['+', '-'];  // Only addition and subtraction
        const operation = operations[Math.floor(Math.random() * operations.length)];
        let a, b, answer;
        
        switch (operation) {
          case '+':
            a = Math.floor(Math.random() * 10) + 1;  // 1-10
            b = Math.floor(Math.random() * 10) + 1;  // 1-10
            answer = a + b;
            break;
          case '-':
            a = Math.floor(Math.random() * 15) + 5;  // 5-19
            b = Math.floor(Math.random() * 5) + 1;   // 1-5
            answer = a - b;
            break;
        }
        
        // Only 2 wrong answers instead of 3
        const wrongAnswers = [
          answer + 1,
          answer - 1
        ].filter(a => a > 0);
        
        const options = [answer, ...wrongAnswers].sort(() => Math.random() - 0.5);
        
        return { question: `${a} ${operation} ${b} = ?`, answer, options };
      }
    }
  };

  // Initialize game
  useEffect(() => {
    startNewGame();
    setTimeLeft(config.timeLimit);
    
    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setGameStatus('failed');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startNewGame = () => {
    const availableGames = config.games;
    const gameType = availableGames[Math.floor(Math.random() * availableGames.length)];
    const game = gameData[gameType];
    const gameInstance = game.generate();
    
    setCurrentGame({
      type: gameType,
      ...gameInstance,
      name: game.name,
      description: game.description,
      icon: game.icon
    });
    setSelectedItems([]);
    setShowHint(false);
  };

  const handleItemClick = (item) => {
    if (gameStatus !== 'playing') return;
    
    setSelectedItems(prev => {
      if (prev.includes(item.id)) {
        return prev.filter(id => id !== item.id);
      } else {
        return [...prev, item.id];
      }
    });
  };

  const handleAnswerSubmit = () => {
    if (gameStatus !== 'playing') return;
    
    setAttempts(prev => prev + 1);
    let isCorrect = false;
    
    switch (currentGame.type) {
      case 'colorMatch':
        const selectedTargets = selectedItems.filter(id => 
          currentGame.items.find(item => item.id === id)?.isTarget
        );
        isCorrect = selectedTargets.length === currentGame.correctAnswers && 
                   selectedItems.length === currentGame.correctAnswers;
        break;
        
      case 'shapeCount':
        const selectedCount = selectedItems.length;
        isCorrect = selectedCount === currentGame.correctAnswer;
        break;
        
      case 'patternComplete':
        const selectedOption = selectedItems[0];
        isCorrect = currentGame.options[selectedOption] === currentGame.answer;
        break;
        
      case 'memorySequence':
        // This would need special handling for sequence input
        isCorrect = true; // Simplified for now
        break;
        
      case 'mathPuzzle':
        const selectedAnswer = selectedItems[0];
        isCorrect = currentGame.options[selectedAnswer] === currentGame.answer;
        break;
    }
    
    if (isCorrect) {
      const newScore = score + 1;
      const newStreak = streak + 1;
      setScore(newScore);
      setStreak(newStreak);
      
      setGameHistory(prev => [...prev, { 
        game: currentGame.name, 
        correct: true, 
        time: config.timeLimit - timeLeft 
      }]);
      
      if (newScore >= config.requiredScore) {
        setGameStatus('success');
        setTimeout(() => onSuccess(), 1000);
      } else {
        // Show success animation and start next game
        setTimeout(() => {
          startNewGame();
        }, 1500);
      }
    } else {
      setStreak(0);
      setGameHistory(prev => [...prev, { 
        game: currentGame.name, 
        correct: false, 
        time: config.timeLimit - timeLeft 
      }]);
      
      if (attempts >= 2) {
        setGameStatus('failed');
      } else {
        // Show error and allow retry
        setTimeout(() => {
          startNewGame();
        }, 1000);
      }
    }
  };

  const renderGame = () => {
    if (!currentGame) return null;

    switch (currentGame.type) {
      case 'colorMatch':
        return (
          <Box>
            <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
              Click all {currentGame.targetColor} items
            </Typography>
            <Box sx={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)',  // Reduced from 4 to 3 columns
              gap: 2, 
              mb: 3 
            }}>
              {currentGame.items.map(item => (
                <Box
                  key={item.id}
                  onClick={() => handleItemClick(item)}
                  sx={{
                    width: 60,
                    height: 60,
                    backgroundColor: item.color,
                    borderRadius: item.shape === 'circle' ? '50%' : 
                                item.shape === 'square' ? '8px' : 
                                item.shape === 'triangle' ? '0' : '8px',
                    border: selectedItems.includes(item.id) ? '3px solid #fff' : '2px solid #333',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s',
                    transform: selectedItems.includes(item.id) ? 'scale(1.1)' : 'scale(1)',
                    boxShadow: selectedItems.includes(item.id) ? '0 4px 8px rgba(0,0,0,0.3)' : '0 2px 4px rgba(0,0,0,0.1)'
                  }}
                />
              ))}
            </Box>
          </Box>
        );

      case 'shapeCount':
        return (
          <Box>
            <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
              How many {currentGame.targetShape}s do you see?
            </Typography>
            <Box sx={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)',  // Reduced from 4 to 3 columns
              gap: 2, 
              mb: 3 
            }}>
              {currentGame.items.map(item => (
                <Box
                  key={item.id}
                  sx={{
                    width: 50,
                    height: 50,
                    backgroundColor: item.color,
                    borderRadius: item.shape === 'circle' ? '50%' : 
                                item.shape === 'square' ? '8px' : 
                                item.shape === 'triangle' ? '0' : '8px',
                    border: '2px solid #333',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                />
              ))}
            </Box>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
              {[1, 2, 3, 4, 5].map(num => (  // Reduced from 8 to 5 options
                <Button
                  key={num}
                  variant={selectedItems.includes(num) ? 'contained' : 'outlined'}
                  onClick={() => handleItemClick(num)}
                  sx={{ minWidth: 40 }}
                >
                  {num}
                </Button>
              ))}
            </Box>
          </Box>
        );

      case 'patternComplete':
        return (
          <Box>
            <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
              Complete the pattern
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3 }}>
              {currentGame.sequence.map((item, index) => (
                <Box
                  key={index}
                  sx={{
                    width: 50,
                    height: 50,
                    backgroundColor: item === '?' ? '#f0f0f0' : 
                                   ['red', 'blue', 'green', 'yellow'].includes(item) ? item : 'transparent',
                    border: '2px solid #333',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.5rem',
                    fontWeight: 'bold'
                  }}
                >
                  {item === '?' ? '?' : item}
                </Box>
              ))}
            </Box>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
              {currentGame.options.map((option, index) => (
                <Button
                  key={index}
                  variant={selectedItems.includes(index) ? 'contained' : 'outlined'}
                  onClick={() => {
                    setSelectedItems([index]);
                  }}
                  sx={{ minWidth: 60 }}
                >
                  {option}
                </Button>
              ))}
            </Box>
          </Box>
        );

      case 'mathPuzzle':
        return (
          <Box>
            <Typography variant="h4" sx={{ mb: 3, textAlign: 'center' }}>
              {currentGame.question}
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              {currentGame.options.map((option, index) => (
                <Button
                  key={index}
                  variant={selectedItems.includes(index) ? 'contained' : 'outlined'}
                  onClick={() => {
                    setSelectedItems([index]);
                  }}
                  sx={{ minWidth: 80, fontSize: '1.2rem' }}
                >
                  {option}
                </Button>
              ))}
            </Box>
          </Box>
        );

      default:
        return <Typography>Game not implemented yet</Typography>;
    }
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        bgcolor: 'rgba(0, 0, 0, 0.9)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        backdropFilter: 'blur(10px)'
      }}
    >
      <Fade in={true}>
        <Paper
          sx={{
            p: 4,
            maxWidth: 600,
            width: '90%',
            bgcolor: '#1e293b',
            border: '2px solid #ef4444',
            borderRadius: 3,
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <Box
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                bgcolor: gameStatus === 'success' ? '#10b981' : 
                        gameStatus === 'failed' ? '#ef4444' : '#6366f1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ffffff',
                fontSize: '2.5rem',
                mx: 'auto',
                mb: 2,
                animation: gameStatus === 'playing' ? 'pulse 2s infinite' : 'none',
                '@keyframes pulse': {
                  '0%': { transform: 'scale(1)' },
                  '50%': { transform: 'scale(1.05)' },
                  '100%': { transform: 'scale(1)' }
                }
              }}
            >
              {gameStatus === 'success' ? '🎉' : 
               gameStatus === 'failed' ? '❌' : 
               currentGame?.icon || '🤖'}
            </Box>
            
            <Typography variant="h4" sx={{ 
              color: gameStatus === 'success' ? '#10b981' : 
                    gameStatus === 'failed' ? '#ef4444' : '#ffffff',
              fontWeight: 700, 
              mb: 1 
            }}>
              {gameStatus === 'success' ? 'Congratulations!' : 
               gameStatus === 'failed' ? 'Challenge Failed' : 
               'Bot Detection Challenge'}
            </Typography>
            
            <Typography variant="body1" sx={{ color: '#cbd5e1', mb: 2 }}>
              {gameStatus === 'success' ? 'You have proven you are human!' : 
               gameStatus === 'failed' ? 'Please try again to verify you are human.' : 
               'Complete the challenges to prove you are human'}
            </Typography>
          </Box>

          {/* Progress and Stats */}
          {gameStatus === 'playing' && (
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                  Score: {score}/{config.requiredScore}
                </Typography>
                <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                  Time: {timeLeft}s
                </Typography>
                <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                  Streak: {streak}
                </Typography>
              </Box>
              
              <LinearProgress 
                variant="determinate" 
                value={(score / config.requiredScore) * 100}
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  bgcolor: 'rgba(255,255,255,0.1)',
                  '& .MuiLinearProgress-bar': {
                    bgcolor: '#10b981'
                  }
                }}
              />
              
              <Box sx={{ display: 'flex', gap: 1, mt: 2, justifyContent: 'center' }}>
                <Chip 
                  label={`${currentGame?.name || 'Challenge'}`} 
                  size="small" 
                  color="primary" 
                />
                <Chip 
                  label={`Difficulty: ${difficulty}`} 
                  size="small" 
                  color="secondary" 
                />
                <Chip 
                  label={`Attempts: ${attempts}`} 
                  size="small" 
                  color="default" 
                />
              </Box>
            </Box>
          )}

          {/* Game Content */}
          {gameStatus === 'playing' && currentGame && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ mb: 2, textAlign: 'center', color: '#06b6d4' }}>
                {currentGame.name}
              </Typography>
              <Typography variant="body2" sx={{ mb: 3, textAlign: 'center', color: '#cbd5e1' }}>
                {currentGame.description}
              </Typography>
              
              {renderGame()}
              
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
                <Button
                  variant="contained"
                  onClick={handleAnswerSubmit}
                  disabled={selectedItems.length === 0}
                  sx={{
                    bgcolor: '#10b981',
                    '&:hover': { bgcolor: '#059669' },
                    px: 4
                  }}
                >
                  Submit Answer
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={() => setShowHint(!showHint)}
                  sx={{ color: '#06b6d4', borderColor: '#06b6d4' }}
                >
                  {showHint ? <EyeOffIcon /> : <EyeIcon />} Hint
                </Button>
              </Box>
              
              {showHint && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  💡 Look carefully at the pattern and think step by step!
                </Alert>
              )}
            </Box>
          )}

          {/* Success/Failure Actions */}
          {gameStatus !== 'playing' && (
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="outlined"
                onClick={onClose}
                sx={{ color: '#94a3b8', borderColor: '#475569' }}
              >
                Close
              </Button>
              
              {gameStatus === 'failed' && (
                <Button
                  variant="contained"
                  onClick={() => {
                    setGameStatus('playing');
                    setScore(0);
                    setAttempts(0);
                    setStreak(0);
                    setTimeLeft(config.timeLimit);
                    startNewGame();
                  }}
                  sx={{ bgcolor: '#6366f1' }}
                >
                  Try Again
                </Button>
              )}
            </Box>
          )}

          {/* Close button */}
          <IconButton
            onClick={onClose}
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              color: '#94a3b8',
              '&:hover': { color: '#ffffff' }
            }}
          >
            <CancelIcon />
          </IconButton>
        </Paper>
      </Fade>
    </Box>
  );
};

export default GamifiedCaptcha;
