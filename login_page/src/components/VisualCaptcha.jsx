import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  TextField,
  IconButton,
  Alert,
  Fade,
  Zoom
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Visibility as EyeIcon,
  VisibilityOff as EyeOffIcon
} from '@mui/icons-material';

const VisualCaptcha = ({ onSuccess, onClose, difficulty = 'medium' }) => {
  const [captchaText, setCaptchaText] = useState('');
  const [userInput, setUserInput] = useState('');
  const [showCaptcha, setShowCaptcha] = useState(true);
  const [attempts, setAttempts] = useState(0);
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  const canvasRef = useRef(null);

  // Generate random CAPTCHA text based on difficulty
  const generateCaptchaText = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let length;
    
    switch (difficulty) {
      case 'easy':
        length = 4;
        break;
      case 'medium':
        length = 5;
        break;
      case 'hard':
        length = 6;
        break;
      default:
        length = 5;
    }
    
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  };

  // Draw the CAPTCHA on canvas
  const drawCaptcha = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Set background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);
    
    // Add noise lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo(Math.random() * width, Math.random() * height);
      ctx.lineTo(Math.random() * width, Math.random() * height);
      ctx.stroke();
    }
    
    // Add noise dots
    ctx.fillStyle = '#cbd5e1';
    for (let i = 0; i < 50; i++) {
      ctx.beginPath();
      ctx.arc(Math.random() * width, Math.random() * height, 1, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    // Draw the text
    ctx.font = 'bold 24px Arial';
    ctx.fillStyle = '#1e293b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Add some distortion to the text
    const textX = width / 2;
    const textY = height / 2;
    
    // Draw each character with slight variations
    for (let i = 0; i < captchaText.length; i++) {
      const char = captchaText[i];
      const charX = textX - (captchaText.length * 12) + (i * 24);
      const charY = textY + (Math.random() - 0.5) * 8;
      const rotation = (Math.random() - 0.5) * 0.3;
      
      ctx.save();
      ctx.translate(charX, charY);
      ctx.rotate(rotation);
      ctx.fillText(char, 0, 0);
      ctx.restore();
    }
    
    // Add some additional noise
    ctx.fillStyle = '#94a3b8';
    for (let i = 0; i < 20; i++) {
      ctx.beginPath();
      ctx.arc(Math.random() * width, Math.random() * height, 0.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // Initialize CAPTCHA
  useEffect(() => {
    const newCaptcha = generateCaptchaText();
    setCaptchaText(newCaptcha);
    setUserInput('');
    setError('');
    setSuccess(false);
  }, [difficulty]);

  // Draw CAPTCHA when text changes
  useEffect(() => {
    if (captchaText) {
      drawCaptcha();
    }
  }, [captchaText]);

  const handleRefresh = () => {
    const newCaptcha = generateCaptchaText();
    setCaptchaText(newCaptcha);
    setUserInput('');
    setError('');
    setSuccess(false);
  };

  const handleVerify = async () => {
    if (!userInput.trim()) {
      setError('Please enter the CAPTCHA text');
      return;
    }
    
    setIsVerifying(true);
    setError('');
    
    // Simulate verification delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (userInput.toLowerCase() === captchaText.toLowerCase()) {
      setSuccess(true);
      setTimeout(() => {
        onSuccess();
      }, 1500);
    } else {
      setAttempts(prev => prev + 1);
      setError(`Incorrect! Attempt ${attempts + 1}/3`);
      setUserInput('');
      
      if (attempts >= 2) {
        setError('Too many failed attempts. Please refresh and try again.');
        setTimeout(() => {
          handleRefresh();
          setAttempts(0);
        }, 2000);
      }
    }
    
    setIsVerifying(false);
  };

  const handleClose = () => {
    onClose();
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <Fade in={true}>
        <Paper
          elevation={24}
          sx={{
            p: 4,
            maxWidth: 500,
            width: '90%',
            borderRadius: 3,
            backgroundColor: '#ffffff',
            position: 'relative',
          }}
        >
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h5" sx={{ fontWeight: 'bold', color: '#1e293b' }}>
                🔒 Security Verification
              </Typography>
              <Typography variant="body2" sx={{ color: '#64748b', mt: 0.5 }}>
                Please complete the CAPTCHA to verify you're human
              </Typography>
            </Box>
            <IconButton
              onClick={handleClose}
              sx={{
                color: '#64748b',
                '&:hover': { color: '#ef4444' }
              }}
            >
              <CancelIcon />
            </IconButton>
          </Box>

          {/* CAPTCHA Display */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" sx={{ mb: 1, color: '#64748b' }}>
              Enter the text you see in the image:
            </Typography>
            <Box
              sx={{
                border: '2px solid #e2e8f0',
                borderRadius: 2,
                p: 2,
                backgroundColor: '#f8fafc',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2
              }}
            >
              <canvas
                ref={canvasRef}
                width={300}
                height={80}
                style={{
                  border: '1px solid #cbd5e1',
                  borderRadius: '4px',
                  backgroundColor: '#ffffff'
                }}
              />
            </Box>
          </Box>

          {/* Input Field */}
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Enter CAPTCHA text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              disabled={isVerifying || success}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '&:hover fieldset': {
                    borderColor: '#6366f1',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#6366f1',
                  },
                },
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleVerify();
                }
              }}
            />
          </Box>

          {/* Error/Success Messages */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          {success && (
            <Alert severity="success" sx={{ mb: 2 }}>
              <CheckIcon sx={{ mr: 1 }} />
              CAPTCHA verified successfully!
            </Alert>
          )}

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
              disabled={isVerifying || success}
              sx={{
                color: '#64748b',
                borderColor: '#cbd5e1',
                '&:hover': {
                  borderColor: '#6366f1',
                  color: '#6366f1',
                },
              }}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              onClick={handleVerify}
              disabled={isVerifying || success || !userInput.trim()}
              sx={{
                flex: 1,
                bgcolor: success ? '#10b981' : '#ef4444',
                color: '#ffffff',
                '&:hover': {
                  bgcolor: success ? '#059669' : '#dc2626',
                },
                '&:disabled': {
                  bgcolor: '#cbd5e1',
                  color: '#94a3b8',
                },
              }}
            >
              {isVerifying ? 'Verifying...' : success ? 'Verified!' : 'Verify'}
            </Button>
          </Box>

          {/* Difficulty Indicator */}
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Typography variant="caption" sx={{ color: '#64748b' }}>
              Difficulty: {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)} 
              ({captchaText.length} characters)
            </Typography>
          </Box>
        </Paper>
      </Fade>
    </Box>
  );
};

export default VisualCaptcha;

