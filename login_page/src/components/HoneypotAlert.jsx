import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Alert,
  Fade,
  Zoom,
  LinearProgress,
  Chip,
  IconButton
} from '@mui/material';
import {
  BugReport as BugIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  Close as CloseIcon
} from '@mui/icons-material';

const HoneypotAlert = ({ 
  isVisible, 
  honeypotValue, 
  onClose, 
  onProceed 
}) => {
  const [progress, setProgress] = useState(0);
  const [showDetails, setShowDetails] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);

  useEffect(() => {
    if (isVisible) {
      setProgress(0);
      setShowDetails(false);
      setAnimationStep(0);
      
      // Animate progress bar
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setShowDetails(true);
            return 100;
          }
          return prev + 3;
        });
      }, 30);
      
      // Animation steps
      const animationInterval = setInterval(() => {
        setAnimationStep(prev => (prev + 1) % 4);
      }, 500);
      
      return () => {
        clearInterval(interval);
        clearInterval(animationInterval);
      };
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const getAnimationIcon = () => {
    const icons = ['🍯', '🐝', '🕷️', '🦟'];
    return icons[animationStep];
  };

  const getHoneypotInfo = () => {
    const commonBotValues = ['spam', 'bot', 'automation', 'script', 'crawler', 'scraper'];
    const isCommonBotValue = commonBotValues.includes(honeypotValue?.toLowerCase());
    
    return {
      title: '🍯 Honeypot Trap Activated!',
      message: 'Suspicious form interaction detected',
      color: '#f59e0b',
      icon: <BugIcon />,
      details: isCommonBotValue 
        ? `Bot-like value detected: "${honeypotValue}". This is a common pattern used by automated scripts.`
        : `Hidden field filled with: "${honeypotValue}". Only automated tools typically interact with hidden fields.`,
      severity: isCommonBotValue ? 'error' : 'warning'
    };
  };

  const honeypotInfo = getHoneypotInfo();

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
        zIndex: 9997,
      }}
    >
      <Fade in={true}>
        <Paper
          elevation={24}
          sx={{
            p: 4,
            maxWidth: 600,
            width: '90%',
            borderRadius: 3,
            backgroundColor: '#ffffff',
            position: 'relative',
            border: `3px solid ${honeypotInfo.color}`,
          }}
        >
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Box
              sx={{
                p: 2,
                borderRadius: '50%',
                backgroundColor: `${honeypotInfo.color}20`,
                color: honeypotInfo.color,
                mr: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                minWidth: '60px',
                minHeight: '60px',
              }}
            >
              {getAnimationIcon()}
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: honeypotInfo.color }}>
                {honeypotInfo.title}
              </Typography>
              <Typography variant="h6" sx={{ color: '#64748b', mt: 0.5 }}>
                {honeypotInfo.message}
              </Typography>
            </Box>
            <IconButton
              onClick={onClose}
              sx={{
                color: '#64748b',
                '&:hover': { color: '#ef4444' }
              }}
            >
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Progress Bar */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" sx={{ mb: 1, color: '#64748b' }}>
              Analyzing form interaction patterns...
            </Typography>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 8,
                borderRadius: 4,
                backgroundColor: '#e2e8f0',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: honeypotInfo.color,
                  borderRadius: 4,
                },
              }}
            />
            <Typography variant="caption" sx={{ color: '#64748b', mt: 1, display: 'block' }}>
              {progress}% Complete
            </Typography>
          </Box>

          {/* Detection Details */}
          {showDetails && (
            <Zoom in={true}>
              <Box sx={{ mb: 3 }}>
                <Alert
                  severity={honeypotInfo.severity}
                  sx={{
                    mb: 2,
                    backgroundColor: `${honeypotInfo.color}10`,
                    border: `1px solid ${honeypotInfo.color}30`,
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                    {honeypotInfo.details}
                  </Typography>
                </Alert>

                {/* Honeypot Explanation */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" sx={{ color: '#64748b', mb: 2 }}>
                    <strong>What is a Honeypot Trap?</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#64748b', mb: 2 }}>
                    A honeypot is a hidden form field that humans cannot see or interact with. 
                    Only automated bots typically fill these fields, making them an effective 
                    way to detect automated behavior.
                  </Typography>
                  
                  <Box sx={{ 
                    p: 2, 
                    backgroundColor: '#f8fafc', 
                    borderRadius: 2, 
                    border: '1px solid #e2e8f0',
                    mb: 2 
                  }}>
                    <Typography variant="body2" sx={{ color: '#64748b', mb: 1 }}>
                      <strong>Detected Value:</strong>
                    </Typography>
                    <Chip 
                      label={`"${honeypotValue}"`} 
                      color="error" 
                      size="small"
                      sx={{ fontFamily: 'monospace' }}
                    />
                  </Box>
                </Box>

                {/* Security Measures */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" sx={{ color: '#64748b', mb: 1 }}>
                    Security measures activated:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip label="Honeypot Triggered" size="small" color="error" />
                    <Chip label="Form Validation" size="small" color="warning" />
                    <Chip label="Behavioral Analysis" size="small" color="info" />
                    <Chip label="CAPTCHA Required" size="small" color="error" />
                  </Box>
                </Box>

                {/* Technical Details */}
                <Box sx={{ 
                  p: 2, 
                  backgroundColor: '#f1f5f9', 
                  borderRadius: 2, 
                  border: '1px solid #cbd5e1' 
                }}>
                  <Typography variant="body2" sx={{ color: '#64748b', mb: 1 }}>
                    <strong>Technical Details:</strong>
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                    • Hidden field name: honeypotField<br/>
                    • Field visibility: display: none<br/>
                    • Interaction type: Automated form filling<br/>
                    • Detection method: Honeypot trap analysis
                  </Typography>
                </Box>
              </Box>
            </Zoom>
          )}

          {/* Action Buttons */}
          {showDetails && (
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <Box
                sx={{
                  px: 3,
                  py: 1.5,
                  backgroundColor: '#f8fafc',
                  borderRadius: 2,
                  border: '1px solid #e2e8f0',
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: '#f1f5f9',
                  },
                }}
                onClick={onClose}
              >
                <Typography variant="body2" sx={{ color: '#64748b' }}>
                  Close
                </Typography>
              </Box>
              <Box
                sx={{
                  px: 3,
                  py: 1.5,
                  backgroundColor: honeypotInfo.color,
                  borderRadius: 2,
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: `${honeypotInfo.color}dd`,
                  },
                }}
                onClick={onProceed}
              >
                <Typography variant="body2" sx={{ color: '#ffffff', fontWeight: 'medium' }}>
                  Proceed to Verification
                </Typography>
              </Box>
            </Box>
          )}
        </Paper>
      </Fade>
    </Box>
  );
};

export default HoneypotAlert;

