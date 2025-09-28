import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Alert,
  Fade,
  Zoom,
  LinearProgress,
  Chip
} from '@mui/material';
import {
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';

const BotDetectionAlert = ({ 
  isVisible, 
  detectionType, 
  confidence, 
  onClose, 
  onProceed 
}) => {
  const [progress, setProgress] = useState(0);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    if (isVisible) {
      setProgress(0);
      setShowDetails(false);
      
      // Animate progress bar
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setShowDetails(true);
            return 100;
          }
          return prev + 2;
        });
      }, 50);
      
      return () => clearInterval(interval);
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const getDetectionInfo = () => {
    switch (detectionType) {
      case 'bot':
        return {
          title: '🤖 Bot Detected!',
          message: 'Automated behavior detected',
          color: '#ef4444',
          icon: <SecurityIcon />,
          details: 'Our AI system has identified patterns consistent with automated software. This helps protect against malicious bots.'
        };
      case 'honeypot':
        return {
          title: '🍯 Honeypot Trap Activated!',
          message: 'Suspicious form interaction detected',
          color: '#f59e0b',
          icon: <WarningIcon />,
          details: 'A hidden security field was filled, indicating automated form submission. This is a common bot behavior pattern.'
        };
      case 'ml_analysis':
        return {
          title: '🧠 AI Analysis Complete',
          message: 'Behavioral analysis indicates bot activity',
          color: '#8b5cf6',
          icon: <SecurityIcon />,
          details: 'Machine learning analysis of mouse movements, typing patterns, and interaction timing suggests automated behavior.'
        };
      default:
        return {
          title: '⚠️ Suspicious Activity Detected',
          message: 'Security analysis triggered',
          color: '#6b7280',
          icon: <WarningIcon />,
          details: 'Multiple security indicators suggest automated behavior.'
        };
    }
  };

  const detectionInfo = getDetectionInfo();
  const confidenceLevel = confidence > 0.8 ? 'High' : confidence > 0.6 ? 'Medium' : 'Low';

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
        zIndex: 9998,
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
            border: `3px solid ${detectionInfo.color}`,
          }}
        >
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Box
              sx={{
                p: 2,
                borderRadius: '50%',
                backgroundColor: `${detectionInfo.color}20`,
                color: detectionInfo.color,
                mr: 2,
              }}
            >
              {detectionInfo.icon}
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: detectionInfo.color }}>
                {detectionInfo.title}
              </Typography>
              <Typography variant="h6" sx={{ color: '#64748b', mt: 0.5 }}>
                {detectionInfo.message}
              </Typography>
            </Box>
          </Box>

          {/* Progress Bar */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" sx={{ mb: 1, color: '#64748b' }}>
              Analyzing security patterns...
            </Typography>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 8,
                borderRadius: 4,
                backgroundColor: '#e2e8f0',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: detectionInfo.color,
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
                  severity="warning"
                  sx={{
                    mb: 2,
                    backgroundColor: `${detectionInfo.color}10`,
                    border: `1px solid ${detectionInfo.color}30`,
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                    {detectionInfo.details}
                  </Typography>
                </Alert>

                {/* Confidence Level */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="body2" sx={{ color: '#64748b' }}>
                    Detection Confidence:
                  </Typography>
                  <Chip
                    label={`${confidenceLevel} (${Math.round(confidence * 100)}%)`}
                    color={confidence > 0.8 ? 'error' : confidence > 0.6 ? 'warning' : 'info'}
                    size="small"
                  />
                </Box>

                {/* Security Measures */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" sx={{ color: '#64748b', mb: 1 }}>
                    Security measures activated:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip label="Enhanced Monitoring" size="small" color="error" />
                    <Chip label="Behavioral Analysis" size="small" color="warning" />
                    <Chip label="Form Validation" size="small" color="info" />
                    <Chip label="CAPTCHA Required" size="small" color="error" />
                  </Box>
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
                  backgroundColor: detectionInfo.color,
                  borderRadius: 2,
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: `${detectionInfo.color}dd`,
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

export default BotDetectionAlert;

