import React, { useEffect, useState } from 'react';
import { Box, Paper, Typography, TextField, Button } from '@mui/material';

const MLDetectionMonitor = ({ onBotDetected }) => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [lastCheck, setLastCheck] = useState(null);

  useEffect(() => {
    // Start monitoring when component mounts
    setIsMonitoring(true);
    
    const checkMLResults = async () => {
      try {
        // Check for ML detection results by calling the bot-detection endpoint
        const response = await fetch('/api/bot-detection', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            sessionId: `monitor_${Date.now()}` 
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          if (data.success && data.results) {
            // Check if bot was detected
            if (data.results.loginAttempts && data.results.loginAttempts.length > 0) {
              const lastAttempt = data.results.loginAttempts[data.results.loginAttempts.length - 1];
              
              if (lastAttempt.classification === 'BOT') {
                console.log('🚨 BOT DETECTED by monitor! Triggering CAPTCHA...');
                onBotDetected(data.results);
              }
            }
          }
        }
        
        setLastCheck(new Date());
      } catch (error) {
        console.error('❌ Error checking ML results:', error);
      }
    };

    // Check every 2 seconds
    const interval = setInterval(checkMLResults, 2000);
    
    // Initial check
    checkMLResults();

    return () => {
      clearInterval(interval);
      setIsMonitoring(false);
    };
  }, [onBotDetected]);

  return (
    <Box sx={{ position: 'fixed', bottom: 20, right: 20, zIndex: 1000 }}>
      <Paper
        sx={{
          p: 2,
          bgcolor: 'rgba(0, 0, 0, 0.8)',
          color: '#ffffff',
          minWidth: 200,
        }}
      >
        <Typography variant="body2" sx={{ mb: 1 }}>
          🤖 ML Monitor: {isMonitoring ? 'Active' : 'Inactive'}
        </Typography>
        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
          Last check: {lastCheck ? lastCheck.toLocaleTimeString() : 'Never'}
        </Typography>
      </Paper>
    </Box>
  );
};

export default MLDetectionMonitor; 