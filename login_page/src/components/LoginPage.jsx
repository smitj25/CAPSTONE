import React, { useState, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper,
  Container, 
  Link,
  Fade,
  InputAdornment,
  IconButton,
  Alert
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MLDetectionMonitor from './MLDetectionMonitor';



// Create a modern theme with improved color palette
const modernTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6366f1', // Modern indigo
      light: '#818cf8',
      dark: '#4f46e5',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#06b6d4', // Cyan accent
      light: '#22d3ee',
      dark: '#0891b2',
    },
    success: {
      main: '#10b981', // Emerald green
      light: '#34d399',
      dark: '#059669',
    },
    background: {
      default: '#0f172a', // Slate 900
      paper: 'rgba(30, 41, 59, 0.95)', // Slate 800 with transparency
    },
    text: {
      primary: '#f8fafc', // Slate 50
      secondary: '#cbd5e1', // Slate 300
    },
    divider: 'rgba(148, 163, 184, 0.12)', // Slate 400 with low opacity
  },
  typography: {
    fontFamily: '"Inter", "Poppins", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      lineHeight: 1.2,
    },
    h2: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.02em',
      lineHeight: 1.3,
    },
    h3: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.015em',
      lineHeight: 1.3,
    },
    h4: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.015em',
      lineHeight: 1.4,
      fontSize: '1.75rem',
    },
    h5: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 500,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    h6: {
      fontFamily: '"Poppins", "Inter", sans-serif',
      fontWeight: 500,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    body1: {
      fontFamily: '"Inter", sans-serif',
      fontWeight: 400,
      lineHeight: 1.6,
      letterSpacing: '0.01em',
    },
    body2: {
      fontFamily: '"Inter", sans-serif',
      fontWeight: 400,
      lineHeight: 1.5,
      letterSpacing: '0.01em',
    },
    button: {
      fontFamily: '"Inter", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.02em',
      textTransform: 'none',
    },
    caption: {
      fontFamily: '"Inter", sans-serif',
      fontWeight: 400,
      letterSpacing: '0.03em',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '12px',
          fontFamily: '"Inter", sans-serif',
          fontWeight: 600,
          fontSize: '1rem',
          letterSpacing: '0.02em',
          padding: '12px 24px',
        },
        contained: {
          boxShadow: '0 4px 14px 0 rgba(99, 102, 241, 0.25)',
          '&:hover': {
            boxShadow: '0 6px 20px 0 rgba(99, 102, 241, 0.35)',
            transform: 'translateY(-1px)',
          },
          transition: 'all 0.2s ease-in-out',
        },
        large: {
          fontSize: '1.1rem',
          fontWeight: 600,
          padding: '14px 28px',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '12px',
            backgroundColor: 'rgba(51, 65, 85, 0.3)', // Slate 700 with transparency
            fontFamily: '"Inter", sans-serif',
            '& fieldset': {
              borderColor: 'rgba(148, 163, 184, 0.2)', // Slate 400 with low opacity
            },
            '&:hover fieldset': {
              borderColor: 'rgba(148, 163, 184, 0.4)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6366f1', // Primary color
              borderWidth: '2px',
            },
          },
          '& .MuiInputLabel-root': {
            color: '#cbd5e1', // Slate 300
            fontFamily: '"Inter", sans-serif',
            fontWeight: 500,
            letterSpacing: '0.01em',
            '&.Mui-focused': {
              color: '#6366f1', // Primary color
              fontWeight: 600,
            },
          },
          '& .MuiOutlinedInput-input': {
            color: '#f8fafc', // Slate 50
            fontFamily: '"Inter", sans-serif',
            fontWeight: 500,
            fontSize: '1rem',
            letterSpacing: '0.02em',
            '&::placeholder': {
              color: '#94a3b8',
              fontWeight: 400,
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
        },
      },
    },
    MuiTypography: {
      styleOverrides: {
        h1: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        h2: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        h3: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        h4: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        h5: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        h6: {
          fontFamily: '"Poppins", "Inter", sans-serif',
        },
        body1: {
          fontFamily: '"Inter", sans-serif',
        },
        body2: {
          fontFamily: '"Inter", sans-serif',
        },
      },
    },
    MuiLink: {
      styleOverrides: {
        root: {
          fontFamily: '"Inter", sans-serif',
          fontWeight: 500,
          textDecoration: 'none',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            textDecoration: 'underline',
          },
        },
      },
    },
  },
});

const LoginPage = ({ onLoginSuccess }) => {
  const [formData, setFormData] = useState({
    aadhaarNumber: '',
    honeypotField: '',
  });
  const [loginSuccess, setLoginSuccess] = useState(false);
  const [mouseMovements, setMouseMovements] = useState([]);
  const [sessionId] = useState(() => Math.random().toString(36).substring(2, 15));
  const [loginAttempts, setLoginAttempts] = useState([]);
  const [botDetectionResults, setBotDetectionResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showCaptcha, setShowCaptcha] = useState(false);
  const [captchaValue, setCaptchaValue] = useState('');
  const [captchaInput, setCaptchaInput] = useState('');
  const [validationErrors, setValidationErrors] = useState({});
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    // Track mouse movements
    const handleMouseMove = (e) => {
      setMouseMovements(prev => [...prev, `m(${e.clientX},${e.clientY})`]);
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Auto-trigger ML analysis when component mounts or when Aadhaar number is entered
  useEffect(() => {
    if (formData.aadhaarNumber.trim() && !botDetectionResults && !isAnalyzing) {
      // Small delay to avoid running analysis immediately on every keystroke
      const timer = setTimeout(() => {
        runBotDetection();
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [formData.aadhaarNumber, botDetectionResults, isAnalyzing]);

  // Track and save mouse movements periodically
  useEffect(() => {
    // Only set up the interval if we have a valid sessionId
    if (!sessionId) return;
    
    // Save mouse movements every 5 seconds if there are any new ones
    const saveInterval = setInterval(() => {
      if (mouseMovements.length > 0) {
        saveMouseMovements();
      }
    }, 5000);

    // Cleanup function
    return () => {
      clearInterval(saveInterval);
      
      // Save any remaining mouse movements when component unmounts
      if (mouseMovements.length > 0) {
        saveMouseMovements();
      }
    };
  }, [sessionId]); // Only depend on sessionId, not mouseMovements
  
  // Function to save mouse movements to server
  const saveMouseMovements = () => {
    const mouseMovementData = {
      mouseMovements: {
        timestamp: new Date().toISOString(),
        sessionId: sessionId,
        movements: [...mouseMovements] // Create a copy of the current movements
      }
    };

    // Store in localStorage (optional, for backup)
    localStorage.setItem('mouseMovements', JSON.stringify(mouseMovements));

    // Send to server
    fetch('/api/log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(mouseMovementData)
    })
    .then(() => {
      // Only clear movements after successful save
      setMouseMovements([]);
    })
    .catch(err => {
      console.error('Failed to write mouse movements:', err);
      // Could implement retry logic here if needed
    });
  };

  const validateAadhaar = (aadhaarNumber) => {
    const errors = {};
    
    // Remove spaces and check if empty
    const cleanAadhaar = aadhaarNumber.replace(/\s/g, '');
    
    if (!cleanAadhaar) {
      errors.aadhaarNumber = 'Aadhaar number is required';
    } else if (!/^\d{12}$/.test(cleanAadhaar)) {
      errors.aadhaarNumber = 'Aadhaar number must be exactly 12 digits';
    } else if (!/^[1-9]/.test(cleanAadhaar)) {
      errors.aadhaarNumber = 'Aadhaar number cannot start with 0';
    } else if (/(\d)\1{10}/.test(cleanAadhaar)) {
      errors.aadhaarNumber = 'Aadhaar number cannot have all identical digits';
    }
    
    return errors;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear validation errors when user types
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }

    // Log input events
    logEvent('input_change', { field: name });
  };

  const logEvent = (eventType, details, specificEventType = null, specificSessionId = null) => {
    const timestamp = new Date().toISOString();
    const currentSessionId = specificSessionId || sessionId;
    
    // Create structured web log entry
    const webLogEntry = {
      timestamp,
      method: eventType.toUpperCase(),
      path: window.location.pathname,
      protocol: 'HTTP/1.1',
      status: 200,
      bytes: JSON.stringify(details).length,
      referer: window.location.href,
      sessionId: currentSessionId,
      userAgent: navigator.userAgent,
      details: details // Include the details in the web log entry
    };

    // Store web log in localStorage (optional, for backup)
    try {
      const webLogs = JSON.parse(localStorage.getItem('webLogs') || '[]');
      webLogs.push(webLogEntry);
      localStorage.setItem('webLogs', JSON.stringify(webLogs));
    } catch (error) {
      console.error('Error storing web log in localStorage:', error);
    }

    // Send web log data to server
    const payload = specificEventType 
      ? { webLogEntry, eventType: specificEventType, timestamp, details, sessionId: currentSessionId, userAgent: navigator.userAgent }
      : { webLogEntry };
      
    fetch('/api/log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Web Log saved successfully:', data.message);
    })
    .catch(error => {
      console.error('Error sending web log:', error);
    });
  };

  const generateCaptcha = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < 6; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  };

  const checkBotDetection = (results) => {
    if (results && results.loginAttempts && results.loginAttempts.length > 0) {
      const lastAttempt = results.loginAttempts[results.loginAttempts.length - 1];
      if (lastAttempt.classification === 'BOT') {
        console.log('🚨 BOT DETECTED! Showing CAPTCHA...');
        setShowCaptcha(true);
        setCaptchaValue(generateCaptcha());
        return true;
      }
    }
    return false;
  };

  const handleBotDetected = (results) => {
    console.log('🚨 Bot detected by monitor!', results);
    setBotDetectionResults(results);
    checkBotDetection(results);
    
    // Clear any previous validation errors when ML analysis completes
    if (results && !results.error) {
      setValidationErrors({});
    }
  };

  const runBotDetection = async () => {
    setIsAnalyzing(true);
    setBotDetectionResults(null);
    setShowCaptcha(false);
    
    const startTime = Date.now();
    
    try {
      const response = await fetch('/api/bot-detection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sessionId })
      });
      
      const data = await response.json();
      const processingTime = Date.now() - startTime;
      
      if (data.success) {
        setBotDetectionResults({
          ...data.results,
          processingTime: data.processingTime || processingTime,
          cached: data.cached || false
        });
        console.log('Bot detection results:', data.results);
        
        // Check if bot was detected and show CAPTCHA
        const botDetected = checkBotDetection(data.results);
        if (botDetected) {
          console.log('🎯 CAPTCHA triggered due to bot detection!');
        }
      } else {
        console.error('Bot detection failed:', data.error);
        setBotDetectionResults({ 
          error: data.error,
          processingTime: processingTime
        });
      }
    } catch (error) {
      console.error('Error running bot detection:', error);
      setBotDetectionResults({ 
        error: 'Failed to run bot detection',
        processingTime: Date.now() - startTime
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Prevent multiple submissions
    if (isSubmitting || isAnalyzing || isValidating) {
      return;
    }

    // Validate form before submission
    const errors = validateAadhaar(formData.aadhaarNumber);
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }

    setIsSubmitting(true);
    setIsValidating(true);

    // Run ML analysis if it hasn't been run yet
    if (!botDetectionResults) {
      try {
        await runBotDetection();
      } catch (error) {
        console.error('Error running ML analysis:', error);
        setValidationErrors({ general: 'Security verification failed. Please try again.' });
        setIsSubmitting(false);
        setIsValidating(false);
        return;
      }
    }

    try {
      // Check for honeypot
      if (formData.honeypotField) {
        logEvent('honeypot_triggered', { value: formData.honeypotField });
        setIsSubmitting(false);
        setIsValidating(false);
        return;
      }

      // Validate Aadhaar number with enhanced validation
      const cleanAadhaar = formData.aadhaarNumber.replace(/\s/g, '');
      const isValidAadhaar = /^\d{12}$/.test(cleanAadhaar) && 
                             /^[1-9]/.test(cleanAadhaar) && 
                             !/(\d)\1{10}/.test(cleanAadhaar);
      
      if (!isValidAadhaar) {
        setValidationErrors({ aadhaarNumber: 'Invalid Aadhaar number format' });
        setIsSubmitting(false);
        setIsValidating(false);
        return;
      }

      // Log login attempt
      const attemptData = {
        timestamp: new Date().toISOString(),
        aadhaarNumber: cleanAadhaar.replace(/\d(?=\d{4})/g, '*'), // Mask Aadhaar number for privacy
        mouseMovements: mouseMovements.length,
        success: true
      };

      setLoginAttempts(prev => [...prev, attemptData]);
      
      // Save mouse movements before logging the login attempt
      if (mouseMovements.length > 0) {
        saveMouseMovements();
      }
      
      // Log the login attempt with sessionId
      logEvent('login_attempt', attemptData, 'login_attempt', sessionId);

      // Run bot detection analysis
      await runBotDetection();

      // If bot detection passes, proceed with login
      if (botDetectionResults && botDetectionResults.loginAttempts && botDetectionResults.loginAttempts.length > 0) {
        // Check if any login attempt was classified as BOT
        const hasBotDetection = botDetectionResults.loginAttempts.some(attempt => 
          attempt.classification === 'BOT'
        );
        
        if (!hasBotDetection) {
          // Call the parent component's onLoginSuccess callback
          if (onLoginSuccess) {
            onLoginSuccess({
              aadhaarNumber: cleanAadhaar,
              sessionId: sessionId,
              loginTime: new Date().toISOString(),
              botDetectionResults: botDetectionResults
            });
          }
        } else {
          // If bot was detected, show CAPTCHA
          setShowCaptcha(true);
          setCaptchaValue(generateCaptcha());
          setValidationErrors({ general: 'Bot behavior detected. Please complete CAPTCHA verification.' });
        }
      } else {
        // If bot detection results are incomplete, proceed with login anyway
        if (onLoginSuccess) {
          onLoginSuccess({
            aadhaarNumber: cleanAadhaar,
            sessionId: sessionId,
            loginTime: new Date().toISOString(),
            botDetectionResults: botDetectionResults
          });
        }
      }
    } catch (error) {
      console.error('Error in form submission:', error);
      setValidationErrors({ general: 'An error occurred during login. Please try again.' });
    } finally {
      setIsSubmitting(false);
      setIsValidating(false);
    }
  };

  return (
    <ThemeProvider theme={modernTheme}>
      <Box
        sx={{
          minHeight: '100vh',
          width: '100vw',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          margin: 0,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
          }
        }}
      >
        {/* Main content */}
        <Container 
          component="main" 
          maxWidth="sm"
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '100vh',
            padding: 3
          }}
        >
            <Paper
              elevation={8}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                width: '100%',
                maxWidth: '400px',
                padding: 4
              }}
            >
              {/* Login form */}
              <Box sx={{ width: '100%', mb: 2 }}>
                <Typography 
                  component="h1" 
                  variant="h4" 
                  sx={{ 
                    textAlign: 'center',
                    mb: 4,
                    fontFamily: '"Poppins", "Inter", sans-serif',
                    fontWeight: 700,
                    fontSize: '2rem',
                    letterSpacing: '-0.02em',
                    background: 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)',
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  Aadhaar Login
                </Typography>

                <Box component="form" onSubmit={handleSubmit} noValidate>
                  <TextField
                    margin="normal"
                    required
                    fullWidth
                    id="aadhaarNumber"
                    label="Aadhaar Number"
                    placeholder="1234 5678 9012"
                    name="aadhaarNumber"
                    autoComplete="off"
                    autoFocus
                    value={formData.aadhaarNumber}
                    onChange={handleInputChange}
                    error={!!validationErrors.aadhaarNumber}
                    helperText={validationErrors.aadhaarNumber || ''}
                    sx={{ mb: 3 }}
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton edge="end" tabIndex={-1} sx={{ color: '#cbd5e1' }}>
                            🔑
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                  
                  {/* General validation error */}
                  {validationErrors.general && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                      {validationErrors.general}
                    </Alert>
                  )}
                  
                  {/* ML Analysis Status Messages */}
                  {isAnalyzing && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      🔍 Running ML analysis for security verification. This may take a few seconds...
                    </Alert>
                  )}
                  
                  {botDetectionResults && !botDetectionResults.error && !validationErrors.general && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      ✅ ML Analysis completed successfully! You can now proceed with login.
                    </Alert>
                  )}
                  
                  {/* Honeypot field - hidden from regular users */}
                  <input
                    type="text"
                    name="honeypotField"
                    value={formData.honeypotField}
                    onChange={handleInputChange}
                    style={{ display: 'none' }}
                    tabIndex="-1"
                    aria-hidden="true"
                  />
                  
                  <Button
                    type="submit"
                    fullWidth
                    variant="contained"
                    size="large"
                    disabled={isAnalyzing || isSubmitting || isValidating || !formData.aadhaarNumber.trim()}
                    sx={{ 
                      mt: 2, 
                      mb: 3,
                      py: 1.5,
                      background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                      color: '#ffffff',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #4f46e5 0%, #3730a3 100%)',
                      },
                      '&:disabled': {
                        background: 'rgba(99, 102, 255, 0.5)',
                        color: 'rgba(255, 255, 255, 0.7)',
                      }
                    }}
                  >
                    {isAnalyzing ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 16,
                            height: 16,
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderTop: '2px solid #ffffff',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                            '@keyframes spin': {
                              '0%': { transform: 'rotate(0deg)' },
                              '100%': { transform: 'rotate(360deg)' }
                            }
                          }}
                        />
                        Running ML Analysis...
                      </Box>
                    ) : isSubmitting ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 16,
                            height: 16,
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderTop: '2px solid #ffffff',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                            '@keyframes spin': {
                              '0%': { transform: 'rotate(0deg)' },
                              '100%': { transform: 'rotate(360deg)' }
                            }
                          }}
                        />
                        Processing...
                      </Box>
                    ) : isValidating ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 16,
                            height: 16,
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderTop: '2px solid #ffffff',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                            '@keyframes spin': {
                              '0%': { transform: 'rotate(0deg)' },
                              '100%': { transform: 'rotate(360deg)' }
                            }
                          }}
                        />
                        Validating...
                      </Box>
                    ) : (
                      'Sign In'
                    )}
                  </Button>
                  
                  {/* Helper text for button state */}
                  {isAnalyzing && (
                    <Typography variant="caption" sx={{ color: '#06b6d4', textAlign: 'center', display: 'block', mt: 1 }}>
                      🔍 Running ML analysis for security verification...
                    </Typography>
                  )}
                  
                  {botDetectionResults && !isAnalyzing && (
                    <Typography variant="caption" sx={{ color: '#10b981', textAlign: 'center', display: 'block', mt: 1 }}>
                      ✅ Security verification complete! You can now sign in.
                    </Typography>
                  )}
                  
                  <Box sx={{ textAlign: 'center', mt: 2 }}>
                    <Link 
                      href="#" 
                      sx={{
                        color: '#cbd5e1',
                        textDecoration: 'none',
                        fontSize: '0.9rem',
                        fontFamily: '"Inter", sans-serif',
                        fontWeight: 500,
                        letterSpacing: '0.01em',
                        '&:hover': {
                          textDecoration: 'underline',
                          color: '#06b6d4'
                        }
                      }}
                    >
                      Forgot Aadhaar Number?
                    </Link>
                  </Box>
                </Box>
              </Box>

              {/* ML Analysis Progress Indicator */}
              {isAnalyzing && (
                <Box sx={{ width: '100%', mt: 4 }}>
                  <Paper sx={{ p: 3, bgcolor: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Box
                        sx={{
                          width: 24,
                          height: 24,
                          border: '3px solid rgba(99, 102, 241, 0.3)',
                          borderTop: '3px solid #6366f1',
                          borderRadius: '50%',
                          animation: 'spin 1s linear infinite',
                          '@keyframes spin': {
                            '0%': { transform: 'rotate(0deg)' },
                            '100%': { transform: 'rotate(360deg)' }
                          }
                        }}
                      />
                      <Typography variant="h6" sx={{ ml: 2, color: '#6366f1', fontWeight: 600 }}>
                        🤖 ML Bot Detection in Progress...
                      </Typography>
                    </Box>
                    
                    <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 2 }}>
                      Analyzing your mouse movements and web behavior patterns...
                    </Typography>
                    
                    <Box sx={{ width: '100%', bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 1, overflow: 'hidden' }}>
                      <Box
                        sx={{
                          width: '100%',
                          height: 4,
                          bgcolor: '#6366f1',
                          animation: 'progress 2s ease-in-out infinite',
                          '@keyframes progress': {
                            '0%': { transform: 'translateX(-100%)' },
                            '100%': { transform: 'translateX(100%)' }
                          }
                        }}
                      />
                    </Box>
                    
                    <Typography variant="caption" sx={{ color: '#94a3b8', mt: 1, display: 'block' }}>
                      Processing mouse movements, web logs, and running neural networks...
                    </Typography>
                  </Paper>
                </Box>
              )}

              {/* Bot Detection Results */}
              {botDetectionResults && (
                <Box sx={{ width: '100%', mt: 4 }}>
                  <Paper sx={{ p: 3, bgcolor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Box
                        sx={{
                          width: 32,
                          height: 32,
                          borderRadius: '50%',
                          bgcolor: botDetectionResults.cached ? '#06b6d4' : '#10b981',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#ffffff',
                          fontSize: '1.2rem',
                          mr: 2
                        }}
                      >
                        {botDetectionResults.cached ? '⚡' : '✓'}
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="h6" sx={{ color: '#10b981', fontWeight: 600 }}>
                          ML Analysis Complete!
                        </Typography>
                        {botDetectionResults.processingTime && (
                          <Typography variant="caption" sx={{ color: '#94a3b8', display: 'block' }}>
                            {botDetectionResults.cached ? 'Cached result' : 'Fresh analysis'} • 
                            Processing time: {botDetectionResults.processingTime.toFixed(2)}ms
                          </Typography>
                        )}
                      </Box>
                    </Box>
                    
                    {botDetectionResults.error ? (
                      <Typography color="error" variant="body2">
                        Error: {botDetectionResults.error}
                      </Typography>
                    ) : (
                      <Box>
                        <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 2 }}>
                          ✅ Successfully analyzed your behavior patterns using advanced ML models
                        </Typography>
                        
                        {/* Mouse Movement Results */}
                        {botDetectionResults.mouseMovements && botDetectionResults.mouseMovements.length > 0 && (
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" sx={{ color: '#06b6d4', fontWeight: 600, mb: 1 }}>
                              🖱️ Mouse Movement Analysis
                            </Typography>
                            {botDetectionResults.mouseMovements.map((session, index) => (
                              <Box key={index} sx={{ mb: 1 }}>
                                <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                                  Session {index + 1}: {session.movements} movements analyzed
                                </Typography>
                                <Typography variant="body2" sx={{ 
                                  color: session.classification === 'HUMAN' ? '#10b981' : '#ef4444',
                                  fontWeight: 600 
                                }}>
                                  Score: {session.botScore.toFixed(4)} → {session.classification}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        )}

                        {/* Web Log Results */}
                        {botDetectionResults.webLogs && (
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" sx={{ color: '#06b6d4', fontWeight: 600, mb: 1 }}>
                              🌐 Web Behavior Analysis
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
                              {botDetectionResults.webLogs.entries} web interactions analyzed
                            </Typography>
                            <Typography variant="body2" sx={{ 
                              color: botDetectionResults.webLogs.classification === 'HUMAN' ? '#10b981' : '#ef4444',
                              fontWeight: 600 
                            }}>
                              Score: {botDetectionResults.webLogs.botScore.toFixed(4)} → {botDetectionResults.webLogs.classification}
                            </Typography>
                          </Box>
                        )}

                        {/* Login Attempt Results */}
                        {botDetectionResults.loginAttempts && botDetectionResults.loginAttempts.length > 0 && (
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" sx={{ color: '#06b6d4', fontWeight: 600, mb: 1 }}>
                              🔐 Final Classification
                            </Typography>
                            {botDetectionResults.loginAttempts.map((attempt, index) => (
                              <Box key={index} sx={{ p: 1, bgcolor: 'rgba(51, 65, 85, 0.3)', borderRadius: 1 }}>
                                <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 0.5 }}>
                                  Attempt {attempt.attemptNumber}:
                                </Typography>
                                <Typography variant="body2" sx={{ color: '#94a3b8', fontSize: '0.8rem' }}>
                                  Mouse: {attempt.mouseScore.toFixed(4)} | Web: {attempt.webScore.toFixed(4)}
                                </Typography>
                                <Typography variant="body2" sx={{ 
                                  color: attempt.classification === 'HUMAN' ? '#10b981' : '#ef4444',
                                  fontWeight: 600,
                                  fontSize: '0.9rem'
                                }}>
                                  Fusion: {attempt.fusionScore.toFixed(4)} → {attempt.classification}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        )}

                        <Typography variant="body2" sx={{ color: '#10b981', fontWeight: 600, mt: 2 }}>
                          🎯 ML Analysis: All patterns indicate HUMAN behavior with high confidence
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>
              )}
              
              <Typography 
                variant="body2" 
                sx={{ 
                  mt: 4, 
                  color: '#94a3b8', 
                  textAlign: 'center', 
                  fontSize: '0.85rem',
                  fontFamily: '"Inter", sans-serif',
                  fontWeight: 400,
                  letterSpacing: '0.02em',
                }}
              >
                              © 2024 Unique Identification Authority of India
            </Typography>
          </Paper>

          {/* CAPTCHA Modal */}
          {showCaptcha && (
            <Box
              sx={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                bgcolor: 'rgba(0, 0, 0, 0.8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 9999,
              }}
            >
              <Paper
                sx={{
                  p: 4,
                  maxWidth: 400,
                  width: '90%',
                  bgcolor: '#1e293b',
                  border: '2px solid #ef4444',
                  borderRadius: 2,
                }}
              >
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Box
                    sx={{
                      width: 64,
                      height: 64,
                      borderRadius: '50%',
                      bgcolor: '#ef4444',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#ffffff',
                      fontSize: '2rem',
                      mx: 'auto',
                      mb: 2,
                    }}
                  >
                    🤖
                  </Box>
                  <Typography variant="h5" sx={{ color: '#ef4444', fontWeight: 600, mb: 1 }}>
                    Bot Detected!
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                    Suspicious behavior detected. Please complete the CAPTCHA to continue.
                  </Typography>
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" sx={{ color: '#94a3b8', mb: 2 }}>
                    Enter the code shown below:
                  </Typography>
                  
                  <Box
                    sx={{
                      p: 2,
                      bgcolor: '#334155',
                      borderRadius: 1,
                      textAlign: 'center',
                      mb: 2,
                      border: '1px solid #475569',
                    }}
                  >
                    <Typography
                      variant="h4"
                      sx={{
                        color: '#ffffff',
                        fontWeight: 700,
                        letterSpacing: '0.2em',
                        fontFamily: 'monospace',
                        textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
                      }}
                    >
                      {captchaValue}
                    </Typography>
                  </Box>

                  <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Enter CAPTCHA code"
                    value={captchaInput}
                    onChange={(e) => setCaptchaInput(e.target.value.toUpperCase())}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        color: '#ffffff',
                        '& fieldset': {
                          borderColor: '#475569',
                        },
                        '&:hover fieldset': {
                          borderColor: '#6366f1',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#6366f1',
                        },
                      },
                      '& .MuiInputLabel-root': {
                        color: '#94a3b8',
                      },
                    }}
                  />
                </Box>

                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => {
                      setShowCaptcha(false);
                      setCaptchaInput('');
                      setCaptchaValue(generateCaptcha());
                    }}
                    sx={{
                      color: '#94a3b8',
                      borderColor: '#475569',
                      '&:hover': {
                        borderColor: '#6366f1',
                        color: '#6366f1',
                      },
                    }}
                  >
                    Refresh
                  </Button>
                  <Button
                    fullWidth
                    variant="contained"
                    onClick={() => {
                      if (captchaInput === captchaValue) {
                        setShowCaptcha(false);
                        setCaptchaInput('');
                        alert('CAPTCHA verified! Access granted.');
                      } else {
                        alert('Incorrect CAPTCHA! Please try again.');
                        setCaptchaInput('');
                        setCaptchaValue(generateCaptcha());
                      }
                    }}
                    sx={{
                      bgcolor: '#ef4444',
                      color: '#ffffff',
                      '&:hover': {
                        bgcolor: '#dc2626',
                      },
                    }}
                  >
                    Verify
                  </Button>
                </Box>
              </Paper>
            </Box>
          )}
        </Container>
      </Box>
      
      {/* ML Detection Monitor */}
      <MLDetectionMonitor onBotDetected={handleBotDetected} />
    </ThemeProvider>
  );
};

export default LoginPage;