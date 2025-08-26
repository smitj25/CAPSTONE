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
  IconButton
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';



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

const LoginPage = () => {
  const [formData, setFormData] = useState({
    aadhaarNumber: '',
    honeypotField: '',
  });
  const [loginSuccess, setLoginSuccess] = useState(false);
  const [mouseMovements, setMouseMovements] = useState([]);
  const [sessionId] = useState(() => Math.random().toString(36).substring(2, 15));
  const [loginAttempts, setLoginAttempts] = useState([]);

  useEffect(() => {
    // Track mouse movements
    const handleMouseMove = (e) => {
      setMouseMovements(prev => [...prev, `m(${e.clientX},${e.clientY})`]);
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, []);

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

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

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

  const handleSubmit = (e) => {
    e.preventDefault();

    // Check for honeypot
    if (formData.honeypotField) {
      logEvent('honeypot_triggered', { value: formData.honeypotField });
      return;
    }

    // Validate Aadhaar number (basic validation - 12 digits)
    const isValidAadhaar = /^\d{12}$/.test(formData.aadhaarNumber.replace(/\s/g, ''));
    const isLoginSuccessful = isValidAadhaar;

    // Log login attempt
    const attemptData = {
      timestamp: new Date().toISOString(),
      aadhaarNumber: formData.aadhaarNumber.replace(/\d(?=\d{4})/g, '*'), // Mask Aadhaar number for privacy
      mouseMovements: mouseMovements.length,
      success: isLoginSuccessful
    };

    setLoginAttempts(prev => [...prev, attemptData]);
    
    // Save mouse movements before logging the login attempt
    if (mouseMovements.length > 0) {
      saveMouseMovements();
    }
    
    // Log the login attempt with sessionId
    logEvent('login_attempt', attemptData, 'login_attempt', sessionId);

    if (isLoginSuccessful) {
      setLoginSuccess(true);
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
          {loginSuccess ? (
            <Paper
              elevation={8}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                width: '100%',
                maxWidth: '400px',
                padding: 4,
                textAlign: 'center'
              }}
            >
                {/* Success Content */}
                <Box sx={{ width: '100%', textAlign: 'center' }}>
                  <Fade in={true} timeout={500}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <Box 
                        sx={{ 
                          width: 80, 
                          height: 80, 
                          borderRadius: '50%', 
                          bgcolor: '#10b981',
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          color: '#ffffff',
                          fontSize: '2.5rem',
                          mb: 4,
                          boxShadow: '0 8px 32px rgba(16, 185, 129, 0.3)',
                        }}
                      >
                        ✓
                      </Box>
                      
                      <Typography 
                        variant="h4" 
                        sx={{ 
                          mb: 2,
                          color: '#10b981',
                          fontWeight: 600,
                          fontFamily: '"Poppins", "Inter", sans-serif',
                          letterSpacing: '-0.015em',
                        }}
                      >
                        Login Successful
                      </Typography>
                      
                      <Typography 
                        variant="body1" 
                        sx={{ 
                          mb: 4,
                          color: '#cbd5e1',
                          fontFamily: '"Inter", sans-serif',
                          fontSize: '1.1rem',
                          lineHeight: 1.6,
                          fontWeight: 400,
                        }}
                      >
                        You have successfully authenticated with your Aadhaar credentials.
                      </Typography>
                      
                      <Box sx={{ width: '100%', mt: 1 }}>
                        <Button
                          variant="contained"
                          fullWidth
                          size="large"
                          sx={{ 
                            mb: 3,
                            py: 1.5,
                            backgroundColor: '#10b981',
                            color: '#ffffff',
                            '&:hover': {
                              backgroundColor: '#059669',
                              boxShadow: '0 6px 20px rgba(16, 185, 129, 0.4)',
                            }
                          }}
                          onClick={() => window.location.href = "#dashboard"}
                        >
                          Continue to Dashboard
                        </Button>
                        
                        <Button
                          variant="text"
                          sx={{
                            color: '#cbd5e1',
                            '&:hover': {
                              backgroundColor: 'rgba(203, 213, 225, 0.1)',
                              color: '#f8fafc'
                            }
                          }}
                          onClick={() => setLoginSuccess(false)}
                        >
                          Sign Out
                        </Button>
                      </Box>
                    </Box>
                  </Fade>
                </Box>
            </Paper>
          ) : (
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
                    sx={{ 
                      mt: 2, 
                      mb: 3,
                      py: 1.5,
                      background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                      color: '#ffffff',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #4f46e5 0%, #3730a3 100%)',
                      }
                    }}
                  >
                    Sign In
                  </Button>
                  
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
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default LoginPage;