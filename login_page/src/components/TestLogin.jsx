import React from 'react';
import { Box, Typography, Paper, Container } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

const testTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
});

const TestLogin = () => {
  return (
    <ThemeProvider theme={testTheme}>
      <Box
        sx={{
          minHeight: '100vh',
          width: '100vw',
          background: 'linear-gradient(135deg, #0070ba 0%, #004e82 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          margin: 0,
        }}
      >
        <Container maxWidth="sm">
          <Paper
            elevation={8}
            sx={{
              padding: 4,
              textAlign: 'center',
              maxWidth: '400px',
              margin: '0 auto'
            }}
          >
            <Typography variant="h4" sx={{ mb: 2, color: 'white' }}>
              Test Login Page
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
              This is a test to see if the component renders properly.
            </Typography>
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default TestLogin;