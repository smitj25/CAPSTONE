import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Avatar,
  AppBar,
  Toolbar,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Drawer,
  ListItemButton,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Person as PersonIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Menu as MenuIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const Dashboard = ({ onSignOut, userData }) => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const handleSignOut = () => {
    if (onSignOut) {
      onSignOut();
    }
  };

  const toggleDrawer = () => setDrawerOpen(!drawerOpen);

  const renderOverview = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card sx={{ background: 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)' }}>
          <CardContent sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h4" sx={{ color: '#ffffff', mb: 1, fontWeight: 700 }}>
                  Welcome back! 👋
                </Typography>
                <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }}>
                  Aadhaar ID: {userData?.aadhaarNumber || '**** **** ****'}
                </Typography>
              </Box>
              <Avatar sx={{ width: 80, height: 80, bgcolor: 'rgba(255, 255, 255, 0.2)' }}>
                <PersonIcon sx={{ fontSize: 40, color: '#ffffff' }} />
              </Avatar>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center', p: 3 }}>
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: '#6366f1' }}>
              127
            </Typography>
            <Typography variant="body2" sx={{ color: '#94a3b8' }}>
              Total Logins
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent sx={{ textAlign: 'center', p: 3 }}>
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, color: '#10b981' }}>
              95%
            </Typography>
            <Typography variant="body2" sx={{ color: '#94a3b8' }}>
              Security Score
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
              Security Status
            </Typography>
            <Alert severity="success" sx={{ mb: 2 }}>
              Your account is secure with ML-based bot detection enabled.
            </Alert>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
                Bot Detection: 100%
              </Typography>
              <LinearProgress variant="determinate" value={100} sx={{ height: 8, borderRadius: 4 }} />
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
                Threat Prevention: 98%
              </Typography>
              <LinearProgress variant="determinate" value={98} sx={{ height: 8, borderRadius: 4 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'security':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                    Security Overview
                  </Typography>
                  <Alert severity="info">
                    Advanced security features are active and protecting your account.
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );
      default:
        return renderOverview();
    }
  };

  const drawerItems = [
    { text: 'Overview', icon: <DashboardIcon />, value: 'overview' },
    { text: 'Security', icon: <SecurityIcon />, value: 'security' },
    { text: 'Settings', icon: <SettingsIcon />, value: 'settings' },
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: '#0f172a' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton color="inherit" edge="start" onClick={toggleDrawer} sx={{ mr: 2 }}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Aadhaar Dashboard
          </Typography>
          <IconButton color="inherit" onClick={handleSignOut}>
            <LogoutIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Drawer
        variant="temporary"
        open={drawerOpen}
        onClose={toggleDrawer}
        ModalProps={{ keepMounted: true }}
        sx={{
          '& .MuiDrawer-paper': {
            width: 280,
            bgcolor: '#1e293b',
            borderRight: '1px solid rgba(148, 163, 184, 0.1)',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', mt: 2 }}>
          <List>
            {drawerItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton
                  selected={activeTab === item.value}
                  onClick={() => {
                    setActiveTab(item.value);
                    setDrawerOpen(false);
                  }}
                  sx={{
                    mx: 1,
                    borderRadius: 2,
                    '&.Mui-selected': {
                      bgcolor: 'rgba(99, 102, 241, 0.1)',
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: activeTab === item.value ? '#6366f1' : '#cbd5e1' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Container maxWidth="xl">
          {renderContent()}
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard;
