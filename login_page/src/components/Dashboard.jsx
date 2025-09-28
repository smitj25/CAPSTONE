import React, { useState, useEffect } from 'react';
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
  Alert,
  Chip,
  Divider,
  Badge,
  Tooltip,
  CircularProgress
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
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  Shield as ShieldIcon,
  Computer as ComputerIcon,
  Timeline as TimelineIcon,
  Notifications as NotificationsIcon,
  Speed as SpeedIcon,
  Lock as LockIcon,
  Visibility as VisibilityIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';

const Dashboard = ({ onSignOut, userData }) => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [securityMetrics, setSecurityMetrics] = useState({
    botDetectionRate: 98.5,
    threatsPrevented: 247,
    securityScore: 95,
    lastScan: new Date().toLocaleString()
  });

  const handleSignOut = () => {
    if (onSignOut) {
      onSignOut();
    }
  };

  const toggleDrawer = () => setDrawerOpen(!drawerOpen);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSecurityMetrics(prev => ({
        ...prev,
        botDetectionRate: Math.min(100, prev.botDetectionRate + (Math.random() - 0.5) * 0.1),
        threatsPrevented: prev.threatsPrevented + Math.floor(Math.random() * 2),
        lastScan: new Date().toLocaleString()
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const StatCard = ({ title, value, subtitle, icon, color, trend }) => (
    <Card 
      sx={{ 
        height: '100%',
        background: `linear-gradient(135deg, ${color}15 0%, ${color}05 100%)`,
        border: `1px solid ${color}20`,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 25px ${color}25`,
          border: `1px solid ${color}40`,
        }
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h3" sx={{ fontWeight: 700, mb: 0.5, color: color }}>
              {value}
            </Typography>
            <Typography variant="h6" sx={{ color: '#cbd5e1', fontWeight: 500, mb: 1 }}>
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="body2" sx={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box sx={{ 
            p: 1.5, 
            borderRadius: 2, 
            bgcolor: `${color}20`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {React.cloneElement(icon, { sx: { fontSize: 28, color: color } })}
          </Box>
        </Box>
        {trend && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUpIcon sx={{ fontSize: 16, color: '#10b981' }} />
            <Typography variant="caption" sx={{ color: '#10b981', fontWeight: 600 }}>
              {trend}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderOverview = () => (
    <Grid container spacing={3}>
      {/* Welcome Header */}
      <Grid item xs={12}>
        <Card sx={{ 
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%)',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
          }
        }}>
          <CardContent sx={{ p: 4, position: 'relative', zIndex: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ flex: 1, minWidth: 250 }}>
                <Typography variant="h3" sx={{ color: '#ffffff', mb: 1, fontWeight: 700 }}>
                  Welcome back! 👋
                </Typography>
                <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500, mb: 2 }}>
                  Aadhaar ID: {userData?.aadhaarNumber || '**** **** ****'}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip 
                    label="Verified Account" 
                    sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: '#ffffff', fontWeight: 600 }}
                    icon={<CheckCircleIcon sx={{ color: '#ffffff !important' }} />}
                  />
                  <Chip 
                    label="Premium Security" 
                    sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: '#ffffff', fontWeight: 600 }}
                    icon={<ShieldIcon sx={{ color: '#ffffff !important' }} />}
                  />
                </Box>
              </Box>
              <Avatar sx={{ 
                width: 80, 
                height: 80, 
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                border: '3px solid rgba(255, 255, 255, 0.3)'
              }}>
                <PersonIcon sx={{ fontSize: 40, color: '#ffffff' }} />
              </Avatar>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Key Metrics */}
      <Grid item xs={12} sm={6} md={3}>
        <StatCard
          title="Total Logins"
          value="127"
          subtitle="This month"
          icon={<ComputerIcon />}
          color="#6366f1"
          trend="+12% from last month"
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <StatCard
          title="Security Score"
          value={`${securityMetrics.securityScore}%`}
          subtitle="Excellent protection"
          icon={<ShieldIcon />}
          color="#10b981"
          trend="+2% this week"
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <StatCard
          title="Threats Blocked"
          value={securityMetrics.threatsPrevented}
          subtitle="All time"
          icon={<SecurityIcon />}
          color="#ef4444"
          trend="+5 today"
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <StatCard
          title="Bot Detection"
          value={`${securityMetrics.botDetectionRate.toFixed(1)}%`}
          subtitle="Success rate"
          icon={<SpeedIcon />}
          color="#f59e0b"
          trend="Real-time active"
        />
      </Grid>

      {/* Security Status Dashboard */}
      <Grid item xs={12} md={8}>
        <Card sx={{ height: '100%' }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
              <Typography variant="h5" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <SecurityIcon sx={{ color: '#6366f1' }} />
                Security Dashboard
              </Typography>
              <Chip 
                label="Live Monitoring" 
                color="success" 
                size="small"
                icon={<CircularProgress size={12} sx={{ color: '#ffffff !important' }} />}
              />
            </Box>

            <Alert severity="success" sx={{ mb: 3, borderRadius: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                🛡️ Your account is fully protected with advanced ML-based bot detection
              </Typography>
            </Alert>

            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 600 }}>
                      Bot Detection System
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#10b981', fontWeight: 700 }}>
                      {securityMetrics.botDetectionRate.toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={securityMetrics.botDetectionRate} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      bgcolor: 'rgba(16, 185, 129, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: '#10b981',
                        borderRadius: 4,
                      }
                    }} 
                  />
                </Box>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 600 }}>
                      Threat Prevention
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#6366f1', fontWeight: 700 }}>
                      98%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={98} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      bgcolor: 'rgba(99, 102, 241, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: '#6366f1',
                        borderRadius: 4,
                      }
                    }} 
                  />
                </Box>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 600 }}>
                      Data Encryption
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#8b5cf6', fontWeight: 700 }}>
                      100%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={100} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      bgcolor: 'rgba(139, 92, 246, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: '#8b5cf6',
                        borderRadius: 4,
                      }
                    }} 
                  />
                </Box>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 600 }}>
                      Access Control
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#06b6d4', fontWeight: 700 }}>
                      100%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={100} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      bgcolor: 'rgba(6, 182, 212, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: '#06b6d4',
                        borderRadius: 4,
                      }
                    }} 
                  />
                </Box>
              </Grid>
            </Grid>

            <Divider sx={{ my: 2, bgcolor: 'rgba(148, 163, 184, 0.1)' }} />
            
            <Typography variant="caption" sx={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 1 }}>
              <InfoIcon sx={{ fontSize: 14 }} />
              Last security scan: {securityMetrics.lastScan}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Quick Actions */}
      <Grid item xs={12} md={4}>
        <Card sx={{ height: '100%' }}>
          <CardContent sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
              <AnalyticsIcon sx={{ color: '#6366f1' }} />
              Quick Actions
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<SecurityIcon />}
                sx={{ 
                  py: 1.5,
                  borderColor: '#6366f1',
                  color: '#6366f1',
                  '&:hover': {
                    bgcolor: 'rgba(99, 102, 241, 0.1)',
                    borderColor: '#6366f1',
                  }
                }}
              >
                Run Security Scan
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                startIcon={<TimelineIcon />}
                sx={{ 
                  py: 1.5,
                  borderColor: '#10b981',
                  color: '#10b981',
                  '&:hover': {
                    bgcolor: 'rgba(16, 185, 129, 0.1)',
                    borderColor: '#10b981',
                  }
                }}
              >
                View Analytics
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                startIcon={<SettingsIcon />}
                sx={{ 
                  py: 1.5,
                  borderColor: '#8b5cf6',
                  color: '#8b5cf6',
                  '&:hover': {
                    bgcolor: 'rgba(139, 92, 246, 0.1)',
                    borderColor: '#8b5cf6',
                  }
                }}
              >
                Security Settings
              </Button>
            </Box>

            <Divider sx={{ my: 3, bgcolor: 'rgba(148, 163, 184, 0.1)' }} />

            <Typography variant="subtitle2" sx={{ mb: 2, color: '#cbd5e1', fontWeight: 600 }}>
              Recent Activity
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1.5, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 1 }}>
                <CheckCircleIcon sx={{ color: '#10b981', fontSize: 20 }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                    Login successful
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                    2 minutes ago
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1.5, bgcolor: 'rgba(99, 102, 241, 0.1)', borderRadius: 1 }}>
                <ShieldIcon sx={{ color: '#6366f1', fontSize: 20 }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                    Security scan completed
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                    1 hour ago
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 1.5, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 1 }}>
                <WarningIcon sx={{ color: '#ef4444', fontSize: 20 }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                    Bot attempt blocked
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                    3 hours ago
                  </Typography>
                </Box>
              </Box>
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
              <Card sx={{ 
                background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <CardContent sx={{ p: 4, position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                    <Box>
                      <Typography variant="h4" sx={{ color: '#ffffff', mb: 1, fontWeight: 700 }}>
                        Security Center 🛡️
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }}>
                        Advanced threat protection and monitoring
                      </Typography>
                    </Box>
                    <SecurityIcon sx={{ fontSize: 60, color: 'rgba(255, 255, 255, 0.3)' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Security Metrics */}
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Active Threats"
                value="0"
                subtitle="Currently detected"
                icon={<WarningIcon />}
                color="#10b981"
                trend="All clear"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Firewall Status"
                value="Active"
                subtitle="Real-time protection"
                icon={<ShieldIcon />}
                color="#6366f1"
                trend="100% uptime"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Login Attempts"
                value="3"
                subtitle="Last 24 hours"
                icon={<LockIcon />}
                color="#f59e0b"
                trend="All authorized"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Data Integrity"
                value="100%"
                subtitle="Verification complete"
                icon={<CheckCircleIcon />}
                color="#8b5cf6"
                trend="Verified"
              />
            </Grid>

            {/* Detailed Security Analytics */}
            <Grid item xs={12} md={8}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AnalyticsIcon sx={{ color: '#6366f1' }} />
                    Security Analytics
                  </Typography>

                  <Alert severity="success" sx={{ mb: 3, borderRadius: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      🔒 All security systems operational. No threats detected in the last 30 days.
                    </Typography>
                  </Alert>

                  <Box sx={{ mb: 4 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600, color: '#cbd5e1' }}>
                      Real-time Security Monitoring
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Box sx={{ p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 2, border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                            <CheckCircleIcon sx={{ color: '#10b981' }} />
                            <Typography variant="body1" sx={{ fontWeight: 600, color: '#cbd5e1' }}>
                              ML Bot Detection
                            </Typography>
                          </Box>
                          <Typography variant="h4" sx={{ color: '#10b981', fontWeight: 700, mb: 1 }}>
                            {securityMetrics.botDetectionRate.toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                            Detection accuracy rate
                          </Typography>
                        </Box>
                      </Grid>

                      <Grid item xs={12} sm={6}>
                        <Box sx={{ p: 2, bgcolor: 'rgba(99, 102, 241, 0.1)', borderRadius: 2, border: '1px solid rgba(99, 102, 241, 0.2)' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                            <ShieldIcon sx={{ color: '#6366f1' }} />
                            <Typography variant="body1" sx={{ fontWeight: 600, color: '#cbd5e1' }}>
                              Threat Prevention
                            </Typography>
                          </Box>
                          <Typography variant="h4" sx={{ color: '#6366f1', fontWeight: 700, mb: 1 }}>
                            {securityMetrics.threatsPrevented}
                          </Typography>
                          <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                            Threats blocked total
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>

                  <Divider sx={{ my: 3, bgcolor: 'rgba(148, 163, 184, 0.1)' }} />

                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600, color: '#cbd5e1' }}>
                    Security Event Log
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', borderRadius: 1 }}>
                      <CheckCircleIcon sx={{ color: '#10b981' }} />
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                          Successful authentication - Aadhaar verification completed
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                          Today at {new Date().toLocaleTimeString()}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 1 }}>
                      <WarningIcon sx={{ color: '#ef4444' }} />
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                          Suspicious bot activity detected and blocked
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                          3 hours ago
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2, bgcolor: 'rgba(99, 102, 241, 0.1)', borderRadius: 1 }}>
                      <SecurityIcon sx={{ color: '#6366f1' }} />
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" sx={{ color: '#cbd5e1', fontWeight: 500 }}>
                          Security scan completed - All systems secure
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                          6 hours ago
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Security Controls */}
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SettingsIcon sx={{ color: '#6366f1' }} />
                    Security Controls
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 3 }}>
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={<SecurityIcon />}
                      sx={{ 
                        py: 1.5,
                        bgcolor: '#ef4444',
                        '&:hover': { bgcolor: '#dc2626' }
                      }}
                    >
                      Emergency Lockdown
                    </Button>

                    <Button
                      variant="outlined"
                      fullWidth
                      startIcon={<ShieldIcon />}
                      sx={{ 
                        py: 1.5,
                        borderColor: '#6366f1',
                        color: '#6366f1',
                        '&:hover': {
                          bgcolor: 'rgba(99, 102, 241, 0.1)',
                          borderColor: '#6366f1',
                        }
                      }}
                    >
                      Update Security Rules
                    </Button>

                    <Button
                      variant="outlined"
                      fullWidth
                      startIcon={<VisibilityIcon />}
                      sx={{ 
                        py: 1.5,
                        borderColor: '#10b981',
                        color: '#10b981',
                        '&:hover': {
                          bgcolor: 'rgba(16, 185, 129, 0.1)',
                          borderColor: '#10b981',
                        }
                      }}
                    >
                      View Audit Log
                    </Button>
                  </Box>

                  <Divider sx={{ my: 3, bgcolor: 'rgba(148, 163, 184, 0.1)' }} />

                  <Typography variant="subtitle2" sx={{ mb: 2, color: '#cbd5e1', fontWeight: 600 }}>
                    Security Status
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        Two-Factor Auth
                      </Typography>
                      <Chip label="Enabled" color="success" size="small" />
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        Bot Detection
                      </Typography>
                      <Chip label="Active" color="success" size="small" />
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        Data Encryption
                      </Typography>
                      <Chip label="AES-256" color="info" size="small" />
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        Session Timeout
                      </Typography>
                      <Chip label="30 min" color="warning" size="small" />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );
      case 'settings':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card sx={{ 
                background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <CardContent sx={{ p: 4, position: 'relative', zIndex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                    <Box>
                      <Typography variant="h4" sx={{ color: '#ffffff', mb: 1, fontWeight: 700 }}>
                        Settings & Preferences ⚙️
                      </Typography>
                      <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }}>
                        Customize your dashboard experience
                      </Typography>
                    </Box>
                    <SettingsIcon sx={{ fontSize: 60, color: 'rgba(255, 255, 255, 0.3)' }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Account Settings */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PersonIcon sx={{ color: '#6366f1' }} />
                    Account Settings
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 1, color: '#cbd5e1', fontWeight: 600 }}>
                        Profile Information
                      </Typography>
                      <Box sx={{ p: 2, bgcolor: 'rgba(99, 102, 241, 0.1)', borderRadius: 2, border: '1px solid rgba(99, 102, 241, 0.2)' }}>
                        <Typography variant="body2" sx={{ color: '#cbd5e1', mb: 1 }}>
                          Aadhaar Number: {userData?.aadhaarNumber || '**** **** ****'}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                          Last updated: {new Date().toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 2, color: '#cbd5e1', fontWeight: 600 }}>
                        Security Preferences
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, bgcolor: 'rgba(148, 163, 184, 0.05)', borderRadius: 1 }}>
                          <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                            Email Notifications
                          </Typography>
                          <Chip label="Enabled" color="success" size="small" />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, bgcolor: 'rgba(148, 163, 184, 0.05)', borderRadius: 1 }}>
                          <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                            SMS Alerts
                          </Typography>
                          <Chip label="Enabled" color="success" size="small" />
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, bgcolor: 'rgba(148, 163, 184, 0.05)', borderRadius: 1 }}>
                          <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                            Auto-logout
                          </Typography>
                          <Chip label="30 minutes" color="info" size="small" />
                        </Box>
                      </Box>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* System Settings */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ComputerIcon sx={{ color: '#6366f1' }} />
                    System Settings
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 2, color: '#cbd5e1', fontWeight: 600 }}>
                        Dashboard Preferences
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Button
                          variant="outlined"
                          fullWidth
                          sx={{ 
                            py: 1.5,
                            borderColor: '#6366f1',
                            color: '#6366f1',
                            '&:hover': {
                              bgcolor: 'rgba(99, 102, 241, 0.1)',
                              borderColor: '#6366f1',
                            }
                          }}
                        >
                          Change Theme
                        </Button>
                        <Button
                          variant="outlined"
                          fullWidth
                          sx={{ 
                            py: 1.5,
                            borderColor: '#10b981',
                            color: '#10b981',
                            '&:hover': {
                              bgcolor: 'rgba(16, 185, 129, 0.1)',
                              borderColor: '#10b981',
                            }
                          }}
                        >
                          Export Data
                        </Button>
                        <Button
                          variant="outlined"
                          fullWidth
                          sx={{ 
                            py: 1.5,
                            borderColor: '#f59e0b',
                            color: '#f59e0b',
                            '&:hover': {
                              bgcolor: 'rgba(245, 158, 11, 0.1)',
                              borderColor: '#f59e0b',
                            }
                          }}
                        >
                          Reset Dashboard
                        </Button>
                      </Box>
                    </Box>

                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 2, color: '#cbd5e1', fontWeight: 600 }}>
                        System Information
                      </Typography>
                      <Box sx={{ p: 2, bgcolor: 'rgba(148, 163, 184, 0.05)', borderRadius: 2 }}>
                        <Typography variant="body2" sx={{ color: '#94a3b8', mb: 1 }}>
                          Version: 2.1.0
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#94a3b8', mb: 1 }}>
                          Last Update: {new Date().toLocaleDateString()}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                          Server Status: Online
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
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
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
        }}
      >
        <Toolbar>
          <IconButton 
            color="inherit" 
            edge="start" 
            onClick={toggleDrawer} 
            sx={{ 
              mr: 2,
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.1)',
                transform: 'scale(1.05)'
              },
              transition: 'all 0.2s ease'
            }}
          >
            <MenuIcon />
          </IconButton>
          <Typography 
            variant="h6" 
            noWrap 
            component="div" 
            sx={{ 
              flexGrow: 1,
              fontWeight: 700,
              background: 'linear-gradient(45deg, #ffffff 30%, #e2e8f0 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}
          >
            🛡️ Aadhaar Security Dashboard
          </Typography>
          <Badge 
            badgeContent="Live" 
            color="success" 
            sx={{ 
              mr: 2,
              '& .MuiBadge-badge': {
                animation: 'pulse 2s infinite',
                '@keyframes pulse': {
                  '0%': { transform: 'scale(1)', opacity: 1 },
                  '50%': { transform: 'scale(1.1)', opacity: 0.7 },
                  '100%': { transform: 'scale(1)', opacity: 1 }
                }
              }
            }}
          >
            <NotificationsIcon sx={{ color: '#e2e8f0' }} />
          </Badge>
          <Tooltip title="Sign Out" arrow>
            <IconButton 
              color="inherit" 
              onClick={handleSignOut}
              sx={{
                '&:hover': {
                  bgcolor: 'rgba(239, 68, 68, 0.2)',
                  transform: 'scale(1.05)'
                },
                transition: 'all 0.2s ease'
              }}
            >
              <LogoutIcon />
            </IconButton>
          </Tooltip>
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
            background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
            backdropFilter: 'blur(10px)',
            boxShadow: '4px 0 20px rgba(0, 0, 0, 0.15)'
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', mt: 2, px: 1 }}>
          {/* User Profile Section */}
          <Box sx={{ p: 2, mb: 2, textAlign: 'center' }}>
            <Avatar 
              sx={{ 
                width: 60, 
                height: 60, 
                mx: 'auto',
                mb: 2,
                bgcolor: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                border: '3px solid rgba(99, 102, 241, 0.3)',
                boxShadow: '0 4px 15px rgba(99, 102, 241, 0.3)'
              }}
            >
              <PersonIcon sx={{ fontSize: 30 }} />
            </Avatar>
            <Typography variant="subtitle1" sx={{ color: '#e2e8f0', fontWeight: 600, mb: 0.5 }}>
              Secure User
            </Typography>
            <Chip 
              label="Verified" 
              size="small" 
              sx={{ 
                bgcolor: 'rgba(16, 185, 129, 0.2)', 
                color: '#10b981',
                fontWeight: 600,
                border: '1px solid rgba(16, 185, 129, 0.3)'
              }}
              icon={<CheckCircleIcon sx={{ color: '#10b981 !important', fontSize: 16 }} />}
            />
          </Box>

          <Divider sx={{ bgcolor: 'rgba(148, 163, 184, 0.1)', mb: 2 }} />

          <List>
            {drawerItems.map((item) => (
              <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
                <ListItemButton
                  selected={activeTab === item.value}
                  onClick={() => {
                    setActiveTab(item.value);
                    setDrawerOpen(false);
                  }}
                  sx={{
                    mx: 1,
                    borderRadius: 3,
                    py: 1.5,
                    transition: 'all 0.3s ease',
                    '&.Mui-selected': {
                      bgcolor: 'rgba(99, 102, 241, 0.15)',
                      border: '1px solid rgba(99, 102, 241, 0.3)',
                      transform: 'translateX(4px)',
                      boxShadow: '0 4px 15px rgba(99, 102, 241, 0.2)'
                    },
                    '&:hover': {
                      bgcolor: 'rgba(148, 163, 184, 0.1)',
                      transform: 'translateX(2px)'
                    }
                  }}
                >
                  <ListItemIcon sx={{ 
                    color: activeTab === item.value ? '#6366f1' : '#cbd5e1',
                    minWidth: 40,
                    transition: 'color 0.3s ease'
                  }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.text} 
                    sx={{
                      '& .MuiTypography-root': {
                        fontWeight: activeTab === item.value ? 600 : 500,
                        color: activeTab === item.value ? '#e2e8f0' : '#cbd5e1'
                      }
                    }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>

          <Box sx={{ mt: 'auto', p: 2 }}>
            <Alert 
              severity="info" 
              sx={{ 
                bgcolor: 'rgba(6, 182, 212, 0.1)',
                border: '1px solid rgba(6, 182, 212, 0.2)',
                '& .MuiAlert-icon': { color: '#06b6d4' },
                '& .MuiAlert-message': { color: '#cbd5e1', fontSize: '0.8rem' }
              }}
            >
              System Status: All secure 🔒
            </Alert>
          </Box>
        </Box>
      </Drawer>

      <Box 
        component="main" 
        sx={{ 
          flexGrow: 1, 
          p: { xs: 2, sm: 3 },
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
          minHeight: '100vh',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.05) 0%, transparent 50%)',
            pointerEvents: 'none',
          }
        }}
      >
        <Toolbar />
        <Container 
          maxWidth="xl" 
          sx={{ 
            position: 'relative', 
            zIndex: 1,
            mt: { xs: 1, sm: 2 }
          }}
        >
          {renderContent()}
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard;
