---
description: Repository Information Overview
alwaysApply: true
---

# Login Page Application Information

## Summary
A React-based login page application that simulates an Aadhaar (Indian national ID) authentication system. The application includes user behavior tracking features that log mouse movements and login attempts. It consists of a React frontend built with Vite and a Node.js Express backend server.

## Structure
- **src/**: Contains React frontend code
  - **api/**: Backend API handlers
  - **assets/**: Static assets
  - **components/**: React components including the main LoginPage
  - **utils/**: Utility functions
- **public/**: Static public files
- **.vercel/**: Vercel deployment configuration
- **.zencoder/**: Zencoder configuration files

## Language & Runtime
**Language**: JavaScript (ES Modules)
**Frontend Framework**: React 19.1.0
**Build Tool**: Vite 7.0.4
**Backend Runtime**: Node.js with Express 4.18.2
**Package Manager**: npm

## Dependencies
**Main Dependencies**:
- React 19.1.0 and React DOM 19.1.0
- Material UI (@mui/material 7.3.0, @mui/icons-material 7.3.0)
- Express 4.18.2 for backend server
- Body-parser 1.20.2 for request parsing
- Concurrently 9.2.0 for running frontend and backend simultaneously

**Development Dependencies**:
- Vite 7.0.4 with React plugin (@vitejs/plugin-react 4.6.0)
- ESLint 9.30.1 with React plugins
- React type definitions (@types/react 19.1.8, @types/react-dom 19.1.6)

## Build & Installation
```bash
# Install dependencies
npm install

# Run development server (frontend + backend)
npm run dev

# Build for production
npm run build

# Run backend server only
npm run server

# Preview production build
npm run preview
```

## Server Configuration
**Backend Port**: 3001 (configurable via PORT environment variable)
**API Endpoints**:
- POST /api/log: Logs user activities including mouse movements and login attempts
**Proxy Configuration**: Vite development server proxies /api requests to the backend

## Main Files
**Frontend Entry**: src/main.jsx
**Main Component**: src/components/LoginPage.jsx
**Backend Server**: server.js
**API Handler**: src/api/log.js

## Data Storage
**Log Files**:
- web_logs.json: General web activity logs
- mouse_movements.json: User mouse movement tracking
- login_attempts.json: Authentication attempt records
- events.json: Combined event data