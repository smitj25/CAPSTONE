import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define log file paths
const LOGS_DIR = path.join(__dirname, '..', 'logs');
const WEB_LOGS_PATH = path.join(LOGS_DIR, 'web_logs.json');
const MOUSE_MOVEMENTS_PATH = path.join(LOGS_DIR, 'mouse_movements.json');
const LOGIN_ATTEMPTS_PATH = path.join(LOGS_DIR, 'login_attempts.json');
const EVENTS_PATH = path.join(LOGS_DIR, 'events.json');
const BEHAVIOR_PATH = path.join(LOGS_DIR, 'behavior.json');

// Helper function to read JSON file safely
async function readJsonFile(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      return [];
    }
    
    const fileContent = await fs.promises.readFile(filePath, 'utf8');
    if (!fileContent.trim()) {
      return [];
    }
    
    return JSON.parse(fileContent);
  } catch (error) {
    console.error(`Error reading file ${filePath}:`, error);
    return [];
  }
}

// Helper function to write JSON file safely
async function writeJsonFile(filePath, data) {
  try {
    await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2));
    return true;
  } catch (error) {
    console.error(`Error writing file ${filePath}:`, error);
    return false;
  }
}

// Function to automatically run ML detection when suspicious data is detected
async function runMLDetection(sessionId) {
  try {
    console.log(`🤖 Running automatic ML detection for session: ${sessionId}`);
    
    const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', '..', '..', 'src', 'core', 'optimized_bot_detection.py');
    
    // Determine the correct Python executable path (same logic as botDetection.js)
    const projectRoot = path.join(__dirname, '..', '..', '..');
    const venvPythonPath = path.join(projectRoot, '.venv', 'bin', 'python');
    const systemPythonPath = 'python3'; // Fallback to system python3
    
    // Try virtual environment Python first, then fallback to system python3
    let pythonCommand;
    try {
      if (fs.existsSync(venvPythonPath)) {
        pythonCommand = venvPythonPath;
        console.log('Using virtual environment Python for ML detection:', pythonCommand);
      } else {
        pythonCommand = systemPythonPath;
        console.log('Using system Python for ML detection:', pythonCommand);
      }
    } catch (error) {
      pythonCommand = systemPythonPath;
      console.log('Fallback to system Python for ML detection:', pythonCommand);
    }
    
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn(pythonCommand, [PYTHON_SCRIPT_PATH], {
        cwd: projectRoot,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          console.log('✅ ML detection completed successfully');
          resolve(output);
        } else {
          console.error('❌ ML detection failed:', errorOutput);
          reject(new Error(`ML detection failed with code ${code}: ${errorOutput}`));
        }
      });

      pythonProcess.on('error', (error) => {
        console.error('❌ Error running ML detection:', error);
        reject(error);
      });
    });
  } catch (error) {
    console.error('❌ Error in runMLDetection:', error);
    throw error;
  }
}

async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Ensure logs directory exists
    if (!fs.existsSync(LOGS_DIR)) {
      await fs.promises.mkdir(LOGS_DIR, { recursive: true });
    }

    const data = req.body;
    let updateSuccess = true;
    let eventsUpdated = false;
    let behaviorUpdated = false;
    let events = { mouseMovements: [], webLogs: [], sessionStats: null };
    
    // Handle session statistics if provided
    if (data.session_id && typeof data.total_requests !== 'undefined') {
      // This is a session statistics object
      const sessionStats = { ...data };
      
      // Write session stats directly to web_logs.json, replacing previous content
      const success = await writeJsonFile(WEB_LOGS_PATH, [sessionStats]);
      if (!success) updateSuccess = false;
      
      // Update events object with session stats
      events.sessionStats = sessionStats;
      eventsUpdated = true;
      
      // Early return since we've handled the session stats
      if (updateSuccess) {
        res.status(200).json({ message: 'Session statistics updated successfully' });
      } else {
        res.status(500).json({ message: 'Failed to update session statistics' });
      }
      return;
    }
    
    // Read events.json once at the beginning if it exists
    if (fs.existsSync(EVENTS_PATH)) {
      try {
        const eventsContent = await fs.promises.readFile(EVENTS_PATH, 'utf8');
        if (eventsContent.trim()) {
          events = JSON.parse(eventsContent);
        }
      } catch (error) {
        console.error('Error reading events.json:', error);
      }
    }
    
    // Handle web logs
    if (data.webLogEntry) {
      // Get current session ID
      const currentSessionId = data.webLogEntry.sessionId;
      
      // Read existing logs
      const allWebLogs = await readJsonFile(WEB_LOGS_PATH);
      
      // Filter logs to only keep entries from the current session
      const currentSessionLogs = allWebLogs.filter(log => log.sessionId === currentSessionId);
      
      // Add the new log entry
      currentSessionLogs.push(data.webLogEntry);
      
      // Write only the current session logs back to the file
      const success = await writeJsonFile(WEB_LOGS_PATH, currentSessionLogs);
      if (!success) updateSuccess = false;
      
      // Update events object with web logs
      events.webLogs = currentSessionLogs;
      eventsUpdated = true;
    }
    
    // Handle mouse movements
    if (data.mouseMovements) {
      // Get current session ID
      const currentSessionId = data.mouseMovements.sessionId;
      
      // Read existing mouse movements
      const allMouseMovements = await readJsonFile(MOUSE_MOVEMENTS_PATH);
      
      // Filter to only keep movements from the current session
      const currentSessionMovements = allMouseMovements.filter(movement => movement.sessionId === currentSessionId);
      
      // Add the new movement data
      currentSessionMovements.push(data.mouseMovements);
      
      // Write only the current session movements back to the file
      const success = await writeJsonFile(MOUSE_MOVEMENTS_PATH, currentSessionMovements);
      if (!success) updateSuccess = false;

      // Update events object with mouse movements
      events.mouseMovements = currentSessionMovements;
      eventsUpdated = true;
      
      // Check if this looks like suspicious bot behavior and trigger ML detection
      const movements = data.mouseMovements.movements || [];
      if (movements.length > 50) { // If we have enough data for analysis
        console.log(`🚨 Suspicious mouse movements detected (${movements.length} movements)`);
        
        // Run ML detection in background (don't wait for it)
        runMLDetection(currentSessionId).catch(error => {
          console.error('❌ Background ML detection failed:', error);
        });
      }
    }
    
    // Handle behavior signals (keystroke timing, focus/blur, scroll variance, click trust)
    if (data.behaviorSignals) {
      const currentSessionId = data.behaviorSignals.sessionId;
      const allBehavior = await readJsonFile(BEHAVIOR_PATH);
      const currentSessionBehavior = allBehavior.filter(b => b.sessionId === currentSessionId);
      currentSessionBehavior.push({ ...data.behaviorSignals, timestamp: new Date().toISOString() });
      const success = await writeJsonFile(BEHAVIOR_PATH, currentSessionBehavior);
      if (!success) updateSuccess = false;
      behaviorUpdated = true;
    }

    // Handle login attempts
    if (data.eventType === 'login_attempt') {
      // Get current session ID
      const currentSessionId = data.sessionId || '-';
      
      // Read existing login attempts
      const allLoginAttempts = await readJsonFile(LOGIN_ATTEMPTS_PATH);
      
      // Filter to only keep login attempts from the current session
      const currentSessionAttempts = allLoginAttempts.filter(attempt => attempt.sessionId === currentSessionId);
      
      // Get the most recent mouse movements for this session
      const mouseMovements = await readJsonFile(MOUSE_MOVEMENTS_PATH);
      const sessionMouseMovements = mouseMovements
        .filter(movement => movement.sessionId === currentSessionId)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      // Create the login attempt entry with mouse movement data
      const loginEntry = {
        timestamp: data.timestamp,
        details: {
          ...data.details,
          mouseMovementData: sessionMouseMovements.length > 0 ? sessionMouseMovements[0].movements : []
        },
        sessionId: currentSessionId,
        userAgent: data.userAgent || '-'
      };
      
      // Add the new login attempt
      currentSessionAttempts.push(loginEntry);
      
      // Write only the current session attempts back to the file
      const success = await writeJsonFile(LOGIN_ATTEMPTS_PATH, currentSessionAttempts);
      if (!success) updateSuccess = false;
      
      // Trigger ML detection for login attempts
      console.log(`🔐 Login attempt detected for session: ${currentSessionId}`);
      
      // Run ML detection in background
      runMLDetection(currentSessionId).catch(error => {
        console.error('❌ Background ML detection failed:', error);
      });
    }
    
    // Write updated events to events.json if any changes were made
    if (eventsUpdated) {
      try {
        // Only store events from the current session
        const currentSessionId = data.sessionId || data.session_id || 
                               (data.webLogEntry ? data.webLogEntry.sessionId : null) || 
                               (data.mouseMovements ? data.mouseMovements.sessionId : null) || '-';
        
        // Filter events to only include current session data
        const currentSessionEvents = {
          mouseMovements: events.mouseMovements.filter(movement => movement.sessionId === currentSessionId),
          webLogs: events.webLogs.filter(log => log.sessionId === currentSessionId),
          sessionStats: events.sessionStats
        };
        
        const success = await writeJsonFile(EVENTS_PATH, currentSessionEvents);
        if (!success) updateSuccess = false;
      } catch (error) {
        console.error('Error writing to events.json:', error);
        updateSuccess = false;
      }
    }

    if (updateSuccess) {
      res.status(200).json({ message: 'Logs updated successfully' });
    } else {
      res.status(500).json({ message: 'Some logs failed to update' });
    }
  } catch (error) {
    console.error('Error handling logs:', error);
    res.status(500).json({ message: 'Error updating logs' });
  }
}

export default handler;