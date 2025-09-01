import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to the Python script (relative to the project root) - Updated for new structure
const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', '..', '..', 'src', 'tests', 'test.py');

// Cache for bot detection results to avoid redundant processing
const detectionCache = new Map();
const CACHE_TTL = 30000; // 30 seconds cache

// Rate limiting to prevent excessive requests
const requestLimiter = new Map();
const RATE_LIMIT_WINDOW = 5000; // 5 seconds
const MAX_REQUESTS_PER_WINDOW = 3; // Max 3 requests per 5 seconds

// Optimized Python script path for faster processing - Updated for new structure
const OPTIMIZED_PYTHON_SCRIPT_PATH = path.join(__dirname, '..', '..', '..', 'src', 'core', 'optimized_bot_detection.py');

async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { sessionId } = req.body;
    
    // Rate limiting check
    const clientId = req.headers['x-forwarded-for'] || req.connection.remoteAddress || 'unknown';
    const now = Date.now();
    
    if (!requestLimiter.has(clientId)) {
      requestLimiter.set(clientId, { count: 0, resetTime: now + RATE_LIMIT_WINDOW });
    }
    
    const clientLimit = requestLimiter.get(clientId);
    if (now > clientLimit.resetTime) {
      clientLimit.count = 0;
      clientLimit.resetTime = now + RATE_LIMIT_WINDOW;
    }
    
    if (clientLimit.count >= MAX_REQUESTS_PER_WINDOW) {
      return res.status(429).json({
        success: false,
        error: 'Rate limit exceeded',
        details: 'Too many requests. Please wait a few seconds and try again.'
      });
    }
    
    clientLimit.count++;
    
    // Check cache first
    const cacheKey = `session_${sessionId}`;
    const cachedResult = detectionCache.get(cacheKey);
    if (cachedResult && (Date.now() - cachedResult.timestamp) < CACHE_TTL) {
      console.log('Returning cached bot detection result');
      return res.status(200).json({
        success: true,
        results: cachedResult.data,
        cached: true
      });
    }

    console.log('Starting optimized bot detection analysis...');
    
    // Use optimized Python script if available, otherwise fall back to original
    const scriptPath = fs.existsSync(OPTIMIZED_PYTHON_SCRIPT_PATH) 
      ? OPTIMIZED_PYTHON_SCRIPT_PATH 
      : PYTHON_SCRIPT_PATH;
    
    // Run the Python ML script with optimized parameters
    const pythonProcess = spawn('python', [scriptPath, '--fast', '--session-id', sessionId], {
      cwd: path.join(__dirname, '..', '..', '..'), // Set working directory to project root
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { 
        ...process.env, 
        PYTHONIOENCODING: 'utf-8',
        PYTHONUNBUFFERED: '1', // Disable output buffering for faster response
        OMP_NUM_THREADS: '4', // Limit OpenMP threads for better performance
        TF_CPP_MIN_LOG_LEVEL: '2' // Reduce TensorFlow logging
      }
    });

    let output = '';
    let errorOutput = '';

    // Set a timeout for the Python process
    const timeout = setTimeout(() => {
      pythonProcess.kill('SIGTERM');
      console.error('Bot detection timed out');
      res.status(408).json({
        success: false,
        error: 'Bot detection timed out',
        details: 'Processing took too long'
      });
    }, 15000); // 15 second timeout

    // Collect stdout
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    // Collect stderr
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    // Handle process completion
    pythonProcess.on('close', (code) => {
      clearTimeout(timeout);
      
      if (code === 0) {
        // Parse the output to extract results
        const results = parsePythonOutput(output);
        
        // Cache the result
        detectionCache.set(cacheKey, {
          data: results,
          timestamp: Date.now()
        });
        
        // Clean up old cache entries
        cleanupCache();
        
        console.log('Bot detection completed successfully');
        res.status(200).json({
          success: true,
          results: results,
          rawOutput: output,
          processingTime: Date.now() - req.startTime
        });
      } else {
        console.error('Python script failed with code:', code);
        console.error('Error output:', errorOutput);
        res.status(500).json({
          success: false,
          error: 'Bot detection failed',
          details: errorOutput
        });
      }
    });

    // Handle process errors
    pythonProcess.on('error', (error) => {
      clearTimeout(timeout);
      console.error('Failed to start Python process:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to start bot detection process',
        details: error.message
      });
    });

  } catch (error) {
    console.error('Error in bot detection handler:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      details: error.message
    });
  }
}

function cleanupCache() {
  const now = Date.now();
  for (const [key, value] of detectionCache.entries()) {
    if (now - value.timestamp > CACHE_TTL) {
      detectionCache.delete(key);
    }
  }
}

function parsePythonOutput(output) {
  const results = {
    mouseMovements: [],
    webLogs: null,
    loginAttempts: [],
    summary: {},
    processingTime: null
  };

  const lines = output.split('\n');
  let currentSection = '';
  let currentSession = null;

  for (const line of lines) {
    // Parse processing time
    if (line.includes('Processing time:')) {
      const timeMatch = line.match(/Processing time: ([\d.]+)s/);
      if (timeMatch) {
        results.processingTime = parseFloat(timeMatch[1]);
      }
    }
    // Parse mouse movement results
    if (line.includes('Session') && line.includes('Mouse movements:')) {
      const sessionMatch = line.match(/Session (\d+): (\w+)/);
      const movementsMatch = line.match(/Mouse movements: (\d+)/);
      const scoreMatch = line.match(/Bot Score: ([\d.]+)/);
      const classificationMatch = line.match(/Classification: (\w+)/);
      
      if (sessionMatch && movementsMatch && scoreMatch && classificationMatch) {
        results.mouseMovements.push({
          sessionId: sessionMatch[2],
          movements: parseInt(movementsMatch[1]),
          botScore: parseFloat(scoreMatch[1]),
          classification: classificationMatch[1]
        });
      }
    }

    // Parse web log results
    if (line.includes('Session:') && line.includes('Log entries:')) {
      const sessionMatch = line.match(/Session: (\w+)/);
      const entriesMatch = line.match(/Log entries: (\d+)/);
      const scoreMatch = line.match(/Bot Score: ([\d.]+)/);
      const classificationMatch = line.match(/Classification: (\w+)/);
      
      if (sessionMatch && entriesMatch && scoreMatch && classificationMatch) {
        results.webLogs = {
          sessionId: sessionMatch[1],
          entries: parseInt(entriesMatch[1]),
          botScore: parseFloat(scoreMatch[1]),
          classification: classificationMatch[1]
        };
      }
    }

    // Parse login attempt results
    if (line.includes('Login Attempt') && line.includes('Mouse Score:')) {
      const attemptMatch = line.match(/Login Attempt (\d+)/);
      const mouseScoreMatch = line.match(/Mouse Score: ([\d.]+)/);
      const webScoreMatch = line.match(/Web Score: ([\d.]+)/);
      const fusionScoreMatch = line.match(/Fusion Score: ([\d.]+)/);
      const classificationMatch = line.match(/Final Classification: (\w+)/);
      
      if (attemptMatch && mouseScoreMatch && webScoreMatch && fusionScoreMatch && classificationMatch) {
        results.loginAttempts.push({
          attemptNumber: parseInt(attemptMatch[1]),
          mouseScore: parseFloat(mouseScoreMatch[1]),
          webScore: parseFloat(webScoreMatch[1]),
          fusionScore: parseFloat(fusionScoreMatch[1]),
          classification: classificationMatch[1]
        });
      }
    }

    // Parse summary
    if (line.includes('Mouse Movements:') && line.includes('sessions')) {
      const mouseMatch = line.match(/Mouse Movements: (\d+) sessions, (\d+) detected as bots/);
      if (mouseMatch) {
        results.summary.mouseMovements = {
          sessions: parseInt(mouseMatch[1]),
          botsDetected: parseInt(mouseMatch[2])
        };
      }
    }

    if (line.includes('Web Logs:') && line.includes('sessions')) {
      const webMatch = line.match(/Web Logs: (\d+) sessions, (\d+) detected as bots/);
      if (webMatch) {
        results.summary.webLogs = {
          sessions: parseInt(webMatch[1]),
          botsDetected: parseInt(webMatch[2])
        };
      }
    }

    if (line.includes('Login Attempts:') && line.includes('attempts')) {
      const loginMatch = line.match(/Login Attempts: (\d+) attempts, (\d+) detected as bots/);
      if (loginMatch) {
        results.summary.loginAttempts = {
          attempts: parseInt(loginMatch[1]),
          botsDetected: parseInt(loginMatch[2])
        };
      }
    }
  }

  return results;
}

export default handler; 