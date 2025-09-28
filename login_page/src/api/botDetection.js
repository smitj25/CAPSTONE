import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import fetch from 'node-fetch';
import ENTERPRISE_CONFIG from '../config/enterprise-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to the Python script (relative to the project root) - Updated for new structure
const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', '..', '..', 'src', 'core', 'optimized_bot_detection.py');

// Cache for bot detection results to avoid redundant processing
const detectionCache = new Map();
const CACHE_TTL = 0; // Disabled during testing for fresh results each run

// Rate limiting to prevent excessive requests
const requestLimiter = new Map();
const RATE_LIMIT_WINDOW = 5000; // 5 seconds
const MAX_REQUESTS_PER_WINDOW = 3; // Max 3 requests per 5 seconds

// Optimized Python script path for faster processing - Updated for new structure
const OPTIMIZED_PYTHON_SCRIPT_PATH = path.join(__dirname, '..', '..', '..', 'src', 'core', 'optimized_bot_detection.py');

// reCAPTCHA Enterprise configuration
const RECAPTCHA_CONFIG = {
  ...ENTERPRISE_CONFIG,
  scoreThresholds: {
    high: 0.7,      // High confidence human
    medium: 0.5,    // Medium confidence
    low: 0.3,       // Low confidence - likely bot
    critical: 0.1   // Very likely bot
  }
};

/**
 * Verify reCAPTCHA Enterprise token with Google
 * @param {string} token - reCAPTCHA token from frontend
 * @param {string} expectedAction - Expected action from frontend
 * @returns {Object} - Verification result with score analysis
 */
async function verifyRecaptcha(token, expectedAction = 'PAGEVIEW') {
  if (!token || token === 'YOUR_RECAPTCHA_SITE_KEY') {
    return {
      success: false,
      error: 'No reCAPTCHA token provided',
      score: 0.5, // Neutral score when no token
      analysis: {
        score: 0.5,
        confidence: 'unknown',
        recommendation: 'proceed',
        riskLevel: 'medium',
        action: 'none'
      }
    };
  }

  try {
    // Create request body using enterprise config
    const requestBody = RECAPTCHA_CONFIG.createRequestBody(token, expectedAction);
    const url = RECAPTCHA_CONFIG.getAssessmentUrl();
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    const data = await response.json();
    
    if (response.ok && data.tokenProperties && data.tokenProperties.valid) {
      const score = data.riskAnalysis?.score || 0.5;
      const analysis = analyzeRecaptchaScore(score);
      
      return {
        success: true,
        score: score,
        action: data.tokenProperties.action,
        hostname: data.tokenProperties.hostname,
        createTime: data.tokenProperties.createTime,
        analysis: analysis
      };
    } else {
      console.error('reCAPTCHA Enterprise verification failed:', data);
      return {
        success: false,
        error: data.error?.message || 'Verification failed',
        score: 0.3, // Low score for failed verification
        analysis: {
          score: 0.3,
          confidence: 'low',
          recommendation: 'challenge',
          riskLevel: 'high',
          action: 'require_additional_verification'
        }
      };
    }
  } catch (error) {
    console.error('reCAPTCHA Enterprise verification error:', error);
    return {
      success: false,
      error: error.message,
      score: 0.5, // Neutral score on error
      analysis: {
        score: 0.5,
        confidence: 'unknown',
        recommendation: 'proceed',
        riskLevel: 'medium',
        action: 'none'
      }
    };
  }
}

/**
 * Analyze reCAPTCHA score and determine action
 * @param {number} score - reCAPTCHA score (0.0 - 1.0)
 * @returns {Object} - Analysis result with recommendations
 */
function analyzeRecaptchaScore(score) {
  const analysis = {
    score,
    confidence: 'unknown',
    recommendation: 'proceed',
    riskLevel: 'low',
    action: 'none'
  };

  if (score >= RECAPTCHA_CONFIG.scoreThresholds.high) {
    analysis.confidence = 'high';
    analysis.recommendation = 'allow';
    analysis.riskLevel = 'very_low';
    analysis.action = 'proceed_normal';
  } else if (score >= RECAPTCHA_CONFIG.scoreThresholds.medium) {
    analysis.confidence = 'medium';
    analysis.recommendation = 'monitor';
    analysis.riskLevel = 'low';
    analysis.action = 'proceed_with_monitoring';
  } else if (score >= RECAPTCHA_CONFIG.scoreThresholds.low) {
    analysis.confidence = 'low';
    analysis.recommendation = 'challenge';
    analysis.riskLevel = 'medium';
    analysis.action = 'require_additional_verification';
  } else {
    analysis.confidence = 'very_low';
    analysis.recommendation = 'block';
    analysis.riskLevel = 'high';
    analysis.action = 'block_or_captcha';
  }

  return analysis;
}

/**
 * Combine reCAPTCHA score with ML detection results
 * @param {Object} mlResults - ML bot detection results
 * @param {Object} recaptchaAnalysis - reCAPTCHA score analysis
 * @returns {Object} - Combined analysis
 */
function combineRecaptchaWithML(mlResults, recaptchaAnalysis) {
  const combined = {
    mlScore: mlResults?.final_score || mlResults?.loginAttempts?.[0]?.fusionScore || 0.5,
    recaptchaScore: recaptchaAnalysis.score,
    combinedRisk: 'low',
    finalDecision: 'allow',
    confidence: 'medium',
    triggers: []
  };

  // Determine combined risk level
  const mlRisk = combined.mlScore > 0.5 ? 'high' : 'low';
  const recaptchaRisk = recaptchaAnalysis.riskLevel;

  // Combined risk assessment
  if (mlRisk === 'high' && recaptchaRisk === 'high') {
    combined.combinedRisk = 'very_high';
    combined.finalDecision = 'block';
    combined.confidence = 'high';
    combined.triggers.push('ml_bot_detected', 'recaptcha_bot_detected');
  } else if (mlRisk === 'high' || recaptchaRisk === 'high') {
    combined.combinedRisk = 'high';
    combined.finalDecision = 'challenge';
    combined.confidence = 'medium';
    combined.triggers.push(mlRisk === 'high' ? 'ml_bot_detected' : 'recaptcha_bot_detected');
  } else if (mlRisk === 'low' && recaptchaRisk === 'very_low') {
    combined.combinedRisk = 'very_low';
    combined.finalDecision = 'allow';
    combined.confidence = 'high';
  } else {
    combined.combinedRisk = 'medium';
    combined.finalDecision = 'monitor';
    combined.confidence = 'medium';
  }

  return combined;
}

async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { sessionId, recaptchaToken } = req.body;
    
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
    if (CACHE_TTL > 0 && cachedResult && (Date.now() - cachedResult.timestamp) < CACHE_TTL) {
      console.log('Returning cached bot detection result');
      return res.status(200).json({
        success: true,
        results: cachedResult.data,
        cached: true
      });
    }

    console.log('Starting optimized bot detection analysis...');
    
    // Verify reCAPTCHA token if provided
    let recaptchaAnalysis = null;
    if (recaptchaToken) {
      console.log('Verifying reCAPTCHA Enterprise token...');
      const recaptchaResult = await verifyRecaptcha(recaptchaToken, 'LOGIN');
      recaptchaAnalysis = recaptchaResult.analysis;
      console.log('reCAPTCHA Enterprise verification result:', {
        success: recaptchaResult.success,
        score: recaptchaResult.score,
        confidence: recaptchaAnalysis.confidence,
        riskLevel: recaptchaAnalysis.riskLevel
      });
    }
    
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
        
        // Combine reCAPTCHA analysis with ML results
        let combinedAnalysis = null;
        if (recaptchaAnalysis) {
          combinedAnalysis = combineRecaptchaWithML(results, recaptchaAnalysis);
          console.log('Combined analysis result:', {
            mlScore: combinedAnalysis.mlScore,
            recaptchaScore: combinedAnalysis.recaptchaScore,
            combinedRisk: combinedAnalysis.combinedRisk,
            finalDecision: combinedAnalysis.finalDecision,
            confidence: combinedAnalysis.confidence,
            triggers: combinedAnalysis.triggers
          });
        }
        
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
          recaptchaAnalysis: recaptchaAnalysis,
          combinedAnalysis: combinedAnalysis,
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