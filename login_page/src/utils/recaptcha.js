/**
 * reCAPTCHA v3 Utility Functions
 * 
 * This module provides utility functions for integrating Google reCAPTCHA v3
 * with the bot detection system.
 */

// reCAPTCHA configuration
const RECAPTCHA_CONFIG = {
  siteKey: import.meta.env.VITE_RECAPTCHA_SITE_KEY || '6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO',
  actions: {
    login: 'LOGIN',
    pageview: 'PAGEVIEW',
    submit: 'SUBMIT',
    mouse_move: 'MOUSE_MOVE',
    form_interaction: 'FORM_INTERACTION'
  },
  scoreThresholds: {
    high: 0.7,      // High confidence human
    medium: 0.5,    // Medium confidence
    low: 0.3,       // Low confidence - likely bot
    critical: 0.1   // Very likely bot
  }
};

/**
 * Execute reCAPTCHA Enterprise for a specific action
 * @param {string} action - The action name (LOGIN, SUBMIT, etc.)
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptcha = async (action = 'PAGEVIEW') => {
  return new Promise((resolve, reject) => {
    if (typeof window.grecaptcha === 'undefined' || !window.grecaptcha.enterprise) {
      reject(new Error('reCAPTCHA Enterprise not loaded'));
      return;
    }

    window.grecaptcha.enterprise.ready(async () => {
      try {
        const token = await window.grecaptcha.enterprise.execute(RECAPTCHA_CONFIG.siteKey, { action });
        console.log(`reCAPTCHA Enterprise executed for action: ${action}`);
        resolve(token);
      } catch (error) {
        console.error('reCAPTCHA Enterprise execution failed:', error);
        reject(error);
      }
    });
  });
};

/**
 * Execute reCAPTCHA on page load for analytics
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptchaPageView = () => {
  return executeRecaptcha('PAGEVIEW');
};

/**
 * Execute reCAPTCHA on form submission
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptchaSubmit = () => {
  return executeRecaptcha('SUBMIT');
};

/**
 * Execute reCAPTCHA on login action
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptchaLogin = () => {
  return executeRecaptcha('LOGIN');
};

/**
 * Execute reCAPTCHA on mouse movement (for suspicious activity)
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptchaMouseMove = () => {
  return executeRecaptcha('MOUSE_MOVE');
};

/**
 * Execute reCAPTCHA on form interaction
 * @returns {Promise<string>} - reCAPTCHA token
 */
export const executeRecaptchaFormInteraction = () => {
  return executeRecaptcha('FORM_INTERACTION');
};

/**
 * Analyze reCAPTCHA score and determine action
 * @param {number} score - reCAPTCHA score (0.0 - 1.0)
 * @returns {Object} - Analysis result with recommendations
 */
export const analyzeRecaptchaScore = (score) => {
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
};

/**
 * Combine reCAPTCHA score with ML detection results
 * @param {Object} mlResults - ML bot detection results
 * @param {Object} recaptchaAnalysis - reCAPTCHA score analysis
 * @returns {Object} - Combined analysis
 */
export const combineRecaptchaWithML = (mlResults, recaptchaAnalysis) => {
  const combined = {
    mlScore: mlResults?.final_score || 0.5,
    recaptchaScore: recaptchaAnalysis.score,
    combinedRisk: 'low',
    finalDecision: 'allow',
    confidence: 'medium',
    triggers: []
  };

  // Determine combined risk level
  const mlRisk = mlResults?.final_score > 0.5 ? 'high' : 'low';
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
};

/**
 * Get reCAPTCHA configuration
 * @returns {Object} - reCAPTCHA configuration
 */
export const getRecaptchaConfig = () => {
  return { ...RECAPTCHA_CONFIG };
};

/**
 * Check if reCAPTCHA Enterprise is loaded and ready
 * @returns {boolean} - True if reCAPTCHA Enterprise is ready
 */
export const isRecaptchaReady = () => {
  return typeof window !== 'undefined' && 
         typeof window.grecaptcha !== 'undefined' && 
         typeof window.grecaptcha.enterprise !== 'undefined' &&
         typeof window.grecaptcha.enterprise.ready === 'function';
};

export default {
  executeRecaptcha,
  executeRecaptchaPageView,
  executeRecaptchaSubmit,
  executeRecaptchaLogin,
  executeRecaptchaMouseMove,
  executeRecaptchaFormInteraction,
  analyzeRecaptchaScore,
  combineRecaptchaWithML,
  getRecaptchaConfig,
  isRecaptchaReady
};
