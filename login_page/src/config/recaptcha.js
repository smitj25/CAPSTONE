/**
 * reCAPTCHA Enterprise Configuration
 * 
 * To use reCAPTCHA Enterprise:
 * 1. Visit https://www.google.com/recaptcha/admin
 * 2. Create a new site with reCAPTCHA Enterprise
 * 3. Add your domain (localhost for development)
 * 4. Copy the Site Key and Secret Key
 * 5. Update the keys below or set environment variables
 * 
 * Enterprise Benefits:
 * - Advanced bot detection algorithms
 * - Enhanced analytics and reporting
 * - Custom risk assessment
 * - Better integration with Google Cloud Security
 */

export const RECAPTCHA_CONFIG = {
  // Get these from Google reCAPTCHA Enterprise Admin Console
  siteKey: import.meta.env.VITE_RECAPTCHA_SITE_KEY || '6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO',
  apiKey: import.meta.env.VITE_RECAPTCHA_API_KEY || '6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO',
  
  // reCAPTCHA Enterprise API endpoint
  verifyUrl: 'https://recaptchaenterprise.googleapis.com/v1/projects/endless-gamma-457506-a0/assessments',
  
  // Score thresholds for different confidence levels
  scoreThresholds: {
    high: parseFloat(import.meta.env.VITE_RECAPTCHA_HIGH_THRESHOLD) || 0.7,      // High confidence human
    medium: parseFloat(import.meta.env.VITE_RECAPTCHA_MEDIUM_THRESHOLD) || 0.5,    // Medium confidence
    low: parseFloat(import.meta.env.VITE_RECAPTCHA_LOW_THRESHOLD) || 0.3,       // Low confidence - likely bot
    critical: parseFloat(import.meta.env.VITE_RECAPTCHA_CRITICAL_THRESHOLD) || 0.1   // Very likely bot
  },
  
  // Action names for different user interactions (Enterprise standard - uppercase)
  actions: {
    pageview: 'PAGEVIEW',
    login: 'LOGIN',
    submit: 'SUBMIT',
    mouse_move: 'MOUSE_MOVE',
    form_interaction: 'FORM_INTERACTION',
    captcha_challenge: 'CAPTCHA_CHALLENGE'
  }
};

export default RECAPTCHA_CONFIG;
