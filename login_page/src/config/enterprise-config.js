/**
 * reCAPTCHA Enterprise API Configuration
 * 
 * This file contains the configuration for reCAPTCHA Enterprise API calls.
 * Replace YOUR_ENTERPRISE_API_KEY with your actual API key from Google Cloud Console.
 */

export const ENTERPRISE_CONFIG = {
  // Google Cloud Project ID
  projectId: 'endless-gamma-457506-a0',
  
  // reCAPTCHA Enterprise Site Key
  siteKey: '6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO',
  
  // API Key for reCAPTCHA Enterprise (get from Google Cloud Console)
  apiKey: process.env.REACT_APP_RECAPTCHA_API_KEY || '6LekL9ArAAAAAFGpIoMxyUuz5GkXnhT-DQocifhO',
  
  // API Endpoint
  baseUrl: 'https://recaptchaenterprise.googleapis.com/v1',
  
  // Full assessment URL
  getAssessmentUrl: () => {
    return `${ENTERPRISE_CONFIG.baseUrl}/projects/${ENTERPRISE_CONFIG.projectId}/assessments?key=${ENTERPRISE_CONFIG.apiKey}`;
  },
  
  // Request body template
  createRequestBody: (token, expectedAction = 'PAGEVIEW') => {
    return {
      event: {
        token: token,
        expectedAction: expectedAction,
        siteKey: ENTERPRISE_CONFIG.siteKey
      }
    };
  }
};

export default ENTERPRISE_CONFIG;
