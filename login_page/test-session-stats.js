import fetch from 'node-fetch';

// Session statistics data
const sessionStats = { 
  session_id: "abc123", 
  total_requests: 37, 
  total_bytes: 19452, 
  http_get: 30, 
  http_post: 5, 
  http_head: 2, 
  perc_3xx: 8.1, 
  perc_4xx: 5.4, 
  perc_img: 17.6, 
  perc_css: 4.3, 
  perc_js: 3.2, 
  html_to_img_ratio: 3.5, 
  depth_sd: 1.14, 
  max_reqs_per_page: 12, 
  avg_reqs_per_page: 2.4, 
  max_consec_seq_http: 7, 
  perc_consec_seq_http: 21.6, 
  session_time: 18.2, 
  browsing_speed: 2.16, 
  inter_req_sd: 0.91 
};

// Send the session statistics to the server
async function sendSessionStats() {
  try {
    const response = await fetch('http://localhost:3001/api/log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sessionStats)
    });
    
    const data = await response.json();
    console.log('Response:', data);
  } catch (error) {
    console.error('Error sending session statistics:', error);
  }
}

// Execute the function
sendSessionStats();