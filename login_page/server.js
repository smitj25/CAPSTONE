import express from 'express';
import bodyParser from 'body-parser';
import path from 'path';
import { fileURLToPath } from 'url';
import logHandler from './src/api/log.js';
import botDetectionHandler from './src/api/botDetection.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(bodyParser.json());

// API routes
app.post('/api/log', (req, res) => logHandler(req, res));
app.post('/api/bot-detection', (req, res) => botDetectionHandler(req, res));

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'dist')));
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});