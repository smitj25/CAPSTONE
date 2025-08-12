const fs = require('fs');
const path = require('path');

class EventLogger {
  constructor() {
    this.logDirectory = path.join(process.cwd(), 'logs');
    this.ensureLogDirectory();
  }

  ensureLogDirectory() {
    if (!fs.existsSync(this.logDirectory)) {
      fs.mkdirSync(this.logDirectory, { recursive: true });
    }
  }

  generateFilename(prefix) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    return path.join(this.logDirectory, `${prefix}_${timestamp}.json`);
  }

  async saveEvents(events, type) {
    const filename = this.generateFilename(type);
    try {
      await fs.promises.writeFile(filename, JSON.stringify(events, null, 2));
      console.log(`Events saved to ${filename}`);
      return filename;
    } catch (error) {
      console.error('Error saving events:', error);
      throw error;
    }
  }
}

module.exports = new EventLogger();