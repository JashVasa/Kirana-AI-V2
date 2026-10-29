const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const parseHandler = require('./api/parse');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

const server = http.createServer(async (req, res) => {
  // CORS & helper extensions
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    return res.end();
  }

  res.status = function(code) {
    this.statusCode = code;
    return this;
  };
  res.json = function(data) {
    this.setHeader('Content-Type', 'application/json; charset=utf-8');
    this.end(JSON.stringify(data));
    return this;
  };

  const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = parsedUrl.pathname;

  // Handle /api/parse
  if (pathname === '/api/parse') {
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'POST only' });
    }
    let bodyRaw = '';
    req.on('data', chunk => {
      bodyRaw += chunk;
      if (bodyRaw.length > 1e6) {
        req.destroy();
      }
    });
    req.on('end', async () => {
      try {
        req.body = bodyRaw ? JSON.parse(bodyRaw) : {};
      } catch (err) {
        return res.status(400).json({ error: 'Invalid JSON body' });
      }
      try {
        await parseHandler(req, res);
      } catch (err) {
        console.error('API Error:', err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Internal server error' });
        }
      }
    });
    return;
  }

  // Handle static files
  let filePath = path.join(__dirname, pathname === '/' ? 'index.html' : pathname);
  try {
    const stats = await fs.promises.stat(filePath);
    if (stats.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    const content = await fs.promises.readFile(filePath);
    res.setHeader('Content-Type', contentType);
    res.statusCode = 200;
    res.end(content);
  } catch (err) {
    res.statusCode = 404;
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.end('404 Not Found');
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Kirana Order Assistant is running at:`);
  console.log(`  > Local:   http://localhost:${PORT}`);
  console.log(`  > Network: http://127.0.0.1:${PORT}`);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.log(`Note: ANTHROPIC_API_KEY is not set. The app will seamlessly use its built-in rule-based NLP parser.`);
  } else {
    console.log(`Anthropic API key detected; LLM parser is active.`);
  }
});
