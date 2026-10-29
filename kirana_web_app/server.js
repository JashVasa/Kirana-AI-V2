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
  '.ico': 'image/x-icon',
};

// Static literal references so @vercel/nft AST parser traces and bundles them:
let preloadedIndexHtml = null;
try {
  preloadedIndexHtml = fs.readFileSync(path.join(__dirname, 'index.html'));
} catch (e) {}

let preloadedCatalogCsv = null;
try {
  preloadedCatalogCsv = fs.readFileSync(path.join(__dirname, 'catalog.csv'));
} catch (e) {}

function loadAsset(filename) {
  if (filename === 'index.html' && preloadedIndexHtml) {
    return preloadedIndexHtml;
  }
  if (filename === 'catalog.csv' && preloadedCatalogCsv) {
    return preloadedCatalogCsv;
  }

  const candidateDirs = [
    __dirname,
    path.join(__dirname, '..'),
    process.cwd(),
    path.join(process.cwd(), 'kirana_web_app')
  ];

  for (const dir of candidateDirs) {
    try {
      const fullPath = path.join(dir, filename);
      if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
        return fs.readFileSync(fullPath);
      }
    } catch (e) {}
  }
  return null;
}

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
  let pathname = parsedUrl.pathname;

  // Handle /api/parse
  if (pathname === '/api/parse' || pathname.endsWith('/api/parse')) {
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

  // Handle root or index.html
  if (pathname === '/' || pathname === '/index.html' || pathname.endsWith('/index.html')) {
    const content = loadAsset('index.html');
    if (content) {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      res.statusCode = 200;
      return res.end(content);
    }
  }

  // Handle catalog.csv
  if (pathname === '/catalog.csv' || pathname.endsWith('/catalog.csv')) {
    const content = loadAsset('catalog.csv');
    if (content) {
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.statusCode = 200;
      return res.end(content);
    }
  }

  // Handle other static files
  const cleanPath = pathname.replace(/^\/+/, '');
  const content = loadAsset(cleanPath);
  if (content) {
    const ext = path.extname(cleanPath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    res.setHeader('Content-Type', contentType);
    res.statusCode = 200;
    return res.end(content);
  }

  res.statusCode = 404;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end('404 Not Found');
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
