"""Run with python app.py, then visit http://127.0.0.1:8000."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from classifier import SpamClassifier

ROOT = Path(__file__).resolve().parent


def load_model():
    path = ROOT / 'artifacts/model.json'
    if not path.exists():
        raise SystemExit('Model missing. Run: python train.py')
    data = json.loads(path.read_text(encoding='utf-8'))
    model = SpamClassifier()
    for key in ('idf', 'priors', 'weights'):
        setattr(model, key, data[key])
    return model, data['metadata']


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, kind='application/json'):
        body = content.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (ROOT / 'index.html').read_text(encoding='utf-8'), 'text/html')
        elif self.path == '/api/metrics':
            self.send(200, json.dumps(self.server.metadata))
        else:
            self.send(404, json.dumps({'error': 'Not found'}))

    def do_POST(self):
        if self.path != '/api/predict':
            self.send(404, json.dumps({'error': 'Not found'}))
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 100000:
                raise ValueError('Email request must be between 1 and 100,000 bytes.')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict) or not isinstance(data.get('text'), str):
                raise ValueError('Provide a text string.')
            self.send(200, json.dumps(self.server.model.predict(data['text'])))
        except (ValueError, UnicodeError) as error:
            self.send(400, json.dumps({'error': str(error)}))

    def log_message(self, *args):
        pass  # Do not log email contents or requests.


if __name__ == '__main__':
    model, metadata = load_model()
    server = ThreadingHTTPServer(('127.0.0.1', 8000), Handler)
    server.model, server.metadata = model, metadata
    print('Spam Email Classifier: http://127.0.0.1:8000 (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
