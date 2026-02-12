from flask import Flask, request, jsonify, render_template_string
from embeddings import load_chunks, search

app = Flask(__name__)

# Load chunks with embeddings
chunks = load_chunks('chunks_with_embeddings.json')

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Second Brain</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        input { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 8px; margin-bottom: 20px; }
        input:focus { outline: none; border-color: #007aff; }
        button { padding: 15px 30px; font-size: 16px; background: #007aff; color: white; border: none; border-radius: 8px; cursor: pointer; }
        button:hover { background: #005ecb; }
        .result { background: #f9f9f9; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #007aff; }
        .path { color: #666; font-size: 14px; margin-bottom: 10px; }
        .score { color: #999; font-size: 12px; }
        .text { white-space: pre-wrap; line-height: 1.6; }
    </style>
</head>
<body>
    <h1>🧠 Second Brain</h1>
    <form onsubmit="search(event)">
        <input type="text" id="query" placeholder="What do you want to know?" autofocus>
        <button type="submit">Search</button>
    </form>
    <div id="results"></div>

    <script>
        async function search(e) {
            e.preventDefault();
            const query = document.getElementById('query').value;
            const res = await fetch('/search?q=' + encodeURIComponent(query));
            const data = await res.json();

            let html = '';
            data.forEach((r, i) => {
                html += `<div class="result">
                    <div class="path">${r.path}</div>
                    <div class="score">Score: ${r.score.toFixed(3)}</div>
                    <div class="text">${r.text}</div>
                </div>`;
            });
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/search')
def search_route():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    results = search(query, chunks)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
