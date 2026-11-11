window.BENCHMARK_DATA = {
  "lastUpdate": 1794438121885,
  "repoUrl": "https://github.com/e11bio/volara-ml",
  "entries": {
    "volara-ml": [
      {
        "commit": {
          "author": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "id": "82aa2b7292830227cdc9f91c8d8b7e9e46bbc657",
          "message": "Bump version to 1.3.0\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T22:57:48Z",
          "url": "https://github.com/e11bio/volara-ml/commit/82aa2b7292830227cdc9f91c8d8b7e9e46bbc657"
        },
        "date": 1794438121258,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara_ml.models, volara_ml.blockwise (cold)",
            "value": 5048.484,
            "range": "± 1051.333",
            "unit": "ms",
            "extra": "interpreter startup 13.4 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 19.599,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}