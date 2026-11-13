window.BENCHMARK_DATA = {
  "lastUpdate": 1794596442288,
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
      },
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "153733920c7f327153c4eed451bd6b4ac02c00aa",
          "message": "Merge pull request #9 from e11bio/ship-py-typed\n\nShip the PEP 561 py.typed marker",
          "timestamp": "2026-11-13T18:58:15Z",
          "url": "https://github.com/e11bio/volara-ml/commit/153733920c7f327153c4eed451bd6b4ac02c00aa"
        },
        "date": 1794596441828,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara_ml.models, volara_ml.blockwise (cold)",
            "value": 3850.088,
            "range": "± 789.283",
            "unit": "ms",
            "extra": "interpreter startup 9.8 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 15.186,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}