# GitHub Pages tutorial

Choose **Settings → Pages → Deploy from a branch → main → /docs** after uploading the repository.

- `index.html`: the full tutorial, all ten approved images, per-step tool explanations and HF demo links.
- `medium.html`: a plain HTML article with the same content and all ten images; intended for Medium's URL importer.
- `tutorial.md`: local Markdown version for GitHub.
- `tutorial.txt`: plain-text article source linked by the Pages site.
- `images/`: the original ten generated PNGs, unmodified.
- `figure-manifest.json`: original image dimensions and SHA-256 hashes.

From the repository root:

```bash
python scripts/build_docs.py
python scripts/validate_docs.py
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. No Node, Python server, model weights or API credentials are needed by the published Pages site. The local Python server above is only a preview convenience.

Set the actual deployment URL with `--public-base-url` when building if it differs from `site-config.json`. The main tutorial works with relative images; Medium's export needs a reachable public image base. See `../publishing/MEDIUM_GUIDE.md` for import instructions.
