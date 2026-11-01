# Publish the illustrated tutorial on Medium

The repository contains a plain article edition, not an automated Medium publisher. It includes all ten original images, tool explanations, figure captions, code snippets and the HF demo link.

## Files

- `docs/index.html`: the styled GitHub Pages tutorial.
- `docs/medium.html`: the import-friendly article served by Pages.
- `publishing/medium.html`: the same plain HTML for reference.
- `publishing/medium.md`: Markdown with public image URLs.
- `docs/tutorial.md`: Markdown using local relative image paths for GitHub.
- `docs/images/`: all ten original PNGs for manual image upload if needed.

## Set the public URL

The configured default is the intended address:

`https://angshulmajumdar.github.io/Free-Mulit-modal-Rag/`

This address becomes usable after the repository and Pages site are published under those names. For a different account, repository name or domain, run from the repository root:

```bash
python scripts/build_docs.py --public-base-url https://ACCOUNT.github.io/REPOSITORY/
python scripts/validate_docs.py
```

Commit the rebuilt files and `site-config.json`. The main tutorial uses relative image paths and works locally. The Medium edition deliberately uses absolute public image URLs, so those images require the public Pages deployment to exist.

## Import the article

1. Enable GitHub Pages from `main` → `/docs` and wait for its deployment to finish.
2. Open `https://ACCOUNT.github.io/REPOSITORY/medium.html`. Check all ten images.
3. In Medium on desktop, open **Stories → Import a story**.
4. Paste that public `medium.html` URL and import it.
5. Review title, subtitle, author, all ten images, captions, code formatting and the HF demo link.
6. Check the canonical link in Medium's story settings. Medium's import workflow normally sets a canonical reference to the imported source; the export also supplies the main tutorial URL as its canonical HTML hint.
7. Publish only when the draft is correct.

The plain edition avoids tables, JavaScript, interactive widgets, iframes, external styles and hidden content. This improves portability; it does not guarantee that Medium's importer will preserve every formatting detail. The image content and its source files remain available if an import needs manual repair.

## Manual fallback

Copy the article's visible text into a Medium draft and upload the ten files from `docs/images/` in numerical order. Paste the captions from `publishing/medium.md` below the corresponding images. Copy code snippets into Medium code blocks. Set an appropriate canonical URL in Medium if republishing your own Pages article.

Do not paste Markdown syntax expecting Medium to turn the entire file into a formatted story automatically. The HTML import route or manual rich-text editing is the supported workflow here.

## Official references

- https://help.medium.com/hc/en-us/articles/214550207-Importing-a-post-to-Medium
- https://help.medium.com/hc/en-us/articles/360033931713-Trouble-importing-content-using-the-import-tool
