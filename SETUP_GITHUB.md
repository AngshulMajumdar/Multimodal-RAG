# Put this repository on GitHub

Configured repository name: `Free-Mulit-modal-Rag`.

## Upload in the browser

1. Create an empty repository in your GitHub account.
2. Extract this archive using Extract All. Open the extracted `Multimodal-RAG` folder. On GitHub choose Add file → Upload files. Drag all files AND folders from that folder into the upload area together. Preserve the folder hierarchy. Do not upload the ZIP or only select the loose files. The root must contain `app.py`, `mrag/`, `README.md`, `requirements.txt`, `docs/`, `content/` and `publishing/`.
3. Commit to `main`. Confirm that `docs/index.html`, the ten files in `docs/images/`, and `mrag/` appear on GitHub before enabling Pages. An absent `docs` folder causes the Pages build to fail with “No such file or directory”.
4. Enable **Settings → Pages → Deploy from a branch → main → /docs**.
5. Wait for GitHub's deployment and open the URL it reports.

The live Python demo remains on Hugging Face: https://huggingface.co/spaces/Angshul/Multimodal-RAG. GitHub Pages hosts the illustrated tutorial, not the Python app.

## Push with Git

From the extracted repository folder:

```bash
git init -b main
git add .
git commit -m "Add multimodal RAG application and illustrated tutorial"
git remote add origin https://github.com/YOUR_ACCOUNT/Free-Mulit-modal-Rag.git
git push -u origin main
```

If the remote already exists locally, inspect it with `git remote -v` rather than adding it twice. Authenticate through your normal GitHub credential manager; do not put tokens in source files.

With an authenticated GitHub CLI, an empty remote can instead be created and pushed with:

```bash
gh repo create YOUR_ACCOUNT/Free-Mulit-modal-Rag --public --source=. --remote=origin --push
```

The build package does not claim that a remote repository or Pages deployment has already been created.

## Check the publishing URL

The default target in `site-config.json` is:

`https://angshulmajumdar.github.io/Free-Mulit-modal-Rag/`

For a different account, repository or domain, set the actual URL before publishing:

```bash
python scripts/build_docs.py --public-base-url https://YOUR_ACCOUNT.github.io/YOUR_REPOSITORY/
python scripts/validate_docs.py
```

Commit the rebuilt `docs/`, `publishing/` and configuration changes. The main tutorial uses relative image paths; the Medium export needs the configured public URL to load its images.

## Publish the same tutorial on Medium

After Pages is live, open its `medium.html` URL. In Medium choose **Stories → Import a story** and paste that URL. Review all ten figures and captions before publishing. Full instructions and manual-copy alternatives are in `publishing/MEDIUM_GUIDE.md`.

## Reproduce local checks

```bash
python -m pip install pillow
python scripts/build_docs.py
python scripts/validate_docs.py
python -m http.server 8000 --directory docs
```

For the application and model integration tests, follow the technical README. The recorded pre-deployment tests remain in `verification/`; this packaging pass adds documentation and lightweight CPU checks without changing the working runtime source.
