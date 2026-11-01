# Hugging Face Gradio / ZeroGPU deployment review

Review date: 1 November 2026.

## Verdict

The corrected package passes local integration tests using the default production retrieval stack. It is prepared for a Gradio Space with **ZeroGPU** hardware. It has **not** been deployed to a hosted Space in this review, and full Qwen3-VL-4B GPU inference, startup packing, quota handling and wall-clock GPU duration still require the hosted smoke test in README.md.

## Corrections in this review

1. **Failed retrieval could launch generation.** Gradio `.then()` runs even after a preceding callback fails. GPU generation is now directly registered using `.success(fn=vlm_answer, ...)`. A blank question or missing index cannot consume GPU time through the browser's normal event chain or reuse a previous pending prompt.
2. **Concurrent operations could invalidate evidence.** CPU index/retrieval/reset callbacks now share a concurrency-one group. GPU generation also has concurrency one. A per-session busy flag and disabled controls prevent re-indexing, clearing evidence or submitting another question during generation. A final web-process callback clears pending data and releases the session on success or failure, including a scheduler/quota error. API clients calling individual endpoints must also call `/finish_answer`; the browser invokes this chained callback automatically.
3. **The stated file cap was only per batch.** The limit is now cumulative across successive uploads and demo loading. The Gradio upload endpoint also enforces the configured per-file byte limit before indexing. Invalid initial requests no longer allocate orphan session directories.
4. **CPU callbacks could be redirected to CUDA by configuration.** Space retrieval is explicitly pinned to CPU, outside the GPU allocation. The VLM retains the correct module-level `.to("cuda")` startup pattern and a directly registered GPU callback. No dedicated GPU, external inference service or external Qdrant server is required.
5. **Overlapping chunks could exceed their configured size.** A nearly full paragraph followed by a full-size paragraph previously produced an oversized chunk. Overlap now keeps only the words that fit; invalid chunk/overlap settings raise an error. A regression covers this case.
6. **Shared OCR engine/cache access was uncontrolled.** OCR/cache access is locked and the shared text cache is bounded to 512 entries. CPU operations are serialized as an additional safeguard.
7. **Dependency/startup configuration was fragile.** Critical OpenCLIP/Qdrant/OCR/OpenCV/ONNX versions are pinned to the locally tested versions. `packages.txt` includes OpenCV system libraries. Python metadata now selects 3.12, and a one-hour startup timeout allows the initial model downloads. PyTorch 2.8.0 remains within the documented ZeroGPU-supported versions.
8. **Temporary data and fallback behavior.** Gradio cache cleanup now runs hourly for files older than three hours, alongside the session-state TTL. Pending prompt/image states are cleared after an answer and on reset. The optional CPU fallback respects the image toggle and requested output-token limit. Only transformers/extractive backends are accepted by this Space entry point; local CLI server backends remain available separately.

Earlier corrections are retained: vector-drawn PDF pages are rendered for visual retrieval, relevant native page text links to those renders, repeated figures preserve their page citations, and document content is treated as untrusted evidence in the system prompt.

## Measured validation

| Check | Measured result |
|---|---|
| Python compilation | Passed |
| Dependency resolution against requirements.txt | Passed, no additional changes required |
| `pip check` | No broken requirements found |
| Full test suite with real default BGE retrieval | **52 passed, 1 skipped**, 75.38 s |
| Additional registered-event configuration regression | **1 passed**, 7.13 s |
| Actual Qwen3-VL-4B processor/tokenizer/config | Passed with text-only evidence and two images; image tokens matched the vision grids. No 4B weights were loaded. |
| Separately run default BGE/BM25/CLIP/OCR pipeline suite | **20 passed**, 70.88 s; overlaps the full suite |
| Earlier WordLlama-based full-suite attempt | 48 non-Gradio tests passed; Gradio setup hit this workspace's missing SOCKS support. Local proxy support was installed, then the complete default-BGE run above passed. |

The skipped test is the optional Streamlit UI, which is outside this Gradio deployment. Seventeen warnings in the full run are upstream Starlette deprecations; they did not fail requests. A preliminary OCR run with an unconstrained newer OpenCV wheel crashed during import in this environment; the pinned OpenCV/ONNX pair above passed the OCR suite.

The integration tests exercise real BGE and OpenCLIP checkpoint loading, RapidOCR on a chart/scanned page/screenshot, Qdrant/BM25 retrieval, image retrieval and negative image queries, PDF/DOCX ingestion, citation provenance, re-ingesting, CLI behavior, local Ollama/OpenAI-protocol servers, Gradio uploads, separate sessions, streamed answers, busy guards, cleanup and cumulative upload limits.

**Generation test scope:** VLM tests use a tiny randomly initialized local Qwen3-VL checkpoint. They test the real transformers architecture, image-token/grid alignment, generation thread, streaming and error propagation. They do **not** measure answer quality, memory use or speed of the actual 4B checkpoint. Hosted ZeroGPU is not emulated by these CPU tests.

The GPU-duration formula is an estimate, not a benchmark. The default request is about 74 seconds for three images and 512 output tokens. Measure representative hosted calls before tuning it. Start with 256 output tokens if quota/duration issues arise.

## Reproduce

Install `requirements-dev.txt`, then run:

```bash
MRAG_TEXT_BACKEND=sentence-transformers OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest -q
```

The recorded full run also set `HF_HUB_DISABLE_XET=1`, disabled Gradio analytics and exempted localhost from this workspace's proxy. Those are local test-environment settings, not mandatory Space variables. Install SOCKS client extras only if your own proxy environment requires them.

The current suite includes the additional event-configuration regression, so a fresh complete run has one more test than the recorded 52-pass full run. Supplying `MRAG_VLM_ID` runs the VLM/app tests against that checkpoint instead of the tiny random fixture; a real 4B model is impractically slow on CPU.

Raw successful test output and measured package versions are in `verification/`.

## Hosted acceptance checks

Follow README.md: choose SDK Gradio and hardware ZeroGPU; upload the extracted files at repository root; leave `MRAG_LLM_BACKEND=transformers`; wait for Running; load the synthetic demo; try all five example questions; verify citations/images and a second session; clear the index; try your own PDF, Word file and image. Confirm in the Space logs that the model uses CUDA during generation and that the GPU callback completes within its reserved duration.

Session isolation is for retrieval/index state. Public Gradio file URLs can outlive a session until cache cleanup, so this remains a public demonstration rather than a confidential-document service.

## Authoritative references checked

- ZeroGPU model placement, supported runtimes and hardware selection: https://huggingface.co/docs/hub/en/spaces-zerogpu
- Space frontmatter and startup timeout: https://huggingface.co/docs/hub/en/spaces-config-reference
- Qwen3-VL-4B processor/model usage: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Gradio event success/failure semantics: https://github.com/gradio-app/gradio/blob/main/gradio/events.py
- Gradio output-file/cache behavior: https://gradio.app/guides/file-access
