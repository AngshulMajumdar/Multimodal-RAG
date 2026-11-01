"""Build GitHub Pages and a Medium-importable article from one tutorial source.

python scripts/build_docs.py
python scripts/build_docs.py --public-base-url https://ACCOUNT.github.io/REPOSITORY/
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'


def esc(value):
    return html.escape(str(value),quote=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--public-base-url',help='Published Pages URL; updates site-config.json for Medium image URLs.')
    args=ap.parse_args()
    config=json.loads((ROOT/'site-config.json').read_text())
    if args.public_base_url:
        url=args.public_base_url.rstrip('/')+'/'
        parsed=urlsplit(url)
        if parsed.scheme not in {'https','http'} or not parsed.netloc or parsed.query or parsed.fragment:
            ap.error('--public-base-url must be a full HTTP(S) URL without query/fragment')
        config['public_base_url']=url
        (ROOT/'site-config.json').write_text(json.dumps(config,indent=2)+'\n')
    base=config['public_base_url'].rstrip('/')+'/'
    source=json.loads((ROOT/'content/tutorial.json').read_text())
    figures={x['number']:x for x in json.loads((DOCS/'figure-manifest.json').read_text())}
    demo=esc(config['demo_url'])
    author=f'<p class="byline"><strong>{esc(config["author"])}</strong><br>{esc(config["affiliation"])}</p>'
    intro='\n'.join(f'<p>{esc(p)}</p>' for p in source['intro'])
    closing='\n'.join(f'<p>{esc(p)}</p>' for p in source['closing'])

    def section(s,absolute=False):
        im=figures[s['figure']]
        prefix=base+'images/' if absolute else 'images/'
        image=prefix+im['file']
        paras='\n'.join(f'<p>{esc(p)}</p>' for p in s['paragraphs'])
        tools='\n'.join(f'<li><strong>{esc(name)}.</strong> {esc(why)}</li>' for name,why in s['tools'])
        code=f'<pre><code>{esc(s["code"])}</code></pre>' if s.get('code') else ''
        return f'''<section id="{esc(s['id'])}"><h2>{esc(s['title'])}</h2>
<figure><a href="{esc(image)}" target="_blank" rel="noopener noreferrer" aria-label="Open figure {s['figure']} at full resolution"><img src="{esc(image)}" alt="{esc(s['caption'])}" width="{im['width']}" height="{im['height']}" decoding="async" style="max-width:100%;height:auto"></a><figcaption>{esc(s['caption'])}</figcaption></figure>
{paras}<h3>Tools used, and why</h3><ul>{tools}</ul>{code}</section>'''

    body='\n'.join(section(s) for s in source['sections'])
    toc=''.join(f'<li><a href="#{esc(s["id"])}">{esc(s["title"])}</a></li>' for s in source['sections'])
    page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(source['title'])}</title><meta name="description" content="{esc(source['subtitle'])}">
<meta name="theme-color" content="#142a3a"><meta property="og:title" content="{esc(source['title'])}">
<meta property="og:description" content="{esc(source['subtitle'])}"><meta property="og:type" content="article">
<meta property="og:image" content="{esc(base+'images/'+figures[1]['file'])}"><meta name="author" content="{esc(config['author'])}">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/css/style.css"></head>
<body><a class="skip" href="#article">Skip to tutorial</a><header class="topbar"><div class="nav wrap"><a class="brand" href="#top"><span aria-hidden="true"></span>Multimodal RAG</a><nav aria-label="Main navigation"><a href="#overview">The tutorial</a><a href="tutorial.txt">Article source</a><a href="medium.html">Medium edition</a><a class="demo-link" href="{demo}" target="_blank" rel="noopener noreferrer">Try the live demo ↗</a></nav></div></header>
<main id="top"><div class="wrap hero"><p class="eyebrow">A practical guide · 10 illustrated steps</p><h1>{esc(source['title'])}</h1><p class="subtitle">{esc(source['subtitle'])}</p>{author}<div class="hero-actions"><a class="button" href="{demo}" target="_blank" rel="noopener noreferrer">Open the HF demo ↗</a><a class="button secondary" href="#overview">Start the tutorial ↓</a></div><div class="tags"><span>CPU retrieval</span><span>ZeroGPU generation</span><span>No task-specific training</span></div></div>
<div class="wrap layout"><aside class="toc"><details open><summary>In this tutorial</summary><ol>{toc}</ol></details><p><a href="{esc(config['space_source_url'])}">Working Space source ↗</a></p><p><a href="medium.html">Open the plain article for Medium ↗</a></p><p class="small">Every figure is shown in full. Select an image to open its original resolution.</p></aside><article id="article"><div class="intro">{intro}</div>{body}<section id="try-it"><p class="eyebrow">Upload. Ask. Inspect the evidence.</p><h2>Try the complete pipeline</h2>{closing}<p><a class="button" href="{demo}" target="_blank" rel="noopener noreferrer">Load the HF demo documents ↗</a></p></section><footer><p><strong>{esc(config['author'])}</strong><br>{esc(config['affiliation'])}</p><p><a href="{demo}">Live demo</a> · <a href="{esc(config['space_source_url'])}">Space source</a> · <a href="medium.html">Medium edition</a></p><p class="small">The figures use illustrative examples. The tutorial explains implementation choices rather than reporting an answer-quality benchmark.</p></footer></article></div></main></body></html>'''
    (DOCS/'index.html').write_text(page,encoding='utf-8')

    # The importer receives complete, visible article content: no script-driven sections,
    # carousel, sidebar, table, iframe, lazy src replacement or external stylesheet.
    medium_body='\n'.join(section(s,True) for s in source['sections'])
    medium=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(source['title'])}</title><meta name="author" content="{esc(config['author'])}"><link rel="canonical" href="{esc(base)}"><style>body{{max-width:940px;margin:36px auto;padding:0 22px;color:#142a3a;font:18px/1.75 Georgia,serif}}h1,h2,h3{{line-height:1.2}}figure{{margin:30px 0}}img{{max-width:100%;height:auto}}figcaption{{font-size:14px;color:#52636d}}pre{{white-space:pre-wrap;background:#f2f3ef;padding:18px}}a{{color:#087f8c}}</style></head><body><article><h1>{esc(source['title'])}</h1><p><em>{esc(source['subtitle'])}</em></p>{author}<p><a href="{demo}">Try the running Hugging Face demo</a></p>{intro}{medium_body}<h2>Upload. Ask. Inspect the evidence.</h2>{closing}<p><a href="{demo}">Open Multimodal RAG on Hugging Face</a></p><p>By {esc(config['author'])}, {esc(config['affiliation'])}.</p></article></body></html>'''
    (DOCS/'medium.html').write_text(medium,encoding='utf-8')
    pub=ROOT/'publishing';pub.mkdir(exist_ok=True)
    (pub/'medium.html').write_text(medium,encoding='utf-8')

    def markdown(absolute=False):
        lines=['# '+source['title'],'',source['subtitle'],'',f'**{config["author"]}** · {config["affiliation"]}','',f'[Try the running HF demo]({config["demo_url"]})','']+sum(([p,''] for p in source['intro']),[])
        for s in source['sections']:
            f=figures[s['figure']]
            image=(base+'images/' if absolute else 'images/')+f['file']
            lines+=['## '+s['title'],'',f'![{s["caption"]}]({image})','',s['caption'],'']
            lines+=sum(([p,''] for p in s['paragraphs']),[])
            lines+=['### Tools used, and why','']
            lines += [f'- **{name}.** {why}' for name,why in s['tools']]
            lines+=['']
            if s.get('code'):lines+=['```python',s['code'],'```','']
        lines+=['## Upload. Ask. Inspect the evidence.','']+sum(([p,''] for p in source['closing']),[])
        lines+=[f'[Open Multimodal RAG on Hugging Face]({config["demo_url"]})','']
        return '\n'.join(lines)
    (DOCS/'tutorial.md').write_text(markdown(),encoding='utf-8')
    (DOCS/'tutorial.txt').write_text(markdown(),encoding='utf-8')
    (pub/'medium.md').write_text(markdown(True),encoding='utf-8')
    print('Built Pages tutorial + plain Medium HTML + Markdown editions; all contain all 10 approved images.')
    print('Medium image base URL:',base)


if __name__=='__main__':main()
