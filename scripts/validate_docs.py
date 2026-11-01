"""Validate the tutorial, Medium export, original images and local documentation links."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.links=[];self.images=[];self.tags=[];self.h1=0

    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for key in ['href','src']:
            if key in a:self.links.append(a[key])
        if tag=='img':self.images.append(a)
        if tag=='h1':self.h1+=1


def main():
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    config=json.loads((ROOT/'site-config.json').read_text())
    manifest=json.loads((DOCS/'figure-manifest.json').read_text())
    tutorial=json.loads((ROOT/'content/tutorial.json').read_text())
    base=config['public_base_url'].rstrip('/')+'/'
    check([f['number'] for f in manifest]==list(range(1,11)),'Figure manifest must contain 1..10 in order')
    check([s['figure'] for s in tutorial['sections']]==list(range(1,11)),'Tutorial must use all ten figures in order')
    check(len(list((DOCS/'images').glob('*.png')))==10,'Expected exactly ten original PNGs')
    for f in manifest:
        path=DOCS/'images'/f['file']
        check(path.is_file(),f'Missing figure {f["number"]}')
        check(hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'],f'Original figure bytes changed: {path.name}')
        with Image.open(path) as im:
            check(im.size==(f['width'],f['height']),f'Image dimensions changed: {path.name}')
            im.verify()
    for filename in ['index.html','medium.html']:
        text=(DOCS/filename).read_text()
        p=Page();p.feed(text)
        check(p.h1==1,f'{filename}: expected one h1')
        check(len(p.ids)==len(set(p.ids)),f'{filename}: duplicate IDs')
        check(len(p.images)==10,f'{filename}: must contain exactly ten figures')
        check('script' not in p.tags,f'{filename}: article must not require JavaScript')
        check(config['demo_url'] in text,f'{filename}: missing HF demo link')
        for im,f in zip(p.images,manifest):
            expected=('images/' if filename=='index.html' else base+'images/')+f['file']
            check(im.get('src')==expected,f'{filename}: wrong image source/order')
            check(bool(im.get('alt')),f'{filename}: missing image alt text')
            check(im.get('width')==str(f['width']) and im.get('height')==str(f['height']),f'{filename}: wrong intrinsic aspect ratio')
        for link in p.links:
            u=urlsplit(link)
            if u.scheme or u.netloc:
                # Public URLs in the Medium export must map to committed image files.
                if link.startswith(base+'images/'):
                    check((DOCS/'images'/unquote(link.split('/')[-1])).is_file(),f'{filename}: unbacked public image URL')
                continue
            if u.path:
                check(not u.path.startswith('/'),f'{filename}: root-relative path breaks project Pages')
                check((DOCS/unquote(u.path)).is_file(),f'{filename}: missing local target {link}')
            if u.fragment and not u.path:check(u.fragment in p.ids,f'{filename}: missing anchor {link}')
        if filename=='medium.html':
            check(not any(tag in p.tags for tag in ['table','iframe','nav','aside','script']), 'Medium edition contains nonportable structures')
            for s in tutorial['sections']:check(s['id'] in p.ids,'Medium export is missing a tutorial section')
    check(not any(any(part.startswith('.') for part in p.relative_to(DOCS).parts) for p in DOCS.rglob('*')), 'Dotfile or dotfolder found in clean repository')
    check((DOCS/'medium.html').read_bytes()==(ROOT/'publishing/medium.html').read_bytes(),'Medium HTML copies differ')
    for relative in ['docs/tutorial.md','publishing/medium.md']:
        text=(ROOT/relative).read_text()
        check(len(re.findall(r'!\[',text))==10,f'{relative}: expected ten image embeds')
    # Validate local Markdown file targets. Same-file heading anchors are ignored.
    for relative in ['README.md','docs/README.md','docs/tutorial.md','publishing/MEDIUM_GUIDE.md']:
        path=ROOT/relative
        for link in re.findall(r'\]\(([^\s)]+)\)',path.read_text()):
            u=urlsplit(link)
            if not u.scheme and u.path:
                check((path.parent/unquote(u.path)).exists(),f'{relative}: broken Markdown link {link}')
    readme=(ROOT/'README.md').read_text()
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name=re.split(r'[<>=!~\[]',line)[0]
            check(name in readme,f'Undocumented dependency: {name}')
    css=(DOCS/'assets/css/style.css').read_text()
    check('height:auto' in css and 'object-fit:cover' not in css,'Figure CSS may crop images')
    check('TCG CREST' not in (DOCS/'index.html').read_text(),'Wrong affiliation')
    if errors:
        print('\n'.join('FAIL: '+e for e in errors));return 1
    print('PASS: all 10 original chat-approved images match their SHA-256 hashes and dimensions.')
    print('PASS: Pages and Medium HTML contain all 10 figures in order with alt text and original aspect ratios.')
    print('PASS: local links/anchors resolve; Pages paths are project-relative; HF links are present.')
    print('PASS: Medium export uses complete static article content and absolute image URLs; no tables/scripts/iframes.')
    print('PASS: both Markdown articles contain 10 images; every application dependency is documented.')
    print('NOTE: public Medium image URLs become reachable after publishing the configured Pages site.')
    return 0


if __name__=='__main__':sys.exit(main())
