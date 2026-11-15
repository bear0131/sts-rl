"""Local game artwork for trajectory contact sheets; assets stay out of Git."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile


def class_strings(data):
    """Read constant metadata (not Java source) for exact relic asset names."""
    if data[:4] != b'\xca\xfe\xba\xbe':
        raise ValueError('invalid class metadata')
    count, = struct.unpack_from('>H', data, 8)
    position, index, values = 10, 1, []
    while index < count:
        tag = data[position]
        position += 1
        if tag == 1:
            size, = struct.unpack_from('>H', data, position)
            position += 2
            values.append(data[position:position+size].decode('utf-8', errors='replace'))
            position += size
        elif tag in (3, 4, 9, 10, 11, 12, 17, 18):
            position += 4
        elif tag in (5, 6):
            position += 8
            index += 1
        elif tag in (7, 8, 16, 19, 20):
            position += 2
        elif tag == 15:
            position += 3
        else:
            raise ValueError('unknown class constant')
        index += 1
    return values


def build_catalog(jar_path, output, rows):
    from PIL import Image, ImageOps
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = dict(cards={card[0] for row in rows for card in row['deck']},
               relics={relic for row in rows for relic in row['relics']})
    catalog = dict(cards={}, relics={})
    normalize = lambda text: re.sub('[^a-z0-9]', '', text.lower())
    with zipfile.ZipFile(jar_path) as archive:
        paths = set(archive.namelist())
        portraits = [n for n in paths if n.startswith('images/1024Portraits/') and n.endswith('.png')]
        constants = [class_strings(archive.read(n)) for n in sorted(paths)
            if n.startswith('com/megacrit/cardcrawl/relics/') and n.endswith('.class') and '$' not in n]
        for kind in ids:
            names = json.loads(archive.read(f'localization/zhs/{kind}.json').decode('utf-8-sig'))
            for identity in sorted(ids[kind]):
                member = None
                if kind == 'cards':
                    stem = {'Strike_R': 'strike', 'Defend_R': 'defend'}.get(identity, identity)
                    choices = [n for n in portraits if normalize(Path(n).stem) == normalize(stem)]
                    preferred = [n for n in choices if n.startswith(('images/1024Portraits/red/', 'images/1024Portraits/curse/'))]
                    if preferred:
                        choices = preferred
                    if identity == 'Clothesline':
                        choices = [n for n in choices if '/attack/' in n]
                    if len(choices) == 1:
                        member = choices[0]
                else:
                    matches = [sorted({s for s in values if s.endswith('.png')})
                        for values in constants if identity in values and any(s.endswith('.png') for s in values)]
                    if len(matches) == 1 and len(matches[0]) == 1:
                        member = next((f'images/{folder}/{matches[0][0]}' for folder in ('largeRelics', 'relics')
                                       if f'images/{folder}/{matches[0][0]}' in paths), None)
                # Canonical simulator IDs sometimes differ only by spaces/case
                # (e.g. Panic Button / PanicButton). Never guess a translation.
                name_key = identity
                if name_key not in names:
                    aliases = [key for key in names if normalize(key) == normalize(identity)]
                    if len(aliases) == 1:
                        name_key = aliases[0]
                item = dict(name=names.get(name_key, {}).get('NAME', identity), image=None)
                if member:
                    raw = archive.read(member)
                    filename = hashlib.sha256(raw + kind.encode() + b'thumbnail-v1').hexdigest() + '.png'
                    target = output / filename
                    if not target.exists():
                        art = Image.open(io.BytesIO(raw)).convert('RGBA')
                        if kind == 'relics':
                            bounds = art.getchannel('A').getbbox()
                            if bounds:
                                art = art.crop(bounds)
                        art = ImageOps.contain(art, (200, 150) if kind == 'cards' else (112, 112), Image.Resampling.LANCZOS)
                        temporary = target.with_suffix('.tmp')
                        art.save(temporary, format='PNG')
                        temporary.replace(target)
                    item['image'] = '/trajectory-assets/' + filename
                catalog[kind][identity] = item
    return catalog
