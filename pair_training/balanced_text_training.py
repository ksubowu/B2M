"""Scale/character balancing and train-only fresh alphanumeric text.

No heldout examples or OCR predictions drive training sample selection.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import string
from PIL import Image, ImageDraw, ImageFont

ALPHABET=string.ascii_letters+string.digits


class ScaleCharacterSampler:
    def __init__(self, items, text, scale):
        self.groups=defaultdict(lambda:defaultdict(list));self.draws=Counter()
        for item in items:
            for char in set(text(item)) & set(ALPHABET):
                self.groups[int(scale(item))][char].append(item)
        if not self.groups:raise ValueError('No text-bearing training examples')
        self.scales=sorted(self.groups)

    def choice(self,rng):
        scale=rng.choice(self.scales);char=rng.choice(sorted(self.groups[scale]))
        self.draws[scale]+=1
        return rng.choice(self.groups[scale][char])

    def coverage(self):
        return {str(s):dict(characters=len(g),missing=''.join(sorted(set(ALPHABET)-g.keys()))) for s,g in self.groups.items()}


class OnlineCharacters:
    """Auxiliary OCR images only; no invented chemical expansion/ground truth."""
    def __init__(self, forbidden, exclude_fonts=()):
        self.forbidden=set(forbidden)
        self.fonts=[p for p in Path('/usr/share/fonts/truetype/dejavu').glob('*.ttf')
                    if 'Condensed' not in p.name and p.name not in set(exclude_fonts)]
        if not self.fonts:raise RuntimeError('No installed OCR auxiliary fonts')

    def sample(self,rng):
        while True:
            n=rng.randint(2,9);chars=rng.choices(ALPHABET,k=n)
            chars[rng.randrange(n)]=rng.choice(ALPHABET);word=''.join(chars)
            if word not in self.forbidden:break
        # Draw large, then area-downsample to a genuinely small clear glyph.
        size=rng.choice([12,14,16,20,24,28]);font=ImageFont.truetype(str(rng.choice(self.fonts)),size*3)
        probe=ImageDraw.Draw(Image.new('L',(1,1)));x,y,u,v=probe.textbbox((0,0),word,font=font)
        image=Image.new('RGB',(u-x+18,v-y+18),'white');draw=ImageDraw.Draw(image)
        draw.text((9-x,9-y),word,font=font,fill=rng.choice(['black','#102060','#502010']))
        image=image.resize((max(1,round(image.width/3)),max(1,round(image.height/3))),Image.Resampling.BOX).convert('L')
        return image,word


def balanced_validation(rows, limit, scale):
    """Smoke cap covers every available scale, instead of the first controls."""
    if not limit:return list(rows)
    groups=defaultdict(list)
    for row in rows:groups[scale(row)].append(row)
    result=[]
    while len(result)<limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(result)<limit:result.append(groups[key].pop(0))
    return result


def retention_eligible(baseline, candidate, tolerance=0.):
    """Every measured domain/scale must retain its initial whole-graph rate."""
    failures=[]
    for key,before in baseline.items():
        after=candidate.get(key)
        if not after or after['total']!=before['total']:failures.append(key+':membership');continue
        if after['correct']/max(1,after['total'])+tolerance+1e-12 < before['correct']/max(1,before['total']):
            failures.append(key)
    return not failures,failures


def whole_graph_rank(metrics, *local_tiebreaks):
    """Select actual molecular accuracy first; local scores only break ties."""
    return (metrics['new']['correct'],metrics['old']['correct'],*local_tiebreaks)
