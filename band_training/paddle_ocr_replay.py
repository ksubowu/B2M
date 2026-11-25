"""Replay frozen Paddle outputs in the existing exact-graph evaluator."""
import json
from collections import defaultdict
from pathlib import Path
from .prepare_paddle_ocr_requests import request_key


class PaddleOCRReplay:
    def __init__(self,path,policy='en_consensus'):
        d=json.loads(Path(path).read_text());self.metadata=dict(d['metadata'],policy=policy)
        self.results={r['key']:r['candidates'] for r in d['results']};self.policy=policy;self.image=None

    def set_image(self,path):self.image=path

    def recognize(self,image,box):
        choices=self.results[request_key(self.image,box)]
        model=self.policy.split('_')[0]
        choices=[r for r in choices if (model=='ensemble' or r['model']==model) and ('single' not in self.policy or r['scale']==1.)]
        if 'consensus' in self.policy and any(r['text'] for r in choices):
            groups=defaultdict(list)
            for r in choices:
                if r['text']:groups[r['normalized']].append(r)
            # Each declared scale/margin/model has one vote. Only literal NFKC
            # + whitespace normalization; no chemistry vocabulary or GT text.
            winner=max(groups.values(),key=lambda g:(len(g),sum(r['confidence'] for r in g)/len(g),max(r['confidence'] for r in g)))
            best=max(winner,key=lambda r:r['confidence'])
            selection='majority normalized literal text across declared crops; mean confidence breaks ties; no GT text or lexicon'
        else:
            best=max(choices,key=lambda r:r['confidence'] if r['text'] else -1.) if choices else dict(text='',normalized='',confidence=0.)
            selection='highest confidence within declared PaddleOCR model/scale policy; no GT text or lexicon'
        return dict(**best,candidates=choices,selection=selection)
