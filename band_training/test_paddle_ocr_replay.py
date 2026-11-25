import json,tempfile,unittest
from pathlib import Path
from .paddle_ocr_replay import PaddleOCRReplay
from .prepare_paddle_ocr_requests import request_key


class PaddleReplayTests(unittest.TestCase):
    def test_consensus_can_reject_one_high_confidence_outlier(self):
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'x.png';box=[0,0,10,10]
            candidates=[dict(model='en',scale=s,text='R2',normalized='R2',confidence=.8) for s in [1.,2.,3.]]
            candidates.append(dict(model='en',scale=4.,text='Rz',normalized='Rz',confidence=.99))
            path=Path(temp)/'r.json';path.write_text(json.dumps(dict(metadata={},results=[dict(key=request_key(image,box),candidates=candidates)])))
            reader=PaddleOCRReplay(path,'en_consensus');reader.set_image(image)
            self.assertEqual(reader.recognize(None,box)['text'],'R2')

    def test_policy_selection_never_sees_expected_label(self):
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'image.png';box=[1,2,30,40]
            candidates=[dict(model='en',scale=1.,text='wrong',normalized='wrong',confidence=.5),
                        dict(model='en',scale=2.,text='CF3',normalized='CF3',confidence=.9),
                        dict(model='latin',scale=1.,text='latin',normalized='latin',confidence=.95)]
            path=Path(temp)/'candidates.json'
            path.write_text(json.dumps(dict(metadata={},results=[dict(key=request_key(image,box),candidates=candidates)])))
            for policy,expected in [('en_single','wrong'),('en_multiscale','CF3'),('ensemble_multiscale','latin')]:
                reader=PaddleOCRReplay(path,policy);reader.set_image(image)
                self.assertEqual(reader.recognize(None,box)['text'],expected)

    def test_request_key_normalizes_numeric_types(self):
        self.assertEqual(request_key('x.png',[1,2,3,4]),request_key('x.png',[1.,2.,3.,4.]))


if __name__=='__main__':unittest.main()
