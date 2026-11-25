"""Geometry-only CLI: atom_boxes (xywh) + bond_instances (label/quad/line)."""
import argparse,json
from pathlib import Path
from .geometry import assemble_band_lines

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--line-association',choices=['forward','reverse_nearest','global_center','exclusive_box'],default='forward');args=p.parse_args()
    data=json.loads(args.input.read_text())
    result=assemble_band_lines(data['atom_boxes'],data['bond_instances'],line_association=args.line_association)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(bonds=len(result['bonds']),unlinked=len(result['unlinked_band_indices']))))

if __name__=='__main__':main()
