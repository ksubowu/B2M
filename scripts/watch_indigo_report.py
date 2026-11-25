#!/usr/bin/env python3
"""Finish reporting after the persistent training process exits successfully."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--pid',type=int,required=True);args=p.parse_args()
while not (args.run/'results.json').exists():
    try:os.kill(args.pid,0)
    except ProcessLookupError:
        (args.run/'report_status.json').write_text(json.dumps({'status':'training_process_exited_without_results','pid':args.pid},indent=2))
        sys.exit(1)
    time.sleep(15)
subprocess.run([sys.executable,'-m','band_training.report_indigo','--run',str(args.run)],check=True)
(args.run/'report_status.json').write_text(json.dumps({'status':'completed','report':str(args.run/'report.txt')},indent=2))
