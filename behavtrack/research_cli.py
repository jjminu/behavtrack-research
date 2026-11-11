"""Scriptable equivalents of the research UI; Windows paths are normal arguments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .integrity import read_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    recover=sub.add_parser("recover");recover.add_argument("folder")
    for name in ("inspect","freeze"):
        p=sub.add_parser(name);p.add_argument("spec")
        if name=="freeze":p.add_argument("output")
    p=sub.add_parser("train");p.add_argument("dataset");p.add_argument("models")
    p=sub.add_parser("evaluate");p.add_argument("dataset");p.add_argument("candidate");p.add_argument("--role",choices=["development","final"],default="development");p.add_argument("--baseline")
    p=sub.add_parser("detector-export");p.add_argument("spec");p.add_argument("output");p.add_argument("--reviewer",required=True)
    p=sub.add_parser("detector-train");p.add_argument("export");p.add_argument("config");p.add_argument("--epochs",type=int,default=5)
    p=sub.add_parser("detector-import");p.add_argument("snapshot");p.add_argument("output")
    args=parser.parse_args(argv)
    from . import ml_workflow as ml
    if args.command=="recover":
        from .recording_io import recover
        result=recover(args.folder)
    elif args.command in ("inspect","freeze"):
        spec=read_json(args.spec);base=Path(args.spec).resolve().parent
        result=ml.inspect_dataset(spec,base) if args.command=="inspect" else ml.freeze_dataset(spec,args.output,base)
    elif args.command=="train":result=ml.train_candidate(args.dataset,args.models)
    elif args.command=="evaluate":result=ml.evaluate_candidate(args.dataset,args.candidate,args.role,args.baseline)
    elif args.command=="detector-export":
        from .dlc_bridge import export_detector
        result=export_detector(read_json(args.spec)["entries"],args.output,args.reviewer)
    elif args.command=="detector-import":
        from .dlc_bridge import import_detector
        result=import_detector(args.snapshot,args.output)
    else:
        from .dlc_bridge import launch
        result=launch(Path(__file__).resolve().parents[1],args.export,args.config,epochs=args.epochs)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
