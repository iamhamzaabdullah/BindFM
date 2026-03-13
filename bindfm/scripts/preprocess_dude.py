#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import preprocess_dude, make_parser
args = make_parser("preprocess_dude").parse_args()
preprocess_dude(args.dude_dir, args.output_dir)
