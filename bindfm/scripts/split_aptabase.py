#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import split_aptabase, make_parser
args = make_parser("split_aptabase").parse_args()
split_aptabase(args.input, args.output_dir, args.test_targets)
