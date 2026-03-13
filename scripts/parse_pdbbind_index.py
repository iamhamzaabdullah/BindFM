#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import parse_pdbbind_index, make_parser
args = make_parser("parse_pdbbind_index").parse_args()
parse_pdbbind_index(args.index_file, args.structures_dir, args.output_csv)
