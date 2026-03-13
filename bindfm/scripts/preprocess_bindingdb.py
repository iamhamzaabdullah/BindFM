#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import preprocess_bindingdb, make_parser
args = make_parser("preprocess_bindingdb").parse_args()
preprocess_bindingdb(args.input, args.output_dir, args.test_fraction, args.seed)
