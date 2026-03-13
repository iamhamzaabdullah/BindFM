#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import build_allosteric_benchmark, make_parser
args = make_parser("build_allosteric_benchmark").parse_args()
build_allosteric_benchmark(args.asd, args.output)
