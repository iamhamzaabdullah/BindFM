#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import create_aptabase_placeholder, make_parser
args = make_parser("create_aptabase_placeholder").parse_args()
create_aptabase_placeholder(args.output, args.n)
