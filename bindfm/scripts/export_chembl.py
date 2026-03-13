#!/usr/bin/env python3
"""BindFM: ChEMBL SQLite exporter"""
import sys, os
sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import export_chembl, make_parser
args = make_parser("export_chembl").parse_args()
export_chembl(args.db, args.output)
