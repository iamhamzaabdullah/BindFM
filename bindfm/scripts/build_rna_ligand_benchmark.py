#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import build_rna_ligand_benchmark, make_parser
args = make_parser("build_rna_ligand_benchmark").parse_args()
build_rna_ligand_benchmark(args.pdb_dir, args.output)
