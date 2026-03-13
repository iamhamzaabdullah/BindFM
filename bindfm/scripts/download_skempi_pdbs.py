#!/usr/bin/env python3
import sys, os; sys.argv[0] = os.path.splitext(os.path.basename(__file__))[0]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocessing_utils import download_skempi_pdbs, make_parser
args = make_parser("download_skempi_pdbs").parse_args()
download_skempi_pdbs(args.skempi_csv, args.output_dir, args.n_workers)
