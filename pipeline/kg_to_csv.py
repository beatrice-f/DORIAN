import glob

import pandas as pd
import rdflib
from tqdm import tqdm

kg_files = [
    rdflib.Dataset().parse(f, format="nquads") for f in tqdm(glob.glob("kg/*.nq"))
]

triples = [[str(s), str(p), str(o)] for f in kg_files for s, p, o, _ in f]
pd.DataFrame(triples).to_csv("kg.tsv", index=False, header=None, sep="\t")
