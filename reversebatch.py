# -*- coding: utf-8 -*-
"""
Batch prediction script for TransformerCPI2.0.

Behavior added:
- Skip proteins with raw sequence length >= 8094
- Record skipped rows in CSV instead of trying inference
- Stop on the first CUDA-related fatal error to avoid contaminating later results
"""

import argparse
import csv
import os
import random
from typing import Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from featurizer import featurizer
from model import Encoder, Decoder, Predictor


SKIP_LEN_THRESHOLD = 8094


class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins = [], [], []
                atom, adj, protein = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                data = pack(atoms, adjs, proteins, self.device)
                predicted_scores = self.model(data)
        return predicted_scores


def pack(atoms, adjs, proteins, device):
    # Kept aligned with the original predict.py numerical path.
    atoms = torch.FloatTensor(atoms)
    adjs = torch.FloatTensor(adjs)
    proteins = torch.FloatTensor(proteins)
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0] + 1)
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atoms_len += 1
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N, atoms_len, 34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, 1:a_len + 1, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        adjs_new[i, 0, :] = 1
        adjs_new[i, :, 0] = 1
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, 1:a_len + 1, 1:a_len + 1] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len), dtype=torch.int64, device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)


def parse_fasta(path: str) -> Iterable[Tuple[str, str]]:
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def read_smiles(smiles_arg: str) -> str:
    if os.path.isfile(smiles_arg):
        with open(smiles_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return smiles_arg.strip()


def select_device(mode: str) -> torch.device:
    mode = mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model(device: torch.device):
    pretrain = torch.load("./Bert.pkl", map_location=device, weights_only=False)
    pretrain.to(device)
    for param in pretrain.parameters():
        param.requires_grad = False
    pretrain.eval()

    encoder = Encoder(pretrain, n_layers=3, device=device)
    decoder = Decoder(n_layers=3, dropout=0.2, device=device)
    model = Predictor(encoder, decoder, device)

    model_path = "./DrugRepurpose.pt"
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(loaded, "state_dict"):
        model.load_state_dict(loaded.state_dict())
    else:
        model.load_state_dict(loaded)
    model.to(device)
    model.eval()
    return Tester(model, device)


def score_to_float(score):
    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().reshape(-1)[0].item())
    if isinstance(score, np.ndarray):
        return float(score.reshape(-1)[0])
    if isinstance(score, (list, tuple)):
        return float(score[0])
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Batch score proteins from a FASTA against one SMILES using TransformerCPI2.0")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--smiles", required=True, help="SMILES string or a text file containing one SMILES")
    parser.add_argument("--output", default="batch_scores.csv", help="Output CSV path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of FASTA records to process")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument(
        "--continue-after-cuda-error",
        action="store_true",
        help="Continue even after a CUDA runtime error. Not recommended because later results can be unreliable after a device-side assert.",
    )
    parser.add_argument(
        "--skip-len-threshold",
        type=int,
        default=SKIP_LEN_THRESHOLD,
        help=f"Skip raw protein sequences with length >= this value (default: {SKIP_LEN_THRESHOLD})",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    print(f"Using device: {device}")

    smiles = read_smiles(args.smiles)
    print(f"SMILES: {smiles}")
    print(f"Skip rule: raw sequence length >= {args.skip_len_threshold} => SKIPPED_TOO_LONG")

    tester = load_model(device)

    rows = []
    fatal_cuda_error = False

    fasta_iter = parse_fasta(args.fasta)
    if args.limit is not None:
        from itertools import islice
        fasta_iter = islice(fasta_iter, args.limit)

    for idx, (header, sequence) in enumerate(tqdm(list(fasta_iter), desc="Scoring"), start=1):
        seq_len = len(sequence)

        if seq_len >= args.skip_len_threshold:
            rows.append({
                "Index": idx,
                "Protein": header,
                "SequenceLength": seq_len,
                "Score": "",
                "Status": "SKIPPED_TOO_LONG",
                "ErrorType": "SequenceTooLong",
                "ErrorMessage": f"Skipped because raw sequence length {seq_len} >= {args.skip_len_threshold}",
            })
            continue

        try:
            compounds, adjacencies, proteins = featurizer(smiles, sequence)
            test_set = list(zip(compounds, adjacencies, proteins))
            score = tester.test(test_set)
            score_val = score_to_float(score)
            rows.append({
                "Index": idx,
                "Protein": header,
                "SequenceLength": seq_len,
                "Score": score_val,
                "Status": "OK",
                "ErrorType": "",
                "ErrorMessage": "",
            })
        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            err_type = type(e).__name__
            rows.append({
                "Index": idx,
                "Protein": header,
                "SequenceLength": seq_len,
                "Score": "Error",
                "Status": "ERROR",
                "ErrorType": err_type,
                "ErrorMessage": msg,
            })
            print(f"\n[ERROR] {idx}: {header}\n  {err_type}: {msg}")

            cuda_related = (
                "cuda" in msg.lower()
                or "device-side assert" in msg.lower()
                or "index out of bounds" in msg.lower()
            )
            if device.type == "cuda" and cuda_related and not args.continue_after_cuda_error:
                fatal_cuda_error = True
                print("\nFatal CUDA-related error detected. Stopping to avoid contaminating later results.")
                break

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Index", "Protein", "SequenceLength", "Score", "Status", "ErrorType", "ErrorMessage"],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["Status"] == "OK"]
    skipped_rows = [r for r in rows if r["Status"] == "SKIPPED_TOO_LONG"]
    error_rows = [r for r in rows if r["Status"] == "ERROR"]

    print(f"\nFinished. OK={len(ok_rows)}, SKIPPED_TOO_LONG={len(skipped_rows)}, ERROR={len(error_rows)}")
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["Score"])
        print(f"Best score: {best['Score']:.6f}")
        print(f"Best protein: {best['Protein']}")
    print(f"Output written to: {args.output}")
    if fatal_cuda_error:
        print("NOTE: Execution stopped at the first CUDA-related error. Later proteins were not processed.")


if __name__ == "__main__":
    main()
