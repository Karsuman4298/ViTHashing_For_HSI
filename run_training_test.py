import torch
import numpy as np
import subprocess
import os
from losses.csq import CSQLoss

def report_hamming_distances():
    print("=== Pairwise Hamming Distances for 7 Class Centers (32-bit) ===")
    loss = CSQLoss(bit_length=32, num_classes=7)
    centers = loss.hash_centers # shape (7, 32)
    binary_centers = (centers > 0).int()
    dists = []
    for i in range(7):
        for j in range(i + 1, 7):
            dist = torch.sum(binary_centers[i] != binary_centers[j]).item()
            dists.append(dist)
    print(f"Distances: {dists}")
    print(f"Mean Hamming Distance: {np.mean(dists):.2f}")
    print("Note: CSQ centers are initialized randomly in the current implementation, so they are not orthogonal (Hadamard).")
    print("===============================================================\n")

def run_train(bit_length, loss_type):
    print(f"--- Running Training with {bit_length} bits ({loss_type.upper()}) ---")
    cmd = [
        "python3", "train_hsi_hashing.py", 
        "--epochs", "2", 
        "--eval_every", "1", 
        "--hash_bit_length", str(bit_length),
        "--batch_size", "8",
        "--scene", "Houston13",
        "--dataset_root", "hsi_data",
        "--loss", loss_type
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    report_hamming_distances()
    for loss_type in ["csq", "dpn"]:
        for b in [16, 32, 64]:
            run_train(b, loss_type)
