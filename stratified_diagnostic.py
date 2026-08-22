import torch
import numpy as np
from pathlib import Path
from data.houston_dataset import HoustonPatchDataset, stratified_split
from eval_hsi_hashing import hash_collapse_diagnostic
from vts_hsi_model import VTSHSIModel

def run_stratified_diagnostic():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = HoustonPatchDataset("hsi_data", scene="Houston13", patch_size=13)
    split = stratified_split(dataset.labels, train_fraction=0.5, query_per_class=100)
    
    # Find one sample per class from the database set
    db_indices = split["database"]
    selected_indices = []
    classes_found = set()
    
    for idx in db_indices:
        _, label = dataset[idx]
        # Valid classes are 1-7 (0 is background and not returned by dataset[idx])
        if label not in classes_found:
            classes_found.add(label)
            selected_indices.append(idx)
        if len(classes_found) == 7:
            break
            
    images = []
    for idx in selected_indices:
        img, _ = dataset[idx]
        images.append(img)
        
    batch = torch.stack(images).to(device)
    
    # Load model trained with CSQ 32-bit
    model = VTSHSIModel(hash_bit_length=32).to(device)
    checkpoint = torch.load("checkpoints/Houston13_csq_32.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
        
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        print("=== Hash Collapse Diagnostic on Stratified Batch ===")
        print(f"Batch size: {len(images)} (one image for each of the {len(classes_found)} classes)")
        hash_collapse_diagnostic(logits)
        print("====================================================")

if __name__ == "__main__":
    run_stratified_diagnostic()
