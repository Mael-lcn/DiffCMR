"""
To generate img data from the raw mat file
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as utils
SCALE = 100000



class CMRxReconDataset(Dataset):
    def __init__(self, file_path, transform=None, length=-1, limit_val=False):
        self.transform = transform
        self.mean = 0.5
        self.std = 0.5

        with open(file_path, "r") as f:
            train_pairs_lines = f.readlines()
        
        if length > 0 and not limit_val:
            train_pairs_lines = train_pairs_lines[:length]
        if limit_val:
            random.Random(666).shuffle(train_pairs_lines)
            train_pairs_lines = train_pairs_lines[:24]

        #  On crée une liste de pointeurs vers chaque frame
        self.all_slices = []
        print(f"Indexation des volumes dans {file_path.split('/')[-1]}...")
        
        for line in train_pairs_lines:
            parts = line.strip().split(" ")
            if len(parts) < 2: continue
            path, gt_path = parts[0], parts[1]
            
            try:
                # On utilise mmap_mode pour obtenir la forme sans charger le fichier
                vol_shape = np.load(path, mmap_mode='r').shape
                num_frames = vol_shape[0]
                
                for frame_idx in range(num_frames):
                    self.all_slices.append((path, gt_path, frame_idx))
            except Exception as e:
                print(f"Erreur d'indexation sur {path}: {e}")
        
        print(f"Dataset prêt : {len(self.all_slices)} frames au total.")

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, index):
        path, GT_path, frame_idx = self.all_slices[index]

        # --- CORRECTION MMAP : Très important pour Jean Zay ---
        item_full = np.load(path, mmap_mode='r')
        GT_item_full = np.load(GT_path, mmap_mode='r')
        
        # On ne convertit en float32 que la frame extraite
        item = np.array(item_full[frame_idx], dtype=np.float32)
        GT_item = np.array(GT_item_full[frame_idx], dtype=np.float32)

        if self.transform:
            # On empile pour que ToTensor() traite les deux images de la même façon
            data = np.stack((item, GT_item), axis=-1) # (H, W, 2)
            transformed_data = self.transform(data)     # Devient (2, H, W)

            # Extraction des canaux (C, H, W)
            output = {
                "input": transformed_data[0:1, :, :], # Canal 0
                "GT": transformed_data[1:2, :, :],    # Canal 1
                "ipath": path, 
                "gtpath": GT_path
            }
        else:
            output = {
                "input": torch.from_numpy(item).unsqueeze(0), 
                "GT": torch.from_numpy(GT_item).unsqueeze(0), 
                "ipath": path, 
                "gtpath": GT_path
            }
            
        return output



if __name__ == "__main__":
    print("\n--- Lancement du test local ---")

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256),                  # On zoome sur le coeur
        transforms.Resize((64, 64), antialias=True), # On réduit à la taille de ton modèle
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    test_file = "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt"

    if os.path.exists(test_file):
        # On limite le scan à 5 patients pour que le test démarre instantanément
        training_set = CMRxReconDataset(test_file, transform=tsfm, length=5)

        print(f"Nombre total de frames générées par ces 5 patients : {len(training_set)}")

        if len(training_set) > 0:
            pair0 = training_set[0] # On prend la toute première frame du premier patient

            # Utilisation de normalize=True pour éviter que l'image soit toute noire
            utils.save_image(pair0["input"], "./test_dataset_input.png", normalize=True)
            utils.save_image(pair0["GT"], "./test_dataset_GT.png", normalize=True)

            print("Test réussi !")
            print("Les images test_dataset_input.png et test_dataset_GT.png ont été sauvegardées dans le dossier courant (./).")
    else:
        print(f"Erreur : Le fichier {test_file} est introuvable.")
