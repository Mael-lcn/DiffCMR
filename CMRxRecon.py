"""
To generate img data from the raw mat file
"""
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils
SCALE = 100000
import random



class CMRxReconDataset(Dataset):
    
    def __init__(self, file_path, transform=None, length=-1, limit_val=False):
        """
        root_dir: absolute path of "ChallengData"
        file_path: the train_pair_file.txt
        """
        self.name_dict = {"MultiCoil":{"AccFactor04":"kspace_sub04",
                                       "AccFactor08":"kspace_sub08",
                                       "AccFactor10":"kspace_sub10",
                                       "FullSample":"kspace_full"}, 
                          "SingleCoil":{"AccFactor04":"kspace_single_sub04",
                                       "AccFactor08":"kspace_single_sub08",
                                       "AccFactor10":"kspace_single_sub10",
                                       "FullSample":"kspace_single_full"}}

        self.mean = 0.5
        self.std = 0.5

        self.file_path = file_path
        file_obj = open(self.file_path, "r")
        self.train_pairs = file_obj.readlines()
        if length>0 and not limit_val:
            self.train_pairs = self.train_pairs[:length]
        if limit_val:
            random.Random(666).shuffle(self.train_pairs)
            self.train_pairs = self.train_pairs[:32]
        self.transform = transform
        file_obj.close()


    def __len__(self):
        return len(self.train_pairs)
    
    def __getitem__(self, index):
        path, GT_path = self.train_pairs[index].replace("\n","").split(" ")
        
        # Chargement des volumes complets (Frames, H, W)
        item_full = np.float32(np.load(path))
        GT_item_full = np.float32(np.load(GT_path))

        num_frames = item_full.shape[0]
        random_frame_idx = random.randint(0, num_frames - 1)

        # On extrait la frame aléatoire pour obtenir une image 2D (H, W)
        item = item_full[random_frame_idx]
        GT_item = GT_item_full[random_frame_idx]

        output = {"input": item, "GT": GT_item}
        if self.transform:
            # Stacking devient (H, W, 2), parfaitement compatible avec ToTensor()
            data = np.stack((item, GT_item), axis=-1)
            transformed_data = self.transform(data)
            output = {"input": transformed_data[0,:,:].unsqueeze(0), "GT": transformed_data[1,:,:].unsqueeze(0), "ipath":path, "gtpath":GT_path}
        return output


import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as utils

class CMRxReconDataset(Dataset):
    def __init__(self, file_path, transform=None, length=-1, limit_val=False):
        self.transform = transform
        self.mean = 0.5
        self.std = 0.5

        # 1. Lecture du fichier texte
        with open(file_path, "r") as f:
            train_pairs_lines = f.readlines()
        
        if length > 0 and not limit_val:
            train_pairs_lines = train_pairs_lines[:length]
        if limit_val:
            random.Random(666).shuffle(train_pairs_lines)
            train_pairs_lines = train_pairs_lines[:24]

        # 2. APLATISSEMENT DU DATASET (1 item = 1 frame)
        self.all_slices = []
        print(f"Scan des volumes dans {file_path.split('/')[-1]}...")
        
        for line in train_pairs_lines:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            path, gt_path = parts[0], parts[1]
            
            try:
                # mmap_mode='r' lit juste l'en-tête pour avoir la shape sans charger la RAM
                vol_shape = np.load(path, mmap_mode='r').shape
                num_frames = vol_shape[0]
                
                for frame_idx in range(num_frames):
                    self.all_slices.append((path, gt_path, frame_idx))
            except Exception as e:
                print(f"Attention, erreur de lecture sur {path}: {e}")
        
        print(f"Dataset prêt : {len(self.all_slices)} images 2D (frames) au total.")

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, index):
        path, GT_path, frame_idx = self.all_slices[index]

        item_full = np.float32(np.load(path))
        GT_item_full = np.float32(np.load(GT_path))
        
        item = item_full[frame_idx]
        GT_item = GT_item_full[frame_idx]

        if self.transform:
            data = np.stack((item, GT_item), axis=-1)
            transformed_data = self.transform(data)
            output = {
                "input": transformed_data[0,:,:].unsqueeze(0), 
                "GT": transformed_data[1,:,:].unsqueeze(0), 
                "ipath": path, 
                "gtpath": GT_path
            }
        else:
            output = {
                "input": torch.tensor(item).unsqueeze(0), 
                "GT": torch.tensor(GT_item).unsqueeze(0), 
                "ipath": path, 
                "gtpath": GT_path
            }
            
        return output


if __name__ == "__main__":
    print("\n--- Lancement du test local ---")

    # Intégration du fameux Crop Central suivi du Resize !
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256),                  # On zoome sur le coeur
        transforms.Resize((64, 64), antialias=True), # On réduit à la taille de ton modèle
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    # J'ai remplacé le chemin de txiang par ton chemin sur Jean Zay
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
