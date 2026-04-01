import os
import argparse
import warnings
import torchvision.transforms as transforms
from time import gmtime, strftime

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from CMRxRecon import CMRxReconDataset
# Import de la bonne fonction d'inférence
from improved_diffusion.sampling_util import CMR_sampling_major_vote_func

warnings.filterwarnings('ignore')



def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    result_dir = "./results/flow_eval/"
    model_path = "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1/model44000.pt"
    val_pair_file = "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt"
    val_bs = 8  # Taille de batch par GPU
    diffusion_steps = 30   # Le NFE de ton Flow Matching 100 pour diff !
    vote_num = 1    # 1 pour ODE deterministe sinon 4 pour diffusion !!!

    # Initialisation Multi-GPU
    dist_util.setup_dist()

    # Seul le rank 0 a le droit d'écrire les logs pour éviter les doublons
    if dist_util.get_rank() == 0:
        os.makedirs(result_dir, exist_ok=True)
        logger.configure(dir=result_dir, format_strs=["stdout", "log", "csv"])
    else:
        logger.configure(dir=result_dir, format_strs=[])

    arg_dict = model_and_diffusion_defaults()
    arg_dict["image_size"] = 128
    arg_dict["diffusion_steps"] = diffusion_steps

    if dist_util.get_rank() == 0:
        logger.log(f"Création du modèle (Steps: {diffusion_steps})...")
        
    model, diffusion = create_model_and_diffusion(**arg_dict)

    # Chargement des poids depuis le checkpoint
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ])

    # Chargement du dataset complet (limit_val=False pour la vraie évaluation)
    dataset = CMRxReconDataset(val_pair_file, transform=tsfm, length=-1, limit_val=False)

    if dist_util.get_rank() == 0:
        logger.log("Lancement de l'inférence et de l'évaluation...")

    # Lancement de l'inférence
    CMR_sampling_major_vote_func(
        batch_size=val_bs, 
        diffusion=diffusion, 
        model=model, 
        output_folder=result_dir, 
        dataset=dataset, 
        logger=logger, 
        is_inference=True, 
        vote_num=vote_num 
    )

if __name__ == "__main__":
    main()
