import argparse
import os
import warnings
import torch
import torch.distributed as dist
import torchvision.transforms as transforms

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from CMRxRecon import CMRxReconDataset
from improved_diffusion.sampling_util import CMR_sampling_major_vote_func



warnings.filterwarnings('ignore')


def create_argparser():
    """
    Définition des arguments pour l'inférence.
    On reprend la base de model_and_diffusion_defaults.
    """
    defaults = dict(
        # --- Chemins ---
        model_path="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/logs_run1/model44000.pt",
        val_pair_file="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt",
        result_dir="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/flow/eval/",

        image_size=128,
        diffusion_steps=100,

        batch_size=8,
        vote_num=4,
        num_workers=4,
    )

    final_defaults = model_and_diffusion_defaults()
    final_defaults.update(defaults)

    parser = argparse.ArgumentParser(description="Inférence et Évaluation DiffCMR / Flow Matching")
    add_dict_to_argparser(parser, final_defaults)
    return parser


def main():
    # Chargement des paramètres
    args = create_argparser().parse_args()

    # Initialisation distribuée (Multi-GPU)
    if torch.cuda.is_available():
        dist_util.GPUS_PER_NODE = torch.cuda.device_count()
    dist_util.setup_dist()

    # Configuration du logger
    if dist.get_rank() == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        logger.configure(dir=args.result_dir, format_strs=["stdout", "log", "csv"])
    else:
        logger.configure(dir=args.result_dir, format_strs=[])

    # Création du Modèle et de la Diffusion
    logger.log(f"Loading model and diffusion (Steps: {args.diffusion_steps})...")
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)

    # Chargement des poids
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Pipeline de données
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
    ])

    dataset = CMRxReconDataset(args.val_pair_file, transform=tsfm, length=-1, limit_val=False)

    if dist.get_rank() == 0:
        logger.log(f"Taille du jeu de validation : {len(dataset)}")
        logger.log("Lancement de l'évaluation complète...")

    # Exécution de la fonction d'évaluation
    CMR_sampling_major_vote_func(
        batch_size=args.batch_size,
        diffusion=diffusion,
        model=model,
        output_folder=args.result_dir,
        dataset=dataset,
        logger=logger,
        is_inference=True,
        vote_num=args.vote_num
    )

    if dist.get_rank() == 0:
        logger.log(f"Évaluation terminée. Résultats disponibles dans : {args.result_dir}")

if __name__ == "__main__":
    main()
