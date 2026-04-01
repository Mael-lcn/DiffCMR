from time import gmtime, strftime

import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import warnings
warnings.filterwarnings('ignore')

import argparse

from CMRxRecon import CMRxReconDataset
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())



def create_argparser():
    """
    Regroupe TOUS les hyperparamètres ici. 
    Plus aucune valeur en dur n'est présente dans le script principal.
    """
    defaults = dict(
        # --- Chemins et Logs ---
        logdir="./log/t1_08_128/",
        trainpairfile="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt",
        valpairfile="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt",

        # --- Hyperparamètres d'entraînement ---
        image_size=128,
        batch_size=8,
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,
        ema_rate="0.9999",

        # --- Optimisation matérielle ---
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_workers=4,

        # --- Fréquence de sauvegarde et Logs ---
        log_interval=500,
        save_interval=2000,
        start_print_iter=1000000,
        resume_checkpoint="",
        run_without_test=True,

        # --- Diffusion ---
        schedule_sampler="uniform",
        clip_denoised=False,
        model_type="diffusion"
    )

    final_defaults = model_and_diffusion_defaults()
    # On ajoute les paramètres par défaut du modèle
    final_defaults.update(defaults)

    parser = argparse.ArgumentParser(description="Entraînement Diffusion model")
    # On donne final_defaults au parser
    add_dict_to_argparser(parser, final_defaults) 
    return parser


def main():
    # Chargement de tous les paramètres
    args = create_argparser().parse_args()

    # Initialisation distribuée 
    if torch.cuda.is_available():
        dist_util.GPUS_PER_NODE = torch.cuda.device_count()
    dist_util.setup_dist()

    # SEUL LE RANK 0 EST AUTORISÉ À ACTIVER LE LOGGER COMPLET
    if dist.get_rank() == 0:
        logger.configure(dir=args.logdir, format_strs=["stdout", "log", "csv"])
    else:
        # Les autres GPUs ne font rien (désactivation silencieuse)
        logger.configure(dir=args.logdir, format_strs=[])

    # Création du Modèle et de la Diffusion
    logger.log("creating model and diffusion...")
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(model_args)

    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # 4. Pipeline de données
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    dataset = CMRxReconDataset(args.trainpairfile, transform=tsfm, length=-1)
    val_dataset = CMRxReconDataset(args.valpairfile, transform=tsfm, length=-1, limit_val=True)
    logger.log(f"Taille du jeu d'entraînement : {len(dataset)}")

    # Sampler Multi-GPU
    sampler = DistributedSampler(dataset, shuffle=True)
    loader_args = dict(num_workers=args.num_workers, pin_memory=True)
    
    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        sampler=sampler,
        drop_last=True, 
        **loader_args
    )

    def load_gen(loader):
        while True:
            yield from loader
            
    train_gen = load_gen(train_loader)

    # 5. Boucle d'entraînement
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_gen,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip_denoised=args.clip_denoised,
        logger=logger,
        image_size=args.image_size,
        val_dataset=val_dataset,
        run_without_test=args.run_without_test,
    ).run_loop(start_print_iter=args.start_print_iter)


if __name__ == "__main__":
    main()
