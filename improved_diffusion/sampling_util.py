import math
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia.enhance import denormalize
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import dist_util
from .metrics import FBound_metric, WCov_metric
# from datasets.monu import MonuDataset
from .utils import set_random_seed_for_iterations
from .Evaluation import ssim, mse, nmse, psnr
from torch.utils.data.distributed import DistributedSampler
from .Evaluation import psnr, ssim, nmse



cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), FBound_metric(predict, target)



def CMR_sampling_major_vote_func(batch_size, diffusion, model, output_folder, dataset, logger, is_inference=False, vote_num=1):
    # Configuration du Dataloader Multi-GPU
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=False)

    # --- Listes pour les métriques ---
    psnr_list, ssim_list, nmse_list, time_list = [], [], [], []

    model.eval()

    for b, batch in enumerate(dataloader):
        condition_on = batch["input"].to(dist_util.dev())
        GT = batch["GT"].to(dist_util.dev())

        # Préparation de l'entrée pour le modèle (concaténation condition + bruit)
        shape = (condition_on.shape[0], 1, condition_on.shape[2], condition_on.shape[3])

        # Chronométrage de l'inférence
        start_time = time.time()

        # Pour le Flow Matching, un seul passage suffit. Si vote_num > 1, on ferait une moyenne.
        final_sample = torch.zeros_like(condition_on)
        for _ in range(vote_num):
            noise = torch.randn_like(condition_on)
            # Adapte cette ligne si ton modèle prend la condition via model_kwargs
            #model_kwargs = {} 
            model_kwargs = {"conditioned_image": condition_on} # (Dé-commente si ton code l'exige)

            sample = diffusion.p_sample_loop(
                model,
                shape=shape,
                noise=noise,
                clip_denoised=True,
                model_kwargs=model_kwargs
            )
            final_sample += sample
        final_sample = final_sample / vote_num

        end_time = time.time()
        inference_time = (end_time - start_time) / condition_on.shape[0] # Temps par image

        final_sample = (final_sample + 1.0) / 2.0
        final_sample = torch.clamp(final_sample, 0.0, 1.0)
        
        GT = (GT + 1.0) / 2.0
        GT = torch.clamp(GT, 0.0, 1.0)
        
        condition_on = (condition_on + 1.0) / 2.0
        condition_on = torch.clamp(condition_on, 0.0, 1.0)

        # Sauvegarde des images
        if b == 0 and dist.get_rank() == 0:
            os.makedirs(output_folder, exist_ok=True)
            for img_idx in range(min(4, final_sample.shape[0])): # Sauvegarde 4 images du premier batch
                tvu.save_image(final_sample[img_idx], os.path.join(output_folder, f"recon_b{b}_i{img_idx}.png"), normalize=True)
                tvu.save_image(GT[img_idx], os.path.join(output_folder, f"gt_b{b}_i{img_idx}.png"), normalize=True)
                tvu.save_image(condition_on[img_idx], os.path.join(output_folder, f"input_b{b}_i{img_idx}.png"), normalize=True)

        # Calcul des métriques du Challenge
        pred_np = final_sample.squeeze(1).cpu().numpy()
        gt_np = GT.squeeze(1).cpu().numpy()

        for i in range(pred_np.shape[0]):
            gt_3d = np.expand_dims(gt_np[i], axis=0)
            pred_3d = np.expand_dims(pred_np[i], axis=0)

            val_psnr = psnr(gt_np[i], pred_np[i]).item()
            val_ssim = ssim(gt_3d, pred_3d)[0].item()
            val_nmse = nmse(gt_np[i], pred_np[i]).item()

            psnr_list.append(val_psnr)
            ssim_list.append(val_ssim)
            nmse_list.append(val_nmse)
            time_list.append(inference_time)

    # Récupération des scores de tous les GPUs
    device = dist_util.dev()
    local_psnr = torch.tensor(psnr_list, device=device)
    local_ssim = torch.tensor(ssim_list, device=device)
    local_nmse = torch.tensor(nmse_list, device=device)
    local_time = torch.tensor(time_list, device=device)

    # Création des listes pour recevoir les données des autres GPUs
    gathered_psnr = [torch.zeros_like(local_psnr) for _ in range(dist.get_world_size())]
    gathered_ssim = [torch.zeros_like(local_ssim) for _ in range(dist.get_world_size())]
    gathered_nmse = [torch.zeros_like(local_nmse) for _ in range(dist.get_world_size())]
    gathered_time = [torch.zeros_like(local_time) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_psnr, local_psnr)
    dist.all_gather(gathered_ssim, local_ssim)
    dist.all_gather(gathered_nmse, local_nmse)
    dist.all_gather(gathered_time, local_time)

    # Le GPU 0 calcule la moyenne finale et écrit les logs
    if dist.get_rank() == 0:
        total_psnr = torch.cat(gathered_psnr).mean().item()
        total_ssim = torch.cat(gathered_ssim).mean().item()
        total_nmse = torch.cat(gathered_nmse).mean().item()
        total_time = torch.cat(gathered_time).mean().item()

        logger.log("\n" + "="*40)
        logger.log("=== RÉSULTATS MÉTRIQUES (TEST COMPLET) ===")
        logger.log("="*40)
        logger.log(f"PSNR Moyen        : {total_psnr:.4f} dB")
        logger.log(f"SSIM Moyen        : {total_ssim:.4f}")
        logger.log(f"NMSE Moyen        : {total_nmse:.6f}")
        logger.log(f"Temps par Image   : {total_time:.4f} secondes")
        logger.log("="*40 + "\n")

        logger.logkv("test_psnr", total_psnr)
        logger.logkv("test_ssim", total_ssim)
        logger.logkv("test_nmse", total_nmse)
        logger.logkv("test_time_per_img", total_time)
        return total_psnr

    return 0.0



def CMR_GTINPUT_sampling_major_vote_func(batch_size, diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, vote_num=4):
    ddp_model.eval()
    batch_size = batch_size
    major_vote_number = vote_num
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    loader_iter = iter(loader)
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = len(loader)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]

    with torch.no_grad():
        for round_index in range(n_rounds):
            print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
            data_ = next(loader_iter)
            gt_mask = data_["GT"]
            condition_on = {"conditioned_image": data_["input"]}
            name = data_["ipath"]
            
            name = [n.split("/")[-5]+"_"+n.split("/")[-1].split(".")[0] for n in name]
            condition_on = condition_on["conditioned_image"]

            for index, (gt_im, out_im) in enumerate(zip(gt_mask, condition_on)):
                gt_im = gt_im.cpu().squeeze(1).detach().numpy()
                out_im = out_im.cpu().squeeze(1).detach().numpy()
                psnr_ = psnr(gt_im, out_im)
                mse_ = mse(gt_im, out_im)
                nmse_ = nmse(gt_im, out_im)
                ssim_ = ssim(gt_im, out_im)
                psnr_list.append(psnr_)
                mse_list.append(mse_)
                nmse_list.append(nmse_)
                ssim_list.append(ssim_)

                logger.info(
                    f"{name[index]} psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")

    my_length = len(psnr_list)
    length_of_data = torch.tensor(len(psnr_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    ssim_list = [i.item() for i in ssim_list]
    nmse_list = [i.item() for i in nmse_list]

    psnr_tensor = torch.tensor(psnr_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    ssim_tensor = torch.tensor(ssim_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    mse_tensor = torch.tensor(mse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    nmse_tensor = torch.tensor(nmse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    gathered_psnr = [torch.ones_like(psnr_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_ssim = [torch.ones_like(ssim_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_mse = [torch.ones_like(mse_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_nmse = [torch.ones_like(nmse_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_psnr, psnr_tensor)
    dist.all_gather(gathered_ssim, ssim_tensor)
    dist.all_gather(gathered_mse, mse_tensor)
    dist.all_gather(gathered_nmse, nmse_tensor)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    gathered_psnr = torch.cat(gathered_psnr)
    # gathered_psnr = gathered_psnr[gathered_psnr != -1]
    logger.info(f"mean psnr {gathered_psnr.mean()}")

    gathered_ssim = torch.cat(gathered_ssim)
    # gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean ssim {gathered_ssim.mean()}")
    gathered_mse = torch.cat(gathered_mse)
    # gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean mse {gathered_mse.mean()}")
    gathered_nmse = torch.cat(gathered_nmse)
    # gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean nmse {gathered_nmse.mean()}")

    dist.barrier()
    return gathered_psnr.mean().item(), gathered_ssim.mean().item(), gathered_nmse.mean().item()


def sampling_major_vote_func(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, step, n_rounds=3):
    ddp_model.eval()
    batch_size = 1
    major_vote_number = 9
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_iter = iter(loader)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    with torch.no_grad():
        for round_index in tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            data_ = next(loader_iter)
            gt_mask = data_["GT"]
            condition_on = {"conditioned_image": data_["input"]}
            name = data_["ipath"]
            # gt_mask, condition_on, name = next(loader_iter)
            # set_random_seed_for_iterations(step + int(name[0].split("_")[1]))
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())

            for i in range(gt_mask.shape[0]):
                gt_img = Image.fromarray(gt_mask[i][0].detach().cpu().numpy().astype('uint8'))
                gt_img.putpalette(cityspallete)
                gt_img.save(
                    os.path.join(output_folder, f"{name[i]}_gt_palette.png"))
                gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                gt_img.save(
                    os.path.join(output_folder, f"{name[i]}_gt.png"))

            for i in range(condition_on.shape[0]):
                denorm_condition_on = denormalize(condition_on.clone(), mean=dataset.mean, std=dataset.std)
                tvu.save_image(
                    denorm_condition_on[i,] / 255.,
                    os.path.join(output_folder, f"{name[i]}_condition_on.png")
                )

            if dataset is None:
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []
                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for i in range(major_vote_number):
                    x_list = []

                    for index in range(math.ceil(patches.shape[0] / 4)):
                        model_kwargs = {"conditioned_image": patches[index * 4: min((index + 1) * 4, patches.shape[0])]}
                        x = diffusion_model.p_sample_loop(
                                ddp_model,
                                (model_kwargs["conditioned_image"].shape[0], gt_mask.shape[1], model_kwargs["conditioned_image"].shape[2], model_kwargs["conditioned_image"].shape[3]),
                                progress=True,
                                clip_denoised=clip_denoised,
                                model_kwargs=model_kwargs
                            )

                        x_list.append(x)
                    out = torch.cat(x_list)

                    output = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    idx_sum = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index, (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride

                        idx_sum[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += 1

                        output[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y, :min(x + kernel_size, H) - x].cpu().data.numpy()

                    output = output / idx_sum
                    major_vote_list.append(output)

                x = torch.cat(major_vote_list)

            else:
                model_kwargs = {
                    "conditioned_image": torch.cat([former_frame_for_feature_extraction] * major_vote_number)}

                x = diffusion_model.p_sample_loop(
                    ddp_model,
                    (major_vote_number, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                     former_frame_for_feature_extraction.shape[3]),
                    progress=True,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )

            x = (x + 1.0) / 2.0

            if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            x = torch.clamp(x, 0.0, 1.0)

            # major vote result
            x = x.mean(dim=0, keepdim=True).round()

            for i in range(x.shape[0]):
                # save as outer training ids
                # current_output = x[i][0] + 1
                # current_output[current_output == current_output.max()] = 0
                out_img = Image.fromarray(x[i][0].detach().cpu().numpy().astype('uint8'))
                out_img.putpalette(cityspallete)
                out_img.save(
                    os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                out_img = Image.fromarray((x[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{name[i]}_model_output.png"))

            for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):

                f1, miou, wcov, fbound = calculate_metrics(out_im[0], gt_im[0])
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)

                logger.info(
                    f"{name[index]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}, WCov {wcov_list[-1]}, boundF {fbound_list[-1]}")

    my_length = len(miou_list)
    length_of_data = torch.tensor(len(miou_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    wcov_tensor = torch.tensor(wcov_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    boundf_tensor = torch.tensor(fbound_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    gathered_miou = torch.cat(gathered_miou)
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")
    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean()}")
    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean()}")

    dist.barrier()
    return gathered_miou.mean().item()
