import pickle

import cv2
import torch
import os
import numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.utils import save_image


def sample_images(checkpoint_path,model_pt, n):
    checkpoint = os.path.join(checkpoint_path, model_pt)
    gen = torch.load(checkpoint, map_location='cuda')
    samples = gen.sample(n=n, with_grad=False).cpu()
    return samples

def create_images_folders(checkpoint_path, model_name, folder_path):
    samples = sample_images(checkpoint_path, model_name, 100)
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(samples):

        save_image(img, os.path.join(folder_path, str(i)+'.jpg'))

def create_images_for_all_models():
    checkpoints_path = "C:\\BarDir\\DL_class\\hw4-pro\\project\\checkpoints\\finales"
    gen_images_path = "C:\BarDir\DL_class\hw4-pro\project\gen_images"
    dcgann = "vinilla_gan570.pt"
    wgan = 'wgan1.pt'
    spgan = 'spectral_gan1920.pt'
    wgan_sp = 'wgan_plus_spectral_gan990.pt'
    wgan_gp = 'wgan_gp990.pt'
    wgan_sp_gp = 'wgan_sn_gp990.pt'
    for model in [dcgann, wgan, spgan, wgan_sp, wgan_gp, wgan_sp_gp]:
        create_images_folders(checkpoints_path, model, folder_path=os.path.join(gen_images_path,model[:-3]))

def get_fid_score(real_imgs_folder, gen_images_folder):
    return calculate_fid_given_paths([real_imgs_folder, gen_images_folder], batch_size=5, device='cuda',dims=128)

def create_fid_scores_dict():
    fid_scores = {'DCGAN': -1, 'WGAN': -1, 'SP-GAN': -1, 'WGAN+SP': -1}
    gen_images_path = "C:\BarDir\DL_class\hw4-pro\project\gen_images"
    real_images_path = "C:\\BarDir\\DL_class\\hw4-pro\\project\\gen_images\\real_images"
    dcgann = "vinilla_gan570"
    wgan = 'wgan1'
    spgan = 'spectral_gan1920'
    wgan_sp = 'wgan_plus_spectral_gan990'
    wgan_gp = 'wgan_gp990'
    wgan_sp_gp = 'wgan_sn_gp990'

    fid_scores['DCGAN'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, dcgann), real_imgs_folder=real_images_path)
    fid_scores['WGAN'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, wgan), real_imgs_folder=real_images_path)
    fid_scores['SP-GAN'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, spgan), real_imgs_folder=real_images_path)
    fid_scores['WGAN+SP'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, wgan_sp), real_imgs_folder=real_images_path)
    fid_scores['WGAN+GP'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, wgan_gp), real_imgs_folder=real_images_path)
    fid_scores['WGAN+SN+GP'] = get_fid_score(gen_images_folder=os.path.join(gen_images_path, wgan_sp_gp), real_imgs_folder=real_images_path)

    pickle.dump(fid_scores, open('fid_scores','wb'))
if __name__ == '__main__':
    create_images_for_all_models()
    create_fid_scores_dict()