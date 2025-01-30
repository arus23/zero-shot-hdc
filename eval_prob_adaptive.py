import argparse
import numpy as np
import os
import os.path as osp
import time
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")


# function to transform images 
def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)

"""
Evaluate the probability of adaptive sampling.

Args:
    unet: The U-Net model.
    latent: The latent vector.
    text_embeds: The text embeddings.
    scheduler: The scheduler.
    args: The arguments.
    latent_size: The size of the latent vector (default: 64).
    all_noise: The noise samples (default: None).

Returns:
    pred_idx: The predicted index of most promising prompt.
    data: The evaluated data dictionary of all prompts.
"""
def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):

    # initialize scheduler 
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    # initialize noise 
    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    # initialize data structures to store eval results
    data = {}
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    topn = []

    start = T // max_n_samples // 2
    # start = half of T/max_samples, steps of T/max_samples
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    # algorithm iterations through n_samples per batch and n_prompts to retain
    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        
        ts = []
        noise_idxs = []
        text_embed_idxs = []

        # select a subset of timesteps to evaulate in this iteration and filter out timesteps previously evaluated
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]

        # loop over remaining prompts
        for prompt_i in remaining_prmpt_idxs:

            # generate the noisy sample for the timestep-prompt combination
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):

                # repeat current timestep for n trials
                ts.extend([t] * args.n_trials)

                # range of indices for noise samples
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))

                # repeat current prompt and timestep for n trails
                text_embed_idxs.extend([prompt_i] * args.n_trials)

        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
        
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:

            # select data specific to current prompt and select timesteps
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            
            # store prompt and timestep and pred errors
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # calculate mean prediction errors per prompt, negate and append to errors
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]

        # select prompts with lowest errors and update remaining prompts list
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]
        if len(remaining_prmpt_idxs) >= 5:
                topn = remaining_prmpt_idxs
        # print(f"topn: {topn}")

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data, topn

"""
Evaluate the error between the predicted noise and the ground truth noise.

Args:
    unet: The U-Net model.
    scheduler: The scheduler.
    latent: The latent vector.
    all_noise: The noise samples.
    ts: The timesteps.
    noise_idxs: The indices of noise samples.
    text_embeds: The text embeddings.
    text_embed_idxs: The indices of text embeddings.
    batch_size: The batch size (default: 32).
    dtype: The data type (default: 'float32').
    loss: The loss function (default: 'l2').

Returns:
    pred_errors: The predicted errors.
"""
def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0

    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):

            # timesteps for the current batch
            batch_ts = torch.tensor(ts[idx: idx + batch_size])

            # noise for the current batch
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]

            # generate noise predictions 
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'cifar100', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    # assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    print("Setting up dataset and prompts")
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    print("\nLoading pretrained models")

    # get the components from models.py
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # load noise from noise path
    print("\nLoading noise")
    if args.noise_path is not None:
        assert not args.zero_noise
        # load noise tensor to device
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    # tokenize text prompts
    print("\nTokenizing text prompts and creating embeddings")    
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    # create a list of text embeddings
    embeddings = []
    with torch.inference_mode():
        # In batches of 100s, encode the text prompts
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    # subset of dataset to evaluate
    print("\nLoad subset to evaluate")
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))

    # Split the list of indices into subsets for n_workers
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    # Define corectly predicted labels and total labels
    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0

    # Create a progress bar and evaluate the predictions
    pbar = tqdm.tqdm(idxs_to_eval)

    print("\nEvaluating predictions: ")
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')

        # save eval results for the current data point    
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['pred'] == data['label'])
                total += 1
            continue
        image, label = target_dataset[i]

        # disable gradient computation for eval
        with torch.no_grad():
            # convert img to tensor
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            # encode image using vae model, scale latent vector
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215

        # Evaluateprobability of different classes and compute the prediction errors.
        start_time = time.time()
        pred_idx, pred_errors, topn = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        topn_preds = [prompts_df.classidx[idx] for idx in topn]
        end_time = time.time()
        inf_time = end_time - start_time
        print(f"\nInference time: {inf_time:.2f} seconds \n\n\n")
        torch.save(dict(errors=pred_errors, pred=pred, label=label, inf_time=inf_time, topn=topn_preds), fname)
        print(f"prediction: {prompts_df.classname[pred_idx]}")
        if pred == label:
            correct += 1
        total += 1


if __name__ == '__main__':
    main()
