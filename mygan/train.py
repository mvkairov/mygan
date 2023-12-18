import PIL
import wandb
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def get_grad_norm(model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()    
    for i in arr:
        temp = (((i - arr.min()) * diff) / diff_arr) + t_min
        norm_arr.append(temp.unsqueeze(0))
    return norm_arr


def evaluate(generator, discriminator, dataloader, fixed_noise, fid_metric, ssim_metric):
    generator.eval()
    discriminator.eval()
    last_idx = 0
    real_imgs = []
    constructed_imgs = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            real = data[0].to('cuda')
            b_size = real.size(0)
            samples = generator(fixed_noise[last_idx:last_idx + b_size, ...])

            real_imgs.append(real.detach())
            constructed_imgs.append(samples.detach())
            last_idx += b_size

    print(len(real_imgs), real_imgs[0].shape)
    print(len(constructed_imgs), constructed_imgs[0].shape)

    real_imgs = torch.cat(real_imgs)
    real_imgs = normalize(real_imgs, 0, 1)
    real_imgs = torch.cat(real_imgs)

    constructed_imgs = torch.cat(constructed_imgs)
    constructed_imgs = normalize(constructed_imgs, 0, 1)
    constructed_imgs = torch.cat(constructed_imgs)

    print(real_imgs.shape, constructed_imgs.shape)

    fid = fid_metric.compute_metric(real_imgs.flatten(1), constructed_imgs.flatten(1)).cpu().numpy(),
    ssim = ssim_metric(real_imgs, constructed_imgs).item()
    return fid, ssim


def train(generator, discriminator, gen_optim, dis_optim, loader, n_epochs, log_step=50, start_step=0):
    cur_step = 0
    loss_fn = nn.BCELoss()

    for _ in range(n_epochs):
        for i, (data, _) in enumerate(tqdm(loader), 0):
            discriminator.zero_grad()
            real = data[0].to('cuda')
            b_size = real.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device='cuda')
            output = discriminator(real).view(-1)
            errD_real = loss_fn(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, generator.nz, 1, 1, device='cuda')
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = loss_fn(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            dis_optim.step()

            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake).view(-1)
            errG = loss_fn(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            gen_optim.step()
            
            if cur_step % 100 == 0:
                wandb.log({
                    'gen_loss': errG.item(),
                    'dis_loss': errD.item(),
                    'gen_gradn': get_grad_norm(generator),
                    'dis_gradn': get_grad_norm(discriminator)
                }, step=cur_step)
                print(f'step {cur_step}:')
                print(f'Loss: G={errG.item():.5f}, D={errD.item():.5f}')
                print(f'D(x)={D_x}, D(G(z))={D_G_z1:.5f} / {D_G_z2:.5f}\n')

            if i == len(loader) - 1:
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                }, 'last_model.pt')

                img = fake.detach().cpu().numpy()
                img = (normalize(img.reshape(img.shape[1], img.shape[2], img.shape[0]), 0, 1) * 255).astype('uint8')
                wandb.log({
                    'img_res': PIL.Image.fromarray(img, 'RGB')
                }, step=cur_step)
            
            cur_step += 1
