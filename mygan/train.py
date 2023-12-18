import PIL
import wandb
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

from piq import FID, SSIMLoss


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()    
    for i in arr:
        temp = (((i - arr.min()) * diff) / diff_arr) + t_min
        norm_arr.append(temp.unsqueeze(0))
    return norm_arr


@torch.no_grad()
def evaluate(generator, loader):
    generator.eval()
    noise = torch.randn(len(loader.dataset), generator.nz, 1, 1, device='cuda')

    last_i = 0
    real, fake = [data[0].to('cuda').detach() for data in loader], []
    for i in range(len(loader)):
        b_size = real[i].size(0)
        fake.append(generator(noise[last_i:last_i + b_size, ...]).detach())
        last_i += b_size

    real = torch.cat(normalize(torch.cat(real), 0, 1))
    fake = torch.cat(normalize(torch.cat(fake), 0, 1))

    fid_metric = FID().to('cuda')
    ssim_metric = SSIMLoss(data_range=1.0).to('cuda')
    fid = fid_metric.compute_metric(real.flatten(1), fake.flatten(1))
    ssim = ssim_metric(real, fake)
    return fid.cpu().numpy(), ssim.item()


def train(generator, discriminator, gen_optim, dis_optim, loader, n_epochs):
    cur_step = 0
    loss_fn = nn.BCELoss()

    for epoch in range(n_epochs):
        for i, (data, _) in enumerate(tqdm(loader), 0):
            discriminator.zero_grad()
            real = data.to('cuda')
            b_size = real.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device='cuda')
            output = discriminator(real).view(-1)
            errD_real = loss_fn(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, generator.nz, 1, 1, device='cuda')
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = loss_fn(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            dis_optim.step()

            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake).view(-1)
            errG = loss_fn(output, label)
            errG.backward()
            gen_optim.step()
            
            if i == len(loader) - 1:
                wandb.log({
                    'gen_loss': errG.item(),
                    'dis_loss': errD.item(),
                }, step=cur_step)
                
                print(f'EPOCH {epoch}. Loss: G={errG.item():.5f}, D={errD.item():.5f}\n')
                
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                }, 'last_model.pt')
                
                if epoch % 25 == 0 and epoch > 0:
                    fid, ssim = evaluate(generator, loader)
                    wandb.log({
                        'FID': fid,
                        'SSIM': ssim
                    }, step=cur_step)
                
                
            cur_step += 1
