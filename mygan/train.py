import torch
from LLAMA.model import make_seq_mask, make_seq
from tqdm.notebook import tqdm
from itertools import repeat


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def evaluate(model, dataloader, loss_fn, pad_idx):
    model.eval()
    loss_sum = 0
    for tgt, _ in tqdm(dataloader, total=len(dataloader)):
        tgt = tgt.to('cuda')
        tgt_input = tgt[:, :-1]
        mask, pad_mask = make_seq_mask(tgt_input, pad_idx)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(tgt_input, mask, pad_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sum += loss.item()
    return loss_sum / len(dataloader)


def train(model, n_epochs, pad_idx, optimizer, scheduler, train_loader, val_loader, dataset, wandb_instance, steps_per_epoch):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    min_loss = 100
    cur_step = 0

    for _ in range(n_epochs):
        loss_sum = 0
        for i, (tgt, _) in enumerate(tqdm(inf_loop(train_loader), total=steps_per_epoch)):
            model.train()
            tgt = tgt.to('cuda')
            tgt_input = tgt[:, :-1]
            mask, padding_mask = make_seq_mask(tgt_input, pad_idx)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(tgt_input, mask, padding_mask)
                tgt_out = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            cur_step += 1

            if i > 0 and (i + 1) % (steps_per_epoch // 10) == 0:
                wandb_instance.log({
                    'train_loss': loss_sum / (i + 1),
                    'lr': scheduler.get_last_lr()[0]
                }, step=cur_step)
                print(f'step {cur_step}; train_loss {(loss_sum / (i + 1)):.3f}')

            if i == steps_per_epoch - 1:
                val_loss = evaluate(model, val_loader, loss_fn, pad_idx)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                text = dataset.ids2text(make_seq(model, dataset.sp_model, pad_idx))
                wandb_instance.log({
                    'train_loss': loss_sum / steps_per_epoch,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0],
                }, step=cur_step)
                print(f'train_loss {(loss_sum / steps_per_epoch):.3f}; val_loss: {val_loss:.3f}')
                text = dataset.ids2text(make_seq(model, dataset.sp_model, pad_idx))
                print(text, '\n')
                break
