import time
import torch
import numpy as np
import sklearn.metrics as metric
from tqdm import tqdm


def train_model(mode, dataloader, val_dataloader, network, start_epoch, num_epoch, device,
                scheduler, criterion, optimizer, output_path, max_time=None):

    epoch_losses_train = []
    epoch_losses_val = []
    best_f1_weighted, best_f1_binary = 0, 0

    start = time.time()
    for epoch in range(start_epoch, num_epoch+1):
        print("epoch: %d/%d" % (epoch, num_epoch))
        ############################################################################
        # train:
        ############################################################################

        network.train()
        batch_losses = []
        for imgs, filter_img, label, _ in tqdm(dataloader):
            imgs = imgs.to(device)
            filter_img = filter_img.to(device)
            label = (label.type(torch.float32)).to(device)

            optimizer.zero_grad()  # (reset gradients)

            out = network(imgs, filter_img)

            loss = criterion(out, label)

            loss_value = loss.data.detach().cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:

            loss.backward()
            optimizer.step()  # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        print("train loss: %g" % epoch_loss)

        scheduler.step()

        ############################################################################
        # val:
        ############################################################################
        t = []
        p = []
        if True:
            network.eval()
            batch_losses = []
            for imgs, filter_img, label, _ in tqdm(val_dataloader):
                with torch.no_grad():
                    imgs = imgs.to(device)
                    filter_img = filter_img.to(device)
                    label = (label.type(torch.float32)).to(device)

                    out = network(imgs, filter_img)

                    preds = out.detach().max(dim=1)[1].cpu().numpy()
                    targets = label.detach().max(dim=1)[1].cpu().numpy()

                    t.extend(targets)
                    p.extend(preds)

                    # compute the loss:
                    loss = criterion(out, label)
                    loss_value = loss.data.cpu().numpy()
                    batch_losses.append(loss_value)

            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            print("val loss: %g" % epoch_loss)

            f1_weighted = metric.f1_score(t, p, average='weighted')
            print("f1 score: ", f1_weighted)
            f1_binary = metric.f1_score(t, p, average='binary')
            print("f1 binary: ", f1_binary)

        if f1_weighted > best_f1_weighted:
            print("############ Best Result f1_weighted ############")
            print(metric.classification_report(t, p, zero_division=0.0))
            out_filename = output_path / f'best_f1_weighted_{mode}.pth'
            torch.save(network.state_dict(), out_filename)
            best_f1_weighted = f1_weighted

        if f1_binary > best_f1_binary:
            print("############ Best Result f1_binary ############")
            print(metric.classification_report(t, p, zero_division=0.0))
            out_filename = output_path / f'best_f1_binary_{mode}.pth'
            torch.save(network.state_dict(), out_filename)
            best_f1_binary = f1_binary

        end = time.time()
        forward_time = end - start
        if max_time is not None:
            if forward_time > max_time:
                print(f"############ last=> epoch {epoch} time :{forward_time} ############")
                print(metric.classification_report(t, p, zero_division=0.0))
                out_filename = output_path / f'last_weights_{mode}.pth'
                torch.save(network.state_dict(), out_filename)
                break
