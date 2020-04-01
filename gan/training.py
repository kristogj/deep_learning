import torch
import logging
from utils import get_device, show
import torchvision.utils as vutils
import os
import numpy as np
import pandas as pd
from plotting import generate_lineplot


def get_correct_count(prob):
    res = (prob > 0.5).float()
    return res.sum().item()


def freeze_weights(model, switch):
    for param in model.parameters():
        param.requires_grad = switch


def train(generator, discriminator, criterion, optim_g, optim_d, dataloader, fixed_noise, config):
    try:
        os.makedirs(config['res_path'])
    except FileExistsError:
        pass
    # Lists to keep track of progress
    img_list = []
    g_losses = []
    d_losses = []
    g_accs = []
    iters = 0
    device = get_device()
    logging.info("Starting Training Loop...")

    # Real/Fake convention
    epochs = config["epochs"]
    for epoch in range(epochs):
        n_batches = 0
        g_acc = []
        g_loss = 0.0
        d_loss = 0.0
        for i, (data, _) in enumerate(dataloader, 0):
            g_correct_preds = 0
            # Part I: Update D network - maximize log(D(x)) + log(1-D(G(z))
            discriminator.zero_grad()

            # Put data to device
            data = data.to(device)
            batch_size = data.size(0)
            one = torch.tensor(-1, dtype=torch.float).to(device)
            # 6. Soft and Noisy Labels
            real_label = 0.5 * torch.rand((batch_size,), device=device) + 0.7
            fake_label = 0.3 * torch.rand((batch_size,), device=device)

            # Forward pass (real) data batch through discriminator

            output_r = discriminator(data).view(-1)
            g_correct_preds += get_correct_count(output_r)
            if config["wasserstein"]:
                errD_real = torch.mean(output_r)
            else:
                errD_real = criterion(output_r, real_label)
                errD_real.backward()
            D_x = output_r.mean().item()

            # Train with all-fake batch
            # Generate new batch of latent vectors
            noise = torch.randn((batch_size, config["nz"], 1, 1), device=device)
            fake_images = generator(noise)

            # Classify all fake images
            output_f = discriminator(fake_images.detach()).view(-1)
            g_correct_preds += (config["batch_size"] - get_correct_count(output_f))
            if config["wasserstein"]:
                errD_fake = torch.mean(output_f)
            else:
                errD_fake = criterion(output_f, fake_label)
                errD_fake.backward()
            D_G_z1 = output_f.mean().item()

            # Add gradients from the real images and the fake images
            if config["wasserstein"]:
                errD = - errD_real + errD_fake
            else:
                errD = errD_real + errD_fake

            # Update Discriminator
            optim_d.step()

            # Part II: Update G Network: Maximize log(D(G(z))
            generator.zero_grad()
            # discriminator.eval()
            output = discriminator(fake_images).view(-1)
            if config["wasserstein"]:
                # following the implementation of Wasserstein loss found in
                # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
                errG = - torch.mean(output)
                errG.backward()
            else:
                errG = criterion(output, real_label)
                errG.backward()
            D_G_z2 = output.mean().item()
            optim_g.step()
            # discriminator.train()

            if i % 50 == 0:
                logging.info(
                    "[{}/{}][{}/{}] Loss_D: {}, Loss_G: {}, D(x): {}, D(G(z)): {}/{}".format(epoch, epochs,
                                                                                             i, len(dataloader),
                                                                                             errD.item(), errG.item(),
                                                                                             D_x,
                                                                                             D_G_z1, D_G_z2))

            g_loss += errG.item()
            d_loss += errD.item()
            g_acc.append(g_correct_preds / (config['batch_size'] * 2))
            iters += 1
            n_batches += 1

        g_accs.append(np.average(g_acc))
        g_losses.append(g_loss / n_batches)
        d_losses.append(d_loss / n_batches)
        logging.info("===========================================")
        logging.info("Discriminator accuracy at epoch {}: {}".format(epoch, g_accs[-1]))
        logging.info("Generator loss at epoch {}: {}".format(epoch, g_losses[-1]))
        logging.info("Disriminator loss at epoch {}: {}".format(epoch, d_losses[-1]))
        logging.info("===========================================")

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        show(vutils.make_grid(fake, padding=2, normalize=True), epoch, config)

        # Generate plot
        logging.info("Generating plot to loss_graph.png")
        generate_lineplot((g_losses, d_losses), "Loss", config["res_path"] + "loss_graph.png")

    columns = ["Epoch", "Generator Loss", "Discriminator Loss", "Generator Accuracy"]
    df = pd.DataFrame(columns=columns)

    for i in range(epochs):
        df.loc[len(df)] = [i, g_losses[i], d_losses[i], g_accs[i]]
    logging.info("Writing results to res.csv")
    df.to_csv(config["res_path"] + "res.csv")


def train_with_sketch(generator, discriminator, criterion, optim_g, optim_d, dataloader, test_imgs, config):
    # Lists to keep track of progress
    img_list = []
    g_losses = []
    d_losses = []
    g_accs = []
    iters = 0
    device = get_device()
    logging.info("Starting Training Loop...")

    epochs = config["epochs"]
    for epoch in range(epochs):
        n_batches = 0
        g_acc = []
        g_loss = 0.0
        d_loss = 0.0
        for i, (sketch_img, real_img) in enumerate(dataloader, 0):
            g_correct_preds = 0

            # Part I: Update D network - maximize log(D(x)) + log(1-D(G(z))
            discriminator.zero_grad()

            # Put data to device
            sketch_img = sketch_img.to(device)
            real_img = real_img.to(device)
            batch_size = real_img.size(0)
            mone = torch.tensor(-1, dtype=torch.float).to(device)
            # 6. Soft and Noisy Labels
            real_label = 0.5 * torch.rand((batch_size,), device=device) + 0.7
            fake_label = 0.3 * torch.rand((batch_size,), device=device)

            # Forward pass (real) data batch through discriminator
            output_r = discriminator(real_img).view(-1)
            g_correct_preds += get_correct_count(output_r)

            if config["wasserstein"]:
                errD_real = torch.mean(output_r)
            else:
                errD_real = criterion(output_r, real_label)
                errD_real.backward()
            D_x = output_r.mean().item()

            # Train with all-fake batch
            fake_images = generator(sketch_img)

            # Classify all fake images
            output_f = discriminator(fake_images.detach()).view(-1)
            g_correct_preds += (config["batch_size"] - get_correct_count(output_f))

            if config["wasserstein"]:
                errD_fake = torch.mean(output_f)
            else:
                errD_fake = criterion(output_f, fake_label)
                errD_fake.backward()
            D_G_z1 = output_f.mean().item()

            # Add gradients from the real images and the fake images
            if config["wasserstein"]:
                errD = -errD_real + errD_fake
                errD.backward()
            else:
                errD = errD_real + errD_fake

            # Update Discriminator
            # optim_d.step()
            if i % 5 == 0:
                optim_d.step()

            # Clamp weights only if Wasserstein loss is used
            if config["wasserstein"]:
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # Part II: Update G Network: Maximize log(D(G(z))
            generator.zero_grad()
            freeze_weights(discriminator, False)
            output = discriminator(fake_images).view(-1)
            if config["wasserstein"]:
                # following the implementation of Wasserstein loss found in
                # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
                errG = - torch.mean(output)
                errG.backward()
            else:
                errG = criterion(output, real_label)
                errG.backward()
            D_G_z2 = output.mean().item()
            optim_g.step()
            freeze_weights(discriminator, True)

            if i % 20 == 0:
                logging.info(
                    "[{}/{}][{}/{}] Loss_D: {}, Loss_G: {}, D(x): {}, D(G(z)): {}/{}".format(epoch, epochs,
                                                                                             i, len(dataloader),
                                                                                             errD.item(), errG.item(),
                                                                                             D_x,
                                                                                             D_G_z1, D_G_z2))

            g_loss += errG.item()
            d_loss += errD.item()
            g_acc.append(g_correct_preds / (config['batch_size'] * 2))
            iters += 1
            n_batches += 1

        g_accs.append(np.average(g_acc))
        g_losses.append(g_loss / n_batches)
        d_losses.append(d_loss / n_batches)
        logging.info("===========================================")
        logging.info("Discriminator accuracy at epoch {}: {}".format(epoch, g_accs[-1]))
        logging.info("Generator loss at epoch {}: {}".format(epoch, g_losses[-1]))
        logging.info("Disriminator loss at epoch {}: {}".format(epoch, d_losses[-1]))
        logging.info("===========================================")
        # Check how the generator is doing by saving G's output on test_imgs
        with torch.no_grad():
            fake = generator(test_imgs).detach().cpu()
        show(vutils.make_grid(fake, padding=2, normalize=True), epoch, config)

        # Generate plot
        logging.info("Generating plot to loss_graph.png")
        generate_lineplot((g_losses, d_losses), "Loss", config["res_path"] + "loss_graph.png")

    columns = ["Epoch", "Generator Loss", "Discriminator Loss", "Generator Accuracy"]
    df = pd.DataFrame(columns=columns)

    for i in range(epochs):
        df.loc[len(df)] = [i, g_losses[i], d_losses[i], g_accs[i]]
    logging.info("Writing results to res.csv")
    df.to_csv(config["res_path"] + "res.csv")
