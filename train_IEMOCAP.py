import os
import sys

import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from model import (
    MaskedNLLLoss,
    FocalLoss,
    LSTMModel2,
    AcousticGenerator,
    AcousticDiscriminator,
    TextGenerator,
    TextDiscriminator,
    VisualGenerator,
    VisualDiscriminator,
    GAN_FFN,
)
from dataloader import IEMOCAPDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import random

import warnings

warnings.filterwarnings("ignore")

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 23333
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(
    path, batch_size=32, valid=0.2, num_workers=0, pin_memory=False
):
    trainset = IEMOCAPDataset(path=path)
    testset = IEMOCAPDataset(path=path, train=False)

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def train_or_eval_model(
    model, loss_function, dataloader, epoch, optimizer=None, train=False
):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            lambda1 = lambda epoch: 0.98**epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda1
            )  # 打印学习率，防止过拟合
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = (
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        )
        seq_lengths = [
            (umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))
        ]

        # print("textf.shape = ", textf.shape) # (seq_len, 32, 100)
        # print("visuf.shape = ", visuf.shape) # (seq_len, 32, 512)
        # print("acouf.shape = ", acouf.shape) # (seq_len, 32, 100)
        # print("qmask.shape = ", qmask.shape) # (seq_len, 32, 2) 可以看出是哪一个人说的
        # print("umask.shape = ", umask.shape) # (32, seq_len) 可以看出每个句子是否有效 还是为0填充
        # print("label.shape = ", label.shape) # (32, seq_len)
        # print(label)
        # sys.exit()

        log_prob, alpha, alpha_f, alpha_b = model(acouf, visuf, textf)
        # log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf), dim=-1), qmask, umask)
        # log_prob, alpha, alpha_f, alpha_b, hidden = model(textf, acouf, visuf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            scheduler.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float("nan"), float("nan"), [], [], [], float("nan"), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(
        f1_score(labels, preds, sample_weight=masks, average="weighted") * 100, 2
    )
    return (
        avg_loss,
        avg_accuracy,
        labels,
        preds,
        masks,
        avg_fscore,
        [alphas, alphas_f, alphas_b, vids],
    )


def train_GAN(
    acoustic_generator: AcousticGenerator,
    visual_generator: VisualGenerator,
    text_generator: TextGenerator,
    acoustic_discriminator: AcousticDiscriminator,
    visual_discriminator: VisualDiscriminator,
    text_discriminator: TextDiscriminator,
    epochs=1,
    batch_size=32,
    lr=0.002,
    b1=0.6,
    b2=0.999,
):
    # ----------
    #  Training
    # ----------
    print("=" * 15, "start training GAN", "=" * 15)

    # Optimizers
    optimizer_acoustic_G = torch.optim.Adam(
        acoustic_generator.parameters(), lr=lr, betas=(b1, b2)
    )
    optimizer_acoustic_D = torch.optim.Adam(
        acoustic_discriminator.parameters(), lr=lr / 2, betas=(b1, b2)
    )
    optimizer_visual_G = torch.optim.Adam(
        visual_generator.parameters(), lr=lr, betas=(b1, b2)
    )
    optimizer_visual_D = torch.optim.Adam(
        visual_discriminator.parameters(), lr=lr / 2, betas=(b1, b2)
    )
    optimizer_text_G = torch.optim.Adam(
        text_generator.parameters(), lr=lr, betas=(b1, b2)
    )
    optimizer_text_D = torch.optim.Adam(
        text_discriminator.parameters(), lr=lr / 2, betas=(b1, b2)
    )

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()  # 二元交叉熵

    # Dataloaders
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
        "./IEMOCAP_features/IEMOCAP_features.pkl", batch_size=batch_size, valid=0.2
    )

    # Start epochs
    for epoch in range(epochs):
        print("=" * 15, "start Epoch : ", epoch + 1, "=" * 15)
        for i, data in enumerate(train_loader):
            loss = []

            textf, visuf, acouf, qmask, umask, label = (
                [d.to(device) for d in data[:-1]] if device else data[:-1]
            )

            batch_size = textf.size(1)
            seq_len = textf.size(0)

            # Adversarial ground truths
            valid = Variable(
                FloatTensor(seq_len, batch_size, 1).fill_(1.0), requires_grad=False
            ).to(device)
            fake = Variable(
                FloatTensor(seq_len, batch_size, 1).fill_(0.0), requires_grad=False
            ).to(device)

            # Configure input
            real_text = Variable(textf.type(FloatTensor))
            real_visual = Variable(visuf.type(FloatTensor))
            real_acoustic = Variable(acouf.type(FloatTensor))
            label = Variable(label.type(LongTensor))

            # -----------------
            #  Train AcousticGenerator
            # -----------------
            print("=" * 15, "Train AcousticGenerator", "=" * 15)

            optimizer_acoustic_G.zero_grad()

            # Generate a batch of fusions
            acoustic_fusion = acoustic_generator(real_acoustic)

            # Loss measures generator's ability to fool the discriminator
            visual_prob = visual_discriminator(acoustic_fusion)
            text_prob = text_discriminator(acoustic_fusion)
            # print("visual_prob.shape = ", visual_prob.shape) # torch.Size([94, 32, 1])

            g_loss = 0.5 * (
                adversarial_loss(visual_prob, valid)
                + adversarial_loss(text_prob, valid)
            )
            loss.append(g_loss)

            g_loss.backward()
            optimizer_acoustic_G.step()

            # ---------------------
            #  Train VisualDiscriminator
            # ---------------------
            print("-" * 8, "Train VisualDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_visual_prob = visual_discriminator(real_visual)
            d_real_loss = adversarial_loss(real_visual_prob, valid)

            # Loss for fake images
            fake_visual_prob = visual_discriminator(acoustic_fusion.detach())
            d_fake_loss = adversarial_loss(fake_visual_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)

            d_loss.backward()
            optimizer_visual_D.step()

            # ---------------------
            #  Train TextDiscriminator
            # ---------------------
            print("-" * 8, "Train TextDiscriminator", "-" * 8)

            optimizer_text_D.zero_grad()

            # Loss for real images
            real_text_prob = text_discriminator(real_text)
            d_real_loss = adversarial_loss(real_text_prob, valid)

            # Loss for fake images
            fake_text_prob = text_discriminator(acoustic_fusion.detach())
            d_fake_loss = adversarial_loss(fake_text_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)

            d_loss.backward()
            optimizer_text_D.step()

            print(f"Acoustic G D D loss : {loss[0]}, {loss[1]}, {loss[2]}")

            # -----------------
            #  Train VisualGenerator
            # -----------------
            print("=" * 15, "Train VisualGenerator", "=" * 15)

            optimizer_visual_G.zero_grad()

            # Generate a batch of images
            visual_fusion = visual_generator(real_visual)

            # Loss measures generator's ability to fool the discriminator
            acoustic_prob = acoustic_discriminator(visual_fusion)
            text_prob = text_discriminator(visual_fusion)
            # print("visual_prob.shape = ", visual_prob.shape) # torch.Size([94, 32, 1])

            g_loss = 0.5 * (
                adversarial_loss(acoustic_prob, valid)
                + adversarial_loss(text_prob, valid)
            )
            loss.append(g_loss)

            g_loss.backward()
            optimizer_visual_G.step()

            # ---------------------
            #  Train AcousticDiscriminator
            # ---------------------
            print("-" * 8, "Train AcousticDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_acoustic_prob = acoustic_discriminator(real_acoustic)
            d_real_loss = adversarial_loss(real_acoustic_prob, valid)

            # Loss for fake images
            fake_acoustic_prob = acoustic_discriminator(visual_fusion.detach())
            d_fake_loss = adversarial_loss(fake_acoustic_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)

            d_loss.backward()
            optimizer_acoustic_D.step()

            # ---------------------
            #  Train TextDiscriminator
            # ---------------------
            print("-" * 8, "Train TextDiscriminator", "-" * 8)

            optimizer_text_D.zero_grad()

            # Loss for real images
            real_text_prob = text_discriminator(real_text)
            d_real_loss = adversarial_loss(real_text_prob, valid)

            # Loss for fake images
            fake_text_prob = text_discriminator(visual_fusion.detach())
            d_fake_loss = adversarial_loss(fake_text_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)
            print(f"Visual G D D loss : {loss[3]}, {loss[4]}, {loss[5]}")

            d_loss.backward()
            optimizer_text_D.step()

            # -----------------
            #  Train TextGenerator
            # -----------------
            print("=" * 15, "Train TextGenerator", "=" * 15)

            optimizer_text_G.zero_grad()

            # Generate a batch of images
            text_fusion = text_generator(real_text)

            # Loss measures generator's ability to fool the discriminator
            acoustic_prob = acoustic_discriminator(text_fusion)
            visual_prob = visual_discriminator(text_fusion)
            # print("text_prob.shape = ", text_prob.shape) # torch.Size([94, 32, 1])

            g_loss = 0.5 * (
                adversarial_loss(acoustic_prob, valid)
                + adversarial_loss(visual_prob, valid)
            )
            loss.append(g_loss)

            g_loss.backward()
            optimizer_text_G.step()

            # ---------------------
            #  Train AcousticDiscriminator
            # ---------------------
            print("-" * 8, "Train AcousticDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_acoustic_prob = acoustic_discriminator(real_acoustic)
            d_real_loss = adversarial_loss(real_acoustic_prob, valid)

            # Loss for fake images
            fake_acoustic_prob = acoustic_discriminator(text_fusion.detach())
            d_fake_loss = adversarial_loss(fake_acoustic_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)
            print(f"G D D loss : {loss[3]}, {loss[4]}, {loss[5]}")
            d_loss.backward()
            optimizer_acoustic_D.step()

            # ---------------------
            #  Train VisualDiscriminator
            # ---------------------
            print("-" * 8, "Train VisualDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_visual_prob = visual_discriminator(real_visual)
            d_real_loss = adversarial_loss(real_visual_prob, valid)

            # Loss for fake images
            fake_visual_prob = visual_discriminator(text_fusion.detach())
            d_fake_loss = adversarial_loss(fake_visual_prob, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            loss.append(d_loss)
            print(f"Text G D D loss : {loss[6]}, {loss[7]}, {loss[8]}")

            d_loss.backward()
            optimizer_visual_D.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="does not use GPU"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0005, metavar="LR", help="learning rate"
    )
    parser.add_argument(
        "--l2", type=float, default=0.007, metavar="L2", help="L2 regularization weight"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.6, metavar="dropout", help="dropout rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="BS", help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, metavar="E", help="number of epochs"
    )
    parser.add_argument(
        "--class-weight", action="store_true", default=True, help="use class weight"
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        default=False,
        help="use attention on top of lstm",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enables tensorboard log",
    )
    parser.add_argument(
        "--use-trained-model",
        action="store_true",
        default=False,
        help="Use trained model",
    )
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print("Running on GPU")
    else:
        print("Running on CPU")

    path = "./tensorboard"

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(path)

    batch_size = args.batch_size
    cuda = args.cuda
    n_epochs = args.epochs
    dropout = args.dropout

    n_classes = 6
    D_m = 200
    D_e = 30
    D_h = 100

    use_trained_model = args.use_trained_model
    models_name = [
        "acoustic_generator",
        "acoustic_discriminator",
        "visual_generator",
        "visual_discriminator",
        "text_generator",
        "text_discriminator",
    ]

    if use_trained_model == True:
        acoustic_generator = torch.load("acoustic_generator.pth").eval()
        acoustic_discriminator = torch.load("acoustic_discriminator.pth").eval()
        visual_generator = torch.load("visual_generator.pth").eval()
        visual_discriminator = torch.load("visual_discriminator.pth").eval()
        text_generator = torch.load("text_generator.pth").eval()
        text_discriminator = torch.load("text_discriminator.pth").eval()
        print("=" * 15, "loaded trained GAN", "=" * 15)

    else:
        # create GAN components
        acoustic_generator = AcousticGenerator(D_h, dropout=0.2)
        acoustic_discriminator = AcousticDiscriminator(D_h, dropout=0.2)
        visual_generator = VisualGenerator(D_h, dropout=0.2)
        visual_discriminator = VisualDiscriminator(D_h, dropout=0.2)
        text_generator = TextGenerator(D_h, dropout=0.2)
        text_discriminator = TextDiscriminator(D_h, dropout=0.2)

        if cuda:
            acoustic_generator = nn.DataParallel(acoustic_generator).cuda()
            acoustic_discriminator = nn.DataParallel(acoustic_discriminator).cuda()
            visual_generator = nn.DataParallel(visual_generator).cuda()
            visual_discriminator = nn.DataParallel(visual_discriminator).cuda()
            text_generator = nn.DataParallel(text_generator).cuda()
            text_discriminator = nn.DataParallel(text_discriminator).cuda()

        train_GAN(
            acoustic_generator,
            visual_generator,
            text_generator,
            acoustic_discriminator,
            visual_discriminator,
            text_discriminator,
            epochs=50,
            batch_size=32,
            lr=0.0005,
            b1=0.6,
            b2=0.6,
        )

        # save model parameters
        models = [
            acoustic_generator,
            acoustic_discriminator,
            visual_generator,
            visual_discriminator,
            text_generator,
            text_discriminator,
        ]

        for id, model in enumerate(models):
            path = models_name[id] + ".pth"
            torch.save(model, path)

        # change mode to eval
        for model in models:
            model.eval()

        print("=" * 15, "finished training GAN", "=" * 15)

    model = GAN_FFN(
        acoustic_generator,
        visual_generator,
        text_generator,
        n_classes=n_classes,
        dropout=dropout,
    )

    if cuda:
        model = model.cuda()

    # model = LSTMModel2(D_m, D_e, D_h,
    #                    n_classes=n_classes,
    #                    dropout=args.dropout,
    #                    attention=args.attention).to(device)

    # model = Emoformer(D_m, D_e, n_classes=n_classes, dropout=args.dropout, attention=args.attention)

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))

    # model = CNN(200, 100, [2, 3, 4], 6)

    loss_weights = torch.FloatTensor([1.2, 0.60072, 0.38066, 0.94019, 0.67924, 0.34332])

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        # loss_function = FocalLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print("=" * 15, "data loaded", "=" * 15)
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
        "./IEMOCAP_features/IEMOCAP_features.pkl", batch_size=batch_size, valid=0.1
    )

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    acc, f_score = [[]], [[]]
    for e in range(n_epochs):
        print("=" * 15, "FFN Epoch: ", e + 1, "=" * 15)
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(
            model, loss_function, train_loader, e, optimizer, True
        )
        valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(
            model, loss_function, valid_loader, e
        )
        (
            test_loss,
            test_acc,
            test_label,
            test_pred,
            test_mask,
            test_fscore,
            attentions,
        ) = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn = (
                test_loss,
                test_label,
                test_pred,
                test_mask,
                attentions,
            )

        if args.tensorboard:
            writer.add_scalar("test: accuracy/loss", test_acc / test_loss, e)
            writer.add_scalar("train: accuracy/loss", train_acc / train_loss, e)
        print(
            "epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}".format(
                e + 1,
                train_loss,
                train_acc,
                train_fscore,
                valid_loss,
                valid_acc,
                val_fscore,
                test_loss,
                test_acc,
                test_fscore,
                round(time.time() - start_time, 2),
            )
        )
        acc.append(test_acc)
        f_score.append(test_fscore)

    acc.extend(f_score)
    acc = pd.DataFrame(np.array(acc))
    f_score = pd.DataFrame(np.array((f_score)))
    writer = pd.ExcelWriter("test.xlsx")
    acc.to_excel(writer, "sheet_1", float_format="%.2f", header=False, index=False)
    # f_score.to_excel(writer, 'sheet_1', float_format='%.2f', header=False, index=False, columns=[1])
    writer.save()
    writer.close()

    # if args.tensorboard:
    #     writer.close()

    print(acc, f_score)
    print("Test performance..")
    print(
        "Loss {} F1-score {}".format(
            best_loss,
            round(
                f1_score(
                    best_label, best_pred, sample_weight=best_mask, average="weighted"
                )
                * 100,
                2,
            ),
        )
    )
    print(
        classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)
    )
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
