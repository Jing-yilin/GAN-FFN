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
    trainset = IEMOCAPDataset(path=path, train=True)
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
    """
    Utility function to train model for one epoch of train data
    or evaluate model on val/test data.
    :param model: torch NN model
    :param loss_function: loss function to optimize
    :param dataloader: torch Dataloader for train/val/test data
    :param epoch: number of epoch
    :param optimizer: optimizer to use to train model
    :param train: boolean value if train or val/test
    """
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
    dataset_path="./IEMOCAP_features/IEMOCAP_features.pkl",
) -> pd.DataFrame:
    """
    Train the GAN model
    :param acoustic_generator: acoustic generator model
    :param visual_generator: visual generator model
    :param text_generator: text generator model
    :param acoustic_discriminator: acoustic discriminator model
    :param visual_discriminator: visual discriminator model
    :param text_discriminator: text discriminator model
    :param epochs: number of epochs to train
    :param batch_size: batch size
    :param lr: learning rate
    :param b1: Adam optimizer beta1
    :param b2: Adam optimizer beta2
    :param dataset_path: path to dataset

    :return: pandas dataframe with training information
    """
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
        dataset_path, batch_size=batch_size, valid=0.1
    )
    # 新建一个df用于存储训练过程中的loss
    # Acoustic G D D, Visual G D D, Text G D D
    columns = [
        "epoch",
        "acoustic_G_loss",
        "visual_G_loss",
        "text_G_loss",
        "visual_D_loss",
        "text_D_loss",
        "acoustic_D_loss",
    ]
    loss_df = pd.DataFrame(columns=columns)

    # Start epochs
    for epoch in range(epochs):
        print("=" * 15, "start Epoch : ", epoch, "=" * 15)
        for i, data in enumerate(train_loader):
            loss = {
                "epoch": epoch,
                "acoustic_G_loss": 0,
                "visual_G_loss": 0,
                "text_G_loss": 0,
                "visual_D_loss": 0,
                "text_D_loss": 0,
                "acoustic_D_loss": 0,
            }

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
            #  1.1 Train AcousticGenerator
            # -----------------
            # print("=" * 15, "Train AcousticGenerator", "=" * 15)

            optimizer_acoustic_G.zero_grad()

            # Generate a batch of fusions
            acoustic_fusion = acoustic_generator(real_acoustic)

            # Loss measures generator's ability to fool the discriminator
            visual_prob = visual_discriminator(acoustic_fusion)
            text_prob = text_discriminator(acoustic_fusion)
            # print("visual_prob.shape = ", visual_prob.shape) # torch.Size([94, 32, 1])

            g_loss = (
                adversarial_loss(visual_prob, valid)
                + adversarial_loss(text_prob, valid)
            ) / 2.0
            loss["acoustic_G_loss"] = g_loss.detach().numpy()

            g_loss.backward()
            optimizer_acoustic_G.step()

            # ---------------------
            #  1.2 Train VisualDiscriminator
            # ---------------------
            # print("-" * 8, "Train VisualDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_visual_prob = visual_discriminator(real_visual)
            # Loss for fake images
            fake_visual_prob = visual_discriminator(acoustic_fusion.detach())
            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_visual_prob, valid)
                + adversarial_loss(fake_visual_prob, fake)
            ) / 2.0
            loss["visual_D_loss"] = d_loss.detach().numpy()

            d_loss.backward()
            optimizer_visual_D.step()

            # ---------------------
            #  1.3 Train TextDiscriminator
            # ---------------------
            # print("-" * 8, "Train TextDiscriminator", "-" * 8)

            optimizer_text_D.zero_grad()

            # Loss for real images
            real_text_prob = text_discriminator(real_text)

            # Loss for fake images
            fake_text_prob = text_discriminator(acoustic_fusion.detach())

            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_text_prob, valid)
                + adversarial_loss(fake_text_prob, fake)
            ) / 2.0
            loss["text_D_loss"] = d_loss.detach().numpy()

            d_loss.backward()
            optimizer_text_D.step()

            # print("loss = ", loss)

            # -----------------
            #  2.1 Train VisualGenerator
            # -----------------
            # print("=" * 15, "Train VisualGenerator", "=" * 15)

            optimizer_visual_G.zero_grad()

            # Generate a batch of images
            visual_fusion = visual_generator(real_visual)

            # Loss measures generator's ability to fool the discriminator
            acoustic_prob = acoustic_discriminator(visual_fusion)
            text_prob = text_discriminator(visual_fusion)
            # print("visual_prob.shape = ", visual_prob.shape) # torch.Size([94, 32, 1])

            g_loss = (
                adversarial_loss(acoustic_prob, valid)
                + adversarial_loss(text_prob, valid)
            ) / 2.0
            loss["visual_G_loss"] = g_loss.detach().numpy()

            g_loss.backward()
            optimizer_visual_G.step()

            # ---------------------
            #  2.2 Train AcousticDiscriminator
            # ---------------------
            # print("-" * 8, "Train AcousticDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_acoustic_prob = acoustic_discriminator(real_acoustic)

            # Loss for fake images
            fake_acoustic_prob = acoustic_discriminator(visual_fusion.detach())

            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_acoustic_prob, valid)
                + adversarial_loss(fake_acoustic_prob, fake)
            ) / 2.0
            loss["acoustic_D_loss"] = d_loss.detach().numpy()

            d_loss.backward()
            optimizer_acoustic_D.step()

            # ---------------------
            #  2.3 Train TextDiscriminator
            # ---------------------
            # print("-" * 8, "Train TextDiscriminator", "-" * 8)

            optimizer_text_D.zero_grad()

            # Loss for real images
            real_text_prob = text_discriminator(real_text)

            # Loss for fake images
            fake_text_prob = text_discriminator(visual_fusion.detach())

            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_text_prob, valid)
                + adversarial_loss(fake_text_prob, fake)
            ) / 2.0
            loss["text_D_loss"] = d_loss.detach().numpy()
            # print("loss = ", loss)

            d_loss.backward()
            optimizer_text_D.step()

            # -----------------
            #  3.1 Train TextGenerator
            # -----------------
            # print("=" * 15, "Train TextGenerator", "=" * 15)

            optimizer_text_G.zero_grad()

            # Generate a batch of images
            text_fusion = text_generator(real_text)

            # Loss measures generator's ability to fool the discriminator
            acoustic_prob = acoustic_discriminator(text_fusion)
            visual_prob = visual_discriminator(text_fusion)
            # print("text_prob.shape = ", text_prob.shape) # torch.Size([94, 32, 1])

            g_loss = (
                adversarial_loss(acoustic_prob, valid)
                + adversarial_loss(visual_prob, valid)
            ) / 2.0
            loss["text_G_loss"] = g_loss.detach().numpy()

            g_loss.backward()
            optimizer_text_G.step()

            # ---------------------
            #  3.2 Train AcousticDiscriminator
            # ---------------------
            # print("-" * 8, "Train AcousticDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_acoustic_prob = acoustic_discriminator(real_acoustic)

            # Loss for fake images
            fake_acoustic_prob = acoustic_discriminator(text_fusion.detach())

            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_acoustic_prob, valid)
                + adversarial_loss(fake_acoustic_prob, fake)
            ) / 2.0
            loss["acoustic_D_loss"] = d_loss.detach().numpy()
            d_loss.backward()
            optimizer_acoustic_D.step()

            # ---------------------
            #  3.3 Train VisualDiscriminator
            # ---------------------
            # print("-" * 8, "Train VisualDiscriminator", "-" * 8)

            optimizer_visual_D.zero_grad()

            # Loss for real images
            real_visual_prob = visual_discriminator(real_visual)

            # Loss for fake images
            fake_visual_prob = visual_discriminator(text_fusion.detach())

            # Total discriminator loss
            d_loss = (
                adversarial_loss(real_visual_prob, valid)
                + adversarial_loss(fake_visual_prob, fake)
            ) / 2.0
            loss["visual_D_loss"] = d_loss.detach().numpy()

            # 以表格的形式打印loss这个字典
            loss = pd.DataFrame(loss, index=[0])
            print(loss)

            d_loss.backward()
            optimizer_visual_D.step()
            # 在每个epoch的最后一次batch中，将loss添加到loss_df中
            # 把loss添加到loss_df中
            if i == len(train_loader) - 1:
                # loss_df添加上loss
                loss_df = pd.concat([loss_df, loss], axis=0, ignore_index=True)
    return loss_df


def draw_GAN_loss(df: pd.DataFrame) -> None:
    """画出GAN的loss曲线"""
    plt.figure(figsize=(10, 8))
    # 画出loss曲线
    plt.plot(df["epoch"], df["acoustic_G_loss"], label="acoustic_G_loss")
    plt.plot(df["epoch"], df["visual_G_loss"], label="visual_G_loss")
    plt.plot(df["epoch"], df["text_G_loss"], label="text_G_loss")
    plt.plot(df["epoch"], df["visual_D_loss"], label="visual_D_loss")
    plt.plot(df["epoch"], df["text_D_loss"], label="text_D_loss")
    plt.plot(df["epoch"], df["acoustic_D_loss"], label="acoustic_D_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("GAN loss")
    # 保存
    if not os.path.exists("./output"):
        os.mkdir("./output")
    plt.savefig("./output/GAN_loss.png")


def save_GAN_loss(df: pd.DataFrame) -> None:
    # 保存loss
    if not os.path.exists("./output"):
        os.mkdir("./output")
    df.to_csv("./output/GAN_loss.csv", index=False)


def save_GAN_models(models: list, save_path: str) -> None:
    models_name = [
        "acoustic_generator",
        "acoustic_discriminator",
        "visual_generator",
        "visual_discriminator",
        "text_generator",
        "text_discriminator",
    ]
    for id, model in enumerate(models):
        path = save_path + models_name[id] + ".pth"
        torch.save(model, path)


def main():
    dataset_path = "./IEMOCAP_features/IEMOCAP_features.pkl"
    model_save_path = "./GAN_save/"
    # 创建文件
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
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
        "--dropout", type=float, default=0.5, metavar="dropout", help="dropout rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="BS", help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, metavar="E", help="number of epochs"
    )
    parser.add_argument(
        "--GAN-epochs", type=int, default=5, metavar="E", help="number of GAN epochs"
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
        "--use-trained-GAN",
        action="store_true",
        default=False,
        help="Use trained GAN",
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
    g_epochs = args.GAN_epochs

    n_classes = 6
    D_m = 200
    D_e = 30
    D_h = 100

    use_trained_GAN = args.use_trained_GAN

    if use_trained_GAN == True:
        acoustic_generator = torch.load(
            model_save_path + "acoustic_generator.pth"
        ).eval()
        acoustic_discriminator = torch.load(
            model_save_path + "acoustic_discriminator.pth"
        ).eval()
        visual_generator = torch.load(model_save_path + "visual_generator.pth").eval()
        visual_discriminator = torch.load(
            model_save_path + "visual_discriminator.pth"
        ).eval()
        text_generator = torch.load(model_save_path + "text_generator.pth").eval()
        text_discriminator = torch.load(
            model_save_path + "text_discriminator.pth"
        ).eval()
        print("=" * 15, model_save_path + "loaded trained GAN", "=" * 15)

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

        loss_df = train_GAN(
            acoustic_generator,
            visual_generator,
            text_generator,
            acoustic_discriminator,
            visual_discriminator,
            text_discriminator,
            epochs=g_epochs,
            batch_size=32,
            lr=0.0005,
            b1=0.6,
            b2=0.6,
        )

        save_GAN_loss(loss_df)
        draw_GAN_loss(loss_df)
        # save model parameters
        models = [
            acoustic_generator,
            acoustic_discriminator,
            visual_generator,
            visual_discriminator,
            text_generator,
            text_discriminator,
        ]
        save_GAN_models(models, model_save_path)

        # change mode to eval
        for model in models:
            model.eval()

        print("=" * 15, "finished training GAN", "=" * 15)

    # TODO: GAN-FFN还需要进一步调优，目前需要训练到17轮才出现明显的准确度提升
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
        dataset_path, batch_size=batch_size, valid=0.1
    )

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    acc, f_score = [[]], [[]]
    for e in range(n_epochs):
        print("=" * 15, "FFN Epoch: ", e + 1, "=" * 15)
        start_time = time.time()
        # 训练
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(
            model, loss_function, train_loader, e, optimizer, True
        )
        # 验证
        valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(
            model, loss_function, valid_loader, e
        )
        # 测试
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
    # acc = pd.DataFrame(np.array(acc))
    f_score = pd.DataFrame(np.array((f_score)))
    writer = pd.ExcelWriter("test.xlsx")
    # acc.to_excel(writer, "sheet_1", float_format="%.2f", header=False, index=False)
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


if __name__ == "__main__":
    main()
