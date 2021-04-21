#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe,  is_image_file
from utils.loss import  FocalLoss
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate
from option import Options
import warnings, random


warnings.filterwarnings('ignore')


def setup_seed(seed):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(20)

def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(20 + worker_id)

# 设置随机数种子
args = Options().parse()
n_class = args.n_class



data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################

evaluation = args.evaluation
test = False

print("evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
sub_batch_size = args.sub_batch_size # batch size for train reinforcement patches
ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train/Sat")) if is_image_file(image_name)]
ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali/Sat")) if is_image_file(image_name)]
ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "offical_crossvali/Sat")) if is_image_file(image_name)]
ids_train_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali/Sat")) if is_image_file(image_name)]


dataset_train = DeepGlobe(os.path.join(data_path, "train"), ids_train, label=True, transform=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=True, num_workers=0, collate_fn=collate,
                                                pin_memory=True, worker_init_fn=worker_init_fn)

dataset_val = DeepGlobe(os.path.join(data_path, "crossvali"), ids_val, label=True)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size,shuffle=False, num_workers=0, collate_fn=collate,  pin_memory=True
                                             )
dataset_test = DeepGlobe(os.path.join(data_path, "offical_crossvali"), ids_test, label=True)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=False, num_workers=0, collate_fn=collate,  pin_memory=True
                                              )

##### sizes are (w, h) ##############################
# make sure margin / 32 is over 1.5 AND size_g is divisible by 4
size_g = (args.size_g, args.size_g) # resized global image
size_p = (args.size_p, args.size_p) # cropped global out local patch size
size_lp = (args.size_lp, args.size_lp) # cropped local patch size
###################################
print("creating models......")

path_weight = args.path_weight

model = create_model_load_weights(n_class, model_path, path_weight)

###################################
epochs_local = args.epochs_local
epochs_global = args.epochs_global
epochs_L = args.epochs_L

learning_rate0 = args.lr0
learning_rate1 = args.lr1
learning_rate2 = args.lr2
learning_rate3 = args.lr3

optimizer_local = get_optimizer(model, mode=0, learning_rate=learning_rate0)
optimizer_global = get_optimizer(model, mode=1, learning_rate=learning_rate1)
optimizer_classifier = get_optimizer(model, mode=2, learning_rate=learning_rate2)
optimizer_rein = get_optimizer(model, mode=3, learning_rate=learning_rate3)

lr_groups0 = np.array([learning_rate0, learning_rate0])
scheduler_local = LR_Scheduler('poly', lr_groups0, epochs_local, len(dataloader_train))

lr_groups1 = np.array([learning_rate1, learning_rate1])
scheduler_global = LR_Scheduler('poly', lr_groups1, epochs_L, len(dataloader_train))

lr_groups2 = np.array([learning_rate2, learning_rate2])
scheduler_classifier = LR_Scheduler('poly', lr_groups2, epochs_L, len(dataloader_train), decay=0.95)
# scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_classifier, verbose=True, patience=1)

lr_groups3 = np.array([learning_rate3, learning_rate3, learning_rate3, learning_rate3, learning_rate3,learning_rate3])
scheduler_rein = LR_Scheduler('poly', lr_groups3, epochs_L, len(dataloader_train))
##################################



criterion1 = FocalLoss(gamma=3, ignore=0, weight=None)
criterion_classifier = nn.BCELoss()


f_log = open(log_path + task_name + ".log", 'a')


trainer = Trainer(criterion1, n_class, size_g, size_p, size_lp, batch_size=4, sub_batch_size=4,
                  weight_l=None, weight_rein=None, lamda_l=None, lamda_rein=None, mu=None)
evaluator = Evaluator(n_class, size_g, size_p, batch_size=8, sub_batch_size=8, test=test)


print("training adaptive local feature reinforcement")

best_pred_cls = 0.0
best_pred_rein = 0.0
best_local = 0.0
best_fuse_g = 0.0
best_global = 0.0
best_fuse = 0.0
is_save = False

# train global branch
for epoch in range(epochs_global):
    tbar = tqdm(dataloader_train)
    trainer.set_mode(1)
    trainer.set_train(model)
    trainer.set_optimizer(optimizer_global)
    trainer.set_loss(criterion1)
    train_loss_g = 0
    tscores = 0.0
    for i_batch, sample_batched in enumerate(tbar):
        if evaluation: break
        scheduler_global(optimizer_global, i_batch, epoch, best_pred_rein)
        loss_g = trainer.train(sample_batched, model)
        train_loss_g = train_loss_g + loss_g
        score = trainer.get_scores()

        tbar.set_description('Train loss{G:%.3f;},mIoU{G:%.3f;}' % (train_loss_g / (i_batch + 1), score[0]))
    trainer.reset_metrics()
    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            tscores = 0.0
            eval_loss = 0.0
            evaluator.set_mode(1)
            if test:
                tbar = tqdm(dataloader_test)
            else:
                tbar = tqdm(dataloader_val)
            for i_batch, sample_batched in enumerate(tbar):
                evaluator.eval_test(sample_batched, model)
                score = evaluator.get_scores()
                tbar.set_description('mIoU{tea:%.3f}' % (score[0]))

            if test:
                break
            else:
                score = evaluator.get_scores()
                evaluator.reset_metrics()
                if np.mean(score[0]) > best_global:
                    best_global = np.mean(score[0])
                    torch.save(model.module.state_dict(), "./saved_models/" + 'Best_global_' + task_name + ".pth")

                log = "\n================================\n"
                # log = "Reinforcement evaluation\n================================\n" if epoch==0 else ""
                log = log + 'epoch [{}/{}] Reinforced -- IoU: tea = {:.4f}'. \
                    format(epoch + 1, epochs_L, score[0]) + "\n"
                log = log + 'Best IoU: tea = {:.4f}'. \
                    format(best_global) + "\n"
                log += "================================\n"
                print(log)
                if evaluation: break
                f_log.write(log)
                f_log.flush()


for epoch in range(epochs_L):
    # print('train local reinforcement classifier')
    # trainer.set_mode(2)
    # trainer.set_train(model)
    # trainer.set_optimizer(optimizer_classifier)
    # trainer.set_loss(criterion_classifier)
    # optimizer_classifier.zero_grad()
    # # tbar = tqdm(dataloader_val_pre)
    # tbar = tqdm(dataloader_train)
    # train_loss = 0
    # score = 0.0
    # tscores = 0.0
    # tscores_eval_classifier = 0.0
    # tscores_eval_joint = 0.0
    # for i_batch, sample_batched in enumerate(tbar):
    #     scheduler_classifier(optimizer_classifier, i_batch, epoch, best_pred_cls)
    #     loss, score = trainer.train(sample_batched, model)
    #     train_loss += loss.item()
    #     tscores += score
    #     tbar.set_description('Train loss: %.4f; Correct score: %.4f; Mean correct score: %.3f' % (train_loss / (i_batch + 1), score, tscores / (i_batch + 1)))
    # if tscores / (i_batch + 1) > best_pred_cls:
    #     best_pred_cls = tscores / (i_batch + 1)
    #
    # if epoch % 1 == 0:
    #     with torch.no_grad():
    #         model.eval()
    #         evaluator.set_mode(2)
    #         print("evaluating...")
    #         if test:
    #             tbar = tqdm(dataloader_test)
    #         else:
    #             tbar = tqdm(dataloader_val)
    #         for i_batch, sample_batched in enumerate(tbar):
    #             score = evaluator.eval_test(sample_batched, model)
    #             tscores_eval_classifier += score
    #             tbar.set_description('score: %.3f;' % (tscores_eval_classifier / (i_batch + 1)))

    # 联合训练
    trainer.set_mode(3)
    trainer.set_train(model)
    trainer.set_optimizer(optimizer_rein)
    trainer.set_loss(criterion1)
    tbar = tqdm(dataloader_train)
    train_loss = 0
    print('train reinforced global with classifier')
    for i_batch, sample_batched in enumerate(tbar):
        break
        if evaluation: break
        scheduler_rein(optimizer_rein, i_batch, epoch, best_pred_rein)
        loss = trainer.train(sample_batched, model)
        train_loss += loss.item()
        score_train_global, score_train_refine = trainer.get_scores()
        tbar.set_description('Train loss: %.3f; global mIoU: %.3f;refine mIoU: %.3f;' % (train_loss / (i_batch + 1), score_train_global, score_train_refine))
    score_train_global, _ = trainer.get_scores()
    trainer.reset_metrics()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            evaluator.set_mode(3)
            print("evaluating...")

            if test:
                tbar = tqdm(dataloader_test)
            else:
                tbar = tqdm(dataloader_val)

            for i_batch, sample_batched in enumerate(tbar):
                _ = evaluator.eval_test(sample_batched, model)
                score_val_global, score_val_refine = evaluator.get_scores()
                tbar.set_description('global_miou: %.4f;miou_refine %.4f' %(score_val_global, score_val_refine))


if not evaluation: f_log.close()
