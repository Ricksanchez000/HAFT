import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('~/jupyter_base/AutoFS/code')

from lstm.dataset import DenoiseDataModule
from utils.datacollection.logger import info, error

import random
import sys
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor

from controller import GAFS
from feature_env import base_path
from lstm.utils_meter import AvgrageMeter, pairwise_accuracy, hamming_distance, count_parameters_in_MB
from Record import SelectionRecord, TransformationRecord

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
                                                      'ap_omentum_ovary', 'german_credit',
                                                      'higgs', 'housing_boston', 'ionosphere',
                                                      'lymphography', 'messidor_features', 'openml_620',
                                                      'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                      'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
                                                      'openml_589', 'openml_607', 'openml_616', 'openml_618',
                                                      'openml_637'], default='airfoil')
parser.add_argument('--mask_whole_op_p', type=float, default=0.0)
parser.add_argument('--mask_op_p', type=float, default=0.0)
parser.add_argument('--disorder_p', type=float, default=0.0)
parser.add_argument('--num', type=int, default=10)

parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')

parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=128)
parser.add_argument('--encoder_emb_size', type=int, default=64)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=128)

parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_dropout', type=float, default=0)

parser.add_argument('--new_gen', type=int, default=200)

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.80)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--top_k', type=int, default=20)

parser.add_argument('--load_epoch', type=int, default=3000)
parser.add_argument('--train_top_k', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--max_step_size', type=int, default=5)
parser.add_argument('--beams', type=int, default=5)
parser.add_argument('--add_origin', type=bool, default=True)

parser.add_argument('--keyword', type=str, default='hyper')
args = parser.parse_args()
baseline_name = [
    'kbest',
    'mrmr',
    'lasso',
    'rfe',
    # 'gfs',
    'lassonet',
    'sarlfs',
    'marlfs',

]


def gafs_train(train_queue, model: GAFS, optimizer):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']

        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch = model.forward(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())  # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))  # ce loss
        loss = (1 - args.beta) * loss_1 + (args.beta) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


def gafs_valid(queue, model: GAFS):
    pa = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch = model.forward(encoder_input)
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                             predict_value.data.squeeze().tolist())
            hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return mse.avg, pa.avg, hs.avg


# def choice_to_onehot(choice: List[int]):
#     size = len(choice)
#     onehot = torch.zeros(size + 1)
#     onehot[torch.tensor(choice)] = 1
#     return onehot[:-1]
# if choice.dim() == 1:
#     selected = torch.zeros_like(choice)
#     selected[choice] = 1
#     return selected[1:-1]
# else:
#     onehot = torch.empty_like(choice)
#     for i in range(choice.shape[0]):
#         onehot[i] = choice_to_onehot(choice[i])
#     return onehot


def gafs_infer(queue, model, step, direction='+', beams=5):
    new_gen_list = []
    original_transformation = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda(model.gpu)
        model.zero_grad()
        new_gen = model.generate_new_feature(encoder_input, predict_lambda=step, direction=direction, beams=beams)
        new_gen_list.append(new_gen.data)
        original_transformation.append(encoder_input)
    return torch.cat(new_gen_list, 0), torch.cat(original_transformation, 0)


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")

    dm = DenoiseDataModule(args)
    fe = dm.fe
    model = GAFS(fe, args, dm.tokenizer)
    train_queue = dm.train_dataloader()
    valid_queue = dm.val_dataloader()

    maybe_load_from = os.path.join(f'{base_path}', 'history', f'{dm.fe.task_name}', f'model_dmp{args.keyword}',
                                   f'{dm.fe.task_name}_{args.load_epoch}.encoder.pt')
    info(f'we load model from {maybe_load_from}:{os.path.exists(maybe_load_from)}')
    if args.load_epoch > 0 and os.path.exists(maybe_load_from):
        base_load_path = os.path.join(f'{base_path}', 'history')
        start_epoch = args.load_epoch
        model = model.from_pretrain(base_load_path, fe, args, dm.tokenizer, start_epoch, keyword=args.keyword)
        model = model.cuda(device)
        mse, pa, hs = gafs_valid(valid_queue, model)
        info("Evaluation on valid data")
        info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(start_epoch, mse, pa,
                                                                                               hs))
    else:
        start_epoch = 0
        model = model.cuda(device)

    info(f"param size = {count_parameters_in_MB(model)}MB")


    info('Training Encoder-Predictor-Decoder')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        nao_loss, nao_mse, nao_ce = gafs_train(train_queue, model, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            model.save_to(f'{base_path}/history', epoch, keyword=args.keyword)
            info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce))
        if epoch % 100 == 0 or epoch == 1:
            mse, pa, hs = gafs_valid(valid_queue, model)
            info("Evaluation on valid data")
            info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
                                                                                                   hs))
    infer_queue = dm.infer_dataloader()
    new_selection = []
    new_choice = []
    predict_step_size = 0
    while len(new_selection) < args.new_gen:
        predict_step_size += 1
        info('Generate new architectures with step size {:d}'.format(predict_step_size))
        new_record = gafs_infer(infer_queue, model, direction='+', step=predict_step_size ,beams=args.beams)
        new_choice.append(new_record)
        for choice in new_record:
            record_ = TransformationRecord.from_tensor(choice, dm.tokenizer)
            if record_ not in fe.records.r_list and record_ not in new_selection:
                new_selection.append(record_)
                info(f'gen {record_.valid}: {len(record_.ops)}/{len(record_.input_ops)}')
                info(f'{len(new_selection)} new choice generated now', )
        if predict_step_size > args.max_step_size:
            break
    info(f'build {len(new_selection)} new choice !!!')

    new_choice_pt = torch.cat(new_choice, dim=0)

    choice_path = f'{base_path}/history/{fe.task_name}/generated_choice.pt'
    torch.save(new_choice_pt, choice_path)
    info(f'save generated choice to {choice_path}')

    # torch.save(model.state_dict(), f'{base_path}/history/{fe.task_name}/GAFS.model_dict')
    best_selection = None
    best_optimal = -1000
    previous_optimal = max(dm.train_dataset.original_performance)[0]
    info(f'the best performance for this task is {previous_optimal}')
    count = 0
    for record in new_selection:
        # train_data = fe.generate_data(s.operation, 'train')
        # result = fe.get_performance(train_data)
        # test_data = fe.generate_data(s.operation, 'test')
        # test_result = fe.get_performance(test_data)
        # record = TransformationRecord.from_tensor(s, tokenizer=dm.tokenizer)
        if not record.valid:
            count += 1
            info(f'invalid percentage as : {count}/{len(new_selection)}')
            continue
        test_data = record.op(fe.original.copy(), args.add_origin)
        result = fe.get_performance(test_data)
        # if result > previous_optimal:
        #     optimal_selection = s.operation
        #     previous_optimal = result
        #     info(f'found optimal selection! the choice is {s.operation}, the performance on train is {result}')
        if result > best_optimal:
            best_selection = test_data
            best_optimal = result
            info(f'found best on train : {best_optimal}')
            info(f'the column is {test_data.columns}')
    best_str = '{:.4f}'.format(best_optimal * 100)
    best_selection.to_hdf(f'{base_path}/history/{dm.fe.task_name}-{best_str}.hdf', key='xm')

    info(f'the original performance is : {fe.get_performance()}')

    # opt_path = f'{base_path}/history/{fe.task_name}/best-ours.hdf'
    # ori_p = fe.report_performance(best_selection, flag='test')
    # info(f'found train generation in our method! the choice is {best_selection}, the performance is {ori_p}')
    # fe.generate_data(best_selection, 'train').to_hdf(opt_path, key='train')
    # fe.generate_data(best_selection, 'test').to_hdf(opt_path, key='test')

    # opt_path_test = f'{base_path}/history/{fe.task_name}/best-ours-test.hdf'
    # test_p = fe.report_performance(best_selection_test, flag='test')
    # info(f'found test generation in our method! the choice is {best_selection_test}, the performance is {test_p}')
    # fe.generate_data(best_selection_test, 'train').to_hdf(opt_path_test, key='train')
    # fe.generate_data(best_selection_test, 'test').to_hdf(opt_path_test, key='test')
    # ps = []
    # info('given overall validation')
    # report_head = 'RAW\t'
    # raw_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key='raw_test')
    # ps.append('{:.2f}'.format(fe.get_performance(raw_test) * 100))
    # for method in baseline_name:
    #     report_head += f'{method}\t'
    #     spe_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_test')
    #     ps.append('{:.2f}'.format(fe.get_performance(spe_test) * 100))
    # report_head += 'Ours\tOurs_Test'
    # report = ''
    # print(report_head)
    # for per in ps:
    #     report += f'{per}&\t'
    # report += '{:.2f}&\t'.format(ori_p * 100)
    # report += '{:.2f}&\t'.format(test_p * 100)
    # print(report)


#  gen 25
# 0.4341 [1. 1. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4357  [1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4301 gen 100

if __name__ == '__main__':
    main()
