from globals import *
from method.models import GraphSAINT
from method.minibatch import Minibatch
from utils import *
from metric import *
from method.utils import *

import warnings
warnings.filterwarnings("ignore")
import torch
import time

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))
    if mode == 'val':
        node_target = [minibatch.node_val]
    elif mode == 'test':
        node_target = [minibatch.node_test]
    else:
        assert mode == 'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]
    f1mic, f1mac = [], []
    acc, f1m, prec, rec, f1 = [], [], [], [], []
    for n in node_target:
        f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        accuracy, macro_f1, precision, recall, f1_score = calc_metrics(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        f1mic.append(f1_scores[0])
        f1mac.append(f1_scores[1])
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        f1.append(f1_score)
        f1m.append(macro_f1)
    f1mic = f1mic[0] if len(f1mic)==1 else f1mic
    f1mac = f1mac[0] if len(f1mac)==1 else f1mac
    acc = acc[0] if len(acc) == 1 else acc
    prec = prec[0] if len(prec) ==1 else prec
    rec = rec[0] if len(rec) == 1 else rec
    f1 = f1[0] if len(f1) ==1 else f1
    f1m = f1m[0] if len(f1m) == 1 else f1m

    # loss is not very accurate in this case, since loss is also contributed by training nodes
    # on the other hand, for val / test, we mostly care about their accuracy only.
    # so the loss issue is not a problem.
    return loss, f1mic, f1mac, acc, prec, rec, f1, f1m


def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every):
    if not args_global.cpu_eval:
        minibatch_eval=minibatch
    epoch_ph_start = 0
    f1mic_best, ep_best = 0, -1
    f1_best = 0
    # loss_best, ep_best = 100000, -1
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, timestamp)
    early_stop = 0
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip),style='underline')
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()

        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            l_acc_tr, l_f1m_tr, l_prec_tr, l_rec_tr, l_f1_tr = [],[],[],[],[]
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train,preds_train,labels_train = model.train_step(*minibatch.one_batch(mode='train'),current_epoch=e)
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)
                    accuracy, macro_f1, precision, recall, f1_score = calc_metrics(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_acc_tr.append(accuracy)
                    l_prec_tr.append(precision)
                    l_rec_tr.append(recall)
                    l_f1_tr.append(f1_score)
                    l_f1m_tr.append(macro_f1)
            if (e+1)%eval_val_every == 0:
                if args_global.cpu_eval:
                    torch.save(model.state_dict(),'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
                else:
                    model_eval = model
                loss_val, f1mic_val, f1mac_val,acc_val, prec_val, rec_val, f1_val, f1m_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')
                printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}\ttrain time = {:.4f} sec'\
                        .format(f_mean(l_loss_tr), f_mean(l_f1mic_tr), f_mean(l_f1mac_tr),f_mean(l_acc_tr),f_mean(l_prec_tr), f_mean(l_rec_tr), f_mean(l_f1_tr), f_mean(l_f1m_tr) , time_train_ep))
                printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}'\
                        .format(loss_val, f1mic_val, f1mac_val, acc_val,prec_val, rec_val, f1_val, f1m_val), style='yellow')
                # if f1mic_val > f1mic_best: # equals best accuracy
                #     f1mic_best, ep_best = f1mic_val, e
                if f1_val > f1_best:
                    f1_best = f1_val
                    ep_best = e
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving model ...', style='yellow')
                    torch.save(model.state_dict(), path_saver)
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= 20:
                    print("     Early Stop   ")
                    break
            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval=model
        printf('  Restoring model ...', style='yellow')
    loss, f1mic_both, f1mac_both, acc_both, prec_both, rec_both, f1_both, f1m_both = evaluate_full_batch(model_eval, minibatch_eval, mode='valtest')
    f1mic_val, f1mic_test = f1mic_both
    f1mac_val, f1mac_test = f1mac_both
    acc_val, acc_test = acc_both
    prec_val, prec_test = prec_both
    rec_val, rec_test = rec_both
    f1_val, f1_test = f1_both
    f1m_val, f1m_test = f1m_both

    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"\
            .format(ep_best, f1mic_val, f1mac_val,acc_val,prec_val, rec_val, f1_val, f1m_val ), style='red')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"\
            .format(f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test), style='red')
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')
    return f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test


if __name__ == '__main__':
    log_dir = log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)
    result_json_path = log_dir + "result.json"
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP
    f1mic_tests, f1mac_tests, acc_tests, prec_tests, rec_tests, f1_tests, f1m_tests = [], [], [], [], [], [], []
    for _ in range(args_global.repeat_time):
        model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
        f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])
        f1mic_tests.append(f1mic_test)
        f1mac_tests.append(f1mac_test)
        acc_tests.append(acc_test)
        prec_tests.append(prec_test)
        rec_tests.append(rec_test)
        f1_tests.append(f1_test)
        f1m_tests.append(f1m_test)

    json.dump({"repeat_time":args_global.repeat_time,"f1mac":f_mean(f1mac_tests), "f1mac_std":np.std(f1mac_tests),
               "f1mic":f_mean(f1mic_tests), "f1mic_std":np.std(f1mic_tests),
               "acc":f_mean(acc_tests),"acc_std":np.std(acc_tests),
               "prec":f_mean(prec_tests), "prec_std":np.std(prec_tests),
               "rec":f_mean(rec_tests),"rec_std":np.std(rec_tests),
               "f1":f_mean(f1_tests),"f1_std":np.std(f1_tests),
               "f1m":f_mean(f1m_tests), "f1m_std":np.std(f1m_tests)},
              open(result_json_path,"w", encoding="utf-8"))

