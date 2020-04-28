import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.run, args.dataset, args.checkname)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        run_list = sorted([int(m.split('_')[-1]) for m in self.runs])
        run_id = run_list[-1] + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_acc = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            acc = float(f.readline())
                            previous_acc.append(acc)
                    else:
                        continue
                max_acc = max(previous_acc)
                if best_pred > max_acc:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best_all.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best_all.pth.tar'))


    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        with open(logfile, 'w+') as log_file:
            p = OrderedDict()
            p['dataset'] = self.args.dataset
            p['seed'] = self.args.seed
            p['epoch'] = self.args.epoch
            p['n_way'] = self.args.n_way
            p['k_spt'] = self.args.k_spt
            p['k_qry'] = self.args.k_qry
            p['batch_size'] = self.args.batch_size
            p['test_batch_size'] = self.args.test_batch_size
            p['meta_batch_size'] = self.args.meta_batch_size
            p['meta_test_batch_size'] = self.args.meta_test_batch_size
            p['meta_lr_theta'] = self.args.meta_lr_theta
            p['update_lr_theta'] = self.args.update_lr_theta
            p['meta_lr_w'] = self.args.meta_lr_w
            p['update_lr_w'] = self.args.update_lr_w
            p['update_step'] = self.args.update_step
            p['update_step_test'] =self.args.update_step_test

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')

    def create_exp_dir(self, scripts_to_save=None):
        print('Experiment dir : {}'.format(self.experiment_dir))
        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)