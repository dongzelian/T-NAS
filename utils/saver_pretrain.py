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
            p['checkname'] = self.args.checkname
            p['batch_size'] = self.args.batch_size
            p['learning_rate'] = self.args.learning_rate
            p['momentum'] = self.args.momentum
            p['weight_decay'] = self.args.weight_decay
            p['report_freq'] = self.args.report_freq
            p['epochs'] = self.args.epochs
            p['init_channels'] = self.args.init_channels
            p['layers'] = self.args.layers
            p['model_path'] = self.args.model_path
            p['cutout'] = self.args.cutout
            p['cutout_length'] = self.args.cutout_length
            p['save'] = self.args.save
            p['seed'] =self.args.seed
            p['drop_path_prob'] = self.args.drop_path_prob
            p['arch'] = self.args.arch
            p['scheduler'] = self.args.scheduler
            p['learning_rate_min'] = self.args.learning_rate_min
            p['T_mul'] = self.args.T_mul
            p['T0'] = self.args.T0

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')




    def create_exp_dir(self, scripts_to_save=None):
        print('Experiment dir : {}'.format(self.experiment_dir))
        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)