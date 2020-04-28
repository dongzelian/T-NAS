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
            p['run'] = self.args.run
            p['data_path'] = self.args.data_path
            p['batch_size'] = self.args.batch_size
            p['learning_rate'] = self.args.learning_rate
            p['learning_rate_min'] = self.args.learning_rate_min
            p['img_size'] = self.args.img_size
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
            p['grad_clip'] = self.args.grad_clip
            p['train_portion'] = self.args.train_portion
            p['arch_learning_rate'] = self.args.arch_learning_rate
            p['arch_weight_decay'] = self.args.arch_weight_decay

            if self.args.checkname == 'enas':
                p['test_freq'] = self.args.test_freq
                p['gamma'] = self.args.gamma
                p['inner_steps'] = self.args.inner_steps
                p['inner_lr'] = self.args.inner_lr
                p['valid_inner_steps'] =self.args.valid_inner_steps
                p['n_archs'] = self.args.n_archs
                p['controller_type'] = self.args.controller_type
                p['controller_hid'] = self.args.controller_hid
                p['controller_temperature'] = self.args.controller_temperature
                p['controller_tanh_constant'] = self.args.controller_tanh_constant
                p['entropy_coeff'] = self.args.entropy_coeff
                p['lstm_num_layers'] =self.args.lstm_num_layers
                p['controller_op_tanh_reduce'] = self.args.controller_op_tanh_reduce
                p['controller_start_training'] = self.args.controller_start_training
                p['scheduler'] = self.args.scheduler
                p['T_mul'] = self.args.T_mul
                p['T0'] = self.args.T0
                p['store'] = self.args.store
                p['benchmark_path'] = self.args.benchmark_path
                p['restore_path'] = self.args.restore_path
            elif self.args.checkname == 'darts':
                p['drop_path_prob'] = self.args.drop_path_prob
                p['unrolled'] = self.args.unrolled

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')



    def create_exp_dir(self, scripts_to_save=None):
        print('Experiment dir : {}'.format(self.experiment_dir))
        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)