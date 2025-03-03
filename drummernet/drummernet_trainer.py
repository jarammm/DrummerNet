from globals import *
import os, sys
import pickle
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import hpss
import evaluation
import time_freq
import custom_losses
from drummer_net import DrummerNet
from util_etc import dcnp
import wandb


fib = lambda n: pow(2 << n, n + 1, (4 << 2 * n) - (2 << n) - 1) % (2 << n)
FIBS = [fib(i) for i in range(50)]


def exp2folder(exp_name):
    """experiment name - to - result folder subpath"""
    return os.path.join('results', exp_name)


class DrummerNetTrainer(object):
    def __init__(self, drummer_net: DrummerNet, args):
        """
        """
        lr = args.learning_rate
        self.drummer_net = drummer_net
        self.n_seen_items = 0  # num of seen items for training
        self.n_seen_batches = 0
        self.loss_histories = {'training': None, 'test': None}
        n_bins = args.n_cqt_bins
        self.cqters = {
            'c1': time_freq.PseudoCqt(SR_WAV, 64, 1 * 32.703195, n_bins, n_bins),
            'c2': time_freq.PseudoCqt(SR_WAV, 64, 2 * 32.703195, n_bins, n_bins),
            'c3': time_freq.PseudoCqt(SR_WAV, 64, 4 * 32.703195, n_bins, n_bins),
            'c4': time_freq.PseudoCqt(SR_WAV, 64, 8 * 32.703195, n_bins, n_bins),
            'c5': time_freq.PseudoCqt(SR_WAV, 64, 16 * 32.703195, n_bins, n_bins),
            'c6': time_freq.PseudoCqt(SR_WAV, 64, 32 * 32.703195, n_bins, n_bins),
            'c7': time_freq.PseudoCqt(SR_WAV, 64, 64 * 32.703195, n_bins, n_bins),
            'c8': time_freq.PseudoCqt(SR_WAV, 64, 4000, n_bins, n_bins)}  # since sr=16 kHz

        self.optimizer = optim.Adam(self.drummer_net.parameters(), lr=lr)
        self.exp_name = args.exp_name
        self.result_folder = exp2folder(self.exp_name)
        self.loss_domains = args.loss_domains
        metrics_dict = {'mae': F.l1_loss, 'mse': F.mse_loss}
        self.metric_funcs = [metrics_dict[m] for m in args.metrics]
        assert self.metric_funcs != [], 'there is no metric!'
        self.metric_names = args.metrics
        self.src_norm = args.source_norm
        self.hpss = args.compare_after_hpss
        self.args = args

        if 'melgram' in self.loss_domains:
            self.mel_n_fft = 1024
            self.mel_fb = nn.Parameter(
                torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=self.mel_n_fft, n_mels=self.args.n_mels,
                                                     fmin=0.0, fmax=SR / 2.0)).type(TCDTYPE))
        if 'stft' in self.loss_domains:
            self.n_fft = 1024

        if args.eval is True:
            ddfs = [evaluation.get_ddf_bdt()] # evaluation.get_ddf_mdb(), evaluation.get_ddf_enst() --> add if you want
            self.evalers = [evaluation.DrumEvaluator(self.drummer_net, ddf,
                                                     device=DEVICE) for ddf in ddfs]
            self.scores = {ddf.name: np.zeros((0, 5)) for ddf in ddfs}  # N by 5(x_n_item, KD, SD, HH, CY)
        else:
            self.evalers = []

    def prepare(self, args):
        """prepare to train by
            - creating result folder
            - save the drummernet summary as text
            - save `args` in text and pickle
        """
        os.makedirs(self.result_folder, exist_ok=True)
        with open(os.path.join(self.result_folder, 'cmd.txt'), 'w') as f:
            f.write(' '.join(sys.argv))

        orig_stdout = sys.stdout
        with open(os.path.join(self.result_folder, 'summary.txt'), 'w') as f:
            sys.stdout = f
            print(self.drummer_net)
        sys.stdout = orig_stdout

        with open(os.path.join(self.result_folder, 'args.pkl'), 'wb') as f_write:
            pickle.dump(args, f_write)
        with open(os.path.join(self.result_folder, 'args.txt'), 'w') as f:
            f.write(str(args))

    def train_many_epochs(self, n_epoch, tr_loader, n_tr_item=None):
        """Train the drummernet with the given training loader.

        Args:
            n_epoch (int): number of epochs to train

            tr_loader (torch.DataLoader): data loader for training

            n_tr_item (int): number of training items in each epoch.
                If this is bigger than #item in tr_loader, tr_loader is
                repeatedly used to satisfy it.
        """
        if n_tr_item is None:
            n_tr_item = len(tr_loader)

        n_repeat_loader = max(n_tr_item // tr_loader.batch_size // len(tr_loader), 1)
        print('%s: %d items so far, + %d item now. Starting to train for %d epoches x %d times' %
              (self.exp_name, self.n_seen_items, n_tr_item, n_epoch, n_repeat_loader))
        for epoch in range(n_epoch):
            for i in range(n_repeat_loader):
                self.train_epoch(tr_loader)
                if (i + 1) % 20 == 0:
                    print('..%s: %d items so far..' %
                          (self.exp_name, self.n_seen_items))

        self.evaluate(result_subfolder='items_' + str(self.n_seen_items))
        torch.save(self.drummer_net.state_dict(), f'items_{str(self.n_seen_items)}.pth')

    def train_epoch(self, tr_loader):
        """train the model for one epoch

        Args:
            tr_loader (torch.DataLoader): data loader for training
        """
        def _loss_a_batch(mixes):
            """
            Compute loss for a batch of drummernet outputs

            Args:
                mixes (torch.tensor): (batch, time), a batch of audio waveforms
                    which is input of drummernet

            Returns:
                losses (dict): a dict for each losses

                loss (torch.tensor): `torch.sum` of all the `losses`
            """
            self.optimizer.zero_grad()
            mixes = mixes.to(DEVICE)
            mix_t, est_mix, est_irs = self.drummer_net(mixes)

            losses = self._compute_loss(mix_t, est_mix, est_irs)

            loss = None
            for i, key in enumerate(losses):
                if i == 0:
                    loss = losses[key]
                else:
                    loss = loss + losses[key]

            return losses, loss

        def _train_a_batch(mixes):
            """train a batch.

            Args:
                mixes (torch.tensor): a batch of audio waveforms, (batch, time)

            Returns:
                losses (dict): a dict for each losses

            """
            losses, loss = _loss_a_batch(mixes)
            loss.backward()

            self.optimizer.step()

            self.n_seen_items += len(mixes)
            self.n_seen_batches += 1

            return losses

        #
        self.drummer_net.train()
        bar = tqdm(enumerate(tr_loader), total=len(tr_loader))
        accum_losses = defaultdict(lambda: 0.)
        for batch_i, (mix, _, _) in bar:
            losses = _train_a_batch(mix)
            desc = ' '.join('{}:{:4.2f}'.format(k, v) for (k, v) in losses.items())
            bar.set_description(desc)
            for key in losses:
                accum_losses[key] += dcnp(losses[key])

        print(' '.join('{}:{:4.2f}'.format(k, v / float(len(tr_loader))) for (k, v) in accum_losses.items()) +
              ' on average.')

    def evaluate(self):
        """do evaluation on other datasets using self.evalers, then save the result"""
        self.drummer_net.eval()
        keys = ['KD', 'CL', 'HH', 'CY']
        for evaler in self.evalers:
            ddf_name = evaler.ddf.name

            evaler.predict(verbose=True)
            evaler.pickpeaks(evaluation.pickpeak_fix, verbose=True)
            evaler.mir_eval()

            scores = np.zeros((1, 5))
            scores[0, 0] = self.n_seen_items

            for i, key in enumerate(keys):
                scores[0, i + 1] = np.array(evaler.f_scores[key]).mean(axis=0)[0]

            self.scores[ddf_name] = np.concatenate((self.scores[ddf_name], scores), axis=0)
        if self.args.use_wandb:    
            wandb.log({
            f"{ddf_name}_f1_KD": scores[0, 1],
            f"{ddf_name}_f1_CL": scores[0, 2],
            f"{ddf_name}_f1_HH": scores[0, 3],
            f"{ddf_name}_f1_CY": scores[0, 4],
            })

    def _compute_loss(self, mixes, est_mixes, est_impulses):
        """It's an interface to combine
            - signal loss
            - feature loss
            - l1 reg

        Args:
            mixes (torch.Tensor): (batch, time) input waveforms (drum stems)

            est_mixes (torch.Tensor): (batch, time) output waveforms (estimated drum stems).
                The drum reconstruction done by drummer net

            est_impulses (torch.Tensor): (batch, ch=inst, time), transcription estimation

        Returns:
            losses (dict): dictionary, {loss_name: loss_value}
        """
        losses = defaultdict(lambda: 0.)
        if self.hpss:
            mixes = hpss.pss_src(mixes)
            est_mixes = hpss.pss_src(est_mixes)

        if 'spectrum' in self.loss_domains:  # cqt
            self._compute_spectrum_loss(mixes, est_mixes, losses=losses)
        if 'melgram' in self.loss_domains:
            self._compute_melgram_loss(mixes, est_mixes, losses=losses, weight=1.0)
        if 'stft' in self.loss_domains:
            self._compute_stft_loss(mixes, est_mixes, losses=losses, weight=1.0)
        if 'l1_reg' in self.loss_domains:
            losses['l1_reg'] = custom_losses.norm_losses(est_impulses, p=1, weight=self.args.l1_reg_lambda)
        
        if self.args.use_wandb:
            losses_for_logging = {f"loss/{k}": v.item() for k, v in losses.items()}
            wandb.log(losses_for_logging)

        return losses

    def _compute_melgram_loss(self, mix, est_mix, losses, weight=1.0):
        custom_losses.loss_melgram(mix, est_mix, self.mel_fb.to(mix.device),
                                   self.mel_n_fft, self.metric_funcs,
                                   self.metric_names, losses)

    def _compute_stft_loss(self, mix, est_mix, losses, weight=1.0):
        custom_losses.loss_stft(mix, est_mix,
                                self.n_fft, self.metric_funcs,
                                self.metric_names, losses)

    def _compute_spectrum_loss(self, mix, est_mix, losses, weight=1.0, perceptual=False):
        """compute CQT losses"""
        for cqter_key in self.cqters:
            cqter = self.cqters[cqter_key]
            est_cqt = cqter(est_mix)
            org_cqt = cqter(mix)

            for metric_name, metric in zip(self.metric_names, self.metric_funcs):
                loss = weight * 4 * metric(org_cqt, est_cqt)
                if not torch.isnan(loss):
                    losses[cqter_key + metric_name] = loss
                else:
                    print('%s was NaN, it is not added to the total loss' % cqter_key)
