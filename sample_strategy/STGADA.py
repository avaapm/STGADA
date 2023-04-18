import torch
import numpy as np
import sys
sys.path.append('..')
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import copy
from copy import deepcopy
from scipy import stats
import torch.optim as optim
import time
from torchvision import transforms
from .strategy import Strategy
from util.losses import softmax_mse_loss, softmax_kl_loss, adentropy, Margin_loss
from util.utils import *
from itertools import cycle
from util.spectral_transfer import FDA

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
class STGADA(Strategy):
    def __init__(self, source, target_train, target_test, idxs_lb, net, cfg, logger, target_train_fda=None):
        super(STGADA, self).__init__(source, target_train, target_test, idxs_lb, net, cfg,  target_train_fda)
        global device;device = torch.device("cuda:" + self.cfg.DEVICE if self.cfg.USE_CUDA else "cpu")
        self.clf = self.net.to(device)
        self.logger = logger
    def sdm_active(self, inds, candidate_dataset, aim_dataset):
        active_samples = list()
        for i in inds:
            _, target, path, _ = candidate_dataset[i]
            active_samples.append([path,target])
        # self.idxs_lb[inds] = True

        aim_dataset.add_item(active_samples)
        candidate_dataset.remove_item(inds)


    def train_STGADA(self,epoch):
        self.clf.train()
        optimizer = optim.Adadelta(self.clf.parameters_list(self.cfg.LEARN_RATE), lr=self.cfg.LEARN_RATE)
        fda = FDA(l=self.cfg.BETA)
        for batch_idx, ((data, target, path, _), (data_tgt, _, _, _)) in enumerate(zip(self.source, cycle(self.target_train_fda))):
            mini_batchsize = len(target)
            data, target = data.to(device), target.to(device)
            data_tgt = data_tgt.to(device)
            data_fda = fda(data, data_tgt)

            for src_idx in range(len(target)):
                # element = path[src_idx].split("/")[6]
                if 'flir' in path[src_idx].split("/"):
                    data_fda[src_idx, :, :, :] = data[src_idx, :, :, :]
                    print("yes")

            data = normalize(data_fda)
            optimizer.zero_grad()
            feature, logit = self.clf(data)
            loss = F.cross_entropy(logit, target)
            label = torch.unsqueeze(target,dim=1)
            onehot_label = torch.zeros_like(logit).scatter_(1,label.long(),1)
            addition_loss = (F.normalize(logit) * onehot_label).sum() / mini_batchsize
            smooth_marginloss = 1 - (torch.sum(onehot_label * logit,1).unsqueeze(1) * torch.ones_like(logit) - logit)
            margin_loss = Margin_loss(F.normalize(logit),target,weight=smooth_marginloss)
            total_loss = margin_loss - 1 * addition_loss + loss
            total_loss.backward()
            optimizer.step()
            if batch_idx % self.cfg.LOG_INTERVAL == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tClf CE Loss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(self.source.dataset),
                #             100. * batch_idx / len(self.source), loss.item()))
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tMargin Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.source.dataset),
                            100. * batch_idx / len(self.source), margin_loss.item()))
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(self.source.dataset),
                #             100. * batch_idx / len(self.source), total_loss.item()))

    # test
    def test(self):
        self.clf.eval()
        test_loss = 0
        correct = 0
        classes = torch.unique(torch.tensor(np.arange(0, 3)))
        class_acc = torch.zeros(len(classes))
        class_len = torch.zeros(len(classes))
        with torch.no_grad():
            for data, target, path, _ in self.target_test:
                data, target = data.to(device), target.to(device)
                feature, output = self.clf(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                for class_idx, class_id in enumerate(classes):
                    idxes = torch.nonzero(target == class_id.to(target.device), as_tuple=False)
                    class_acc[class_idx] += pred[idxes].eq(target.view_as(pred)[idxes]).cpu().sum()

                    class_len[class_idx] += len(idxes)


        class_acc /= class_len
        avgAcc = 0.0

        for i in range(len(class_acc)):
            avgAcc += class_acc[i]
        avgAcc = avgAcc / len(class_acc)

        test_loss /= len(self.target_test.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Avg. Acc.: ({:.2f}%)\n'.format(
            test_loss, correct, len(self.target_test.dataset),
            100. * correct / len(self.target_test.dataset),
            100. * avgAcc))
        self.logger.info('-----------------------------------------------------------')
        return avgAcc


    def margin_base(self, probs):
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainty = probs_sorted[:, 0] - probs_sorted[:,1]
        return uncertainty.sort()[1].numpy()


    def entropy_base(self, probs):
        log_probs = torch.log(probs)
        uncertainty = (probs*log_probs).sum(1)
        return uncertainty.sort()[1].numpy()
    
    def least_confidence(self, probs):
        uncertainty = probs.max(1)[0]
        return uncertainty.sort()[1].numpy()

    def predict_prob2(self, model):
        model.eval()
        probs = torch.zeros([1, self.cfg.DATA_CLASS])
        logits = torch.zeros([1, self.cfg.DATA_CLASS])
        emb = torch.zeros([1, 256])
        with torch.no_grad():
            for x, y, _, idxs in self.target_test:
                x, y = Variable(x.to(device)), Variable(y.to(device))
                x = normalize(x)
                feature, output = model(x)
                prob = F.softmax(output, dim=1)
                probs = torch.cat([probs, prob.cpu().data],0)
                logits = torch.cat([logits, output.cpu().data],0)
                emb = torch.cat([emb, feature.cpu().data],0)
        return probs[1:], emb[1:], logits[1:]

    def STGADA_query(self,num_active):
        st_uncertainty, st_probs, start_time = self.pdf(self.clf)
        s_probs_sorted, _ = st_probs.sort(descending=True)
        # margin
        margin = s_probs_sorted[:, 0] - s_probs_sorted[:, 1]
        # uncertainty
        # uncertainty = self.entropy_base(st_probs)
        # uncertainty = self.least_confidence(st_probs)
        # uncertainty = margin
        uncertainty = margin - self.cfg.STGADA_LAMBDA * st_uncertainty
        uncertainty = uncertainty.sort()[1].numpy()
        #print(margin.sort()[1].numpy() == uncertainty)
        #print(uncertainty)
        chosen = uncertainty[:num_active]
        end_time = time.time()
        # print('total time : ', end_time-start_time)
        return chosen

    def pdf(self,model):
        # margin pdf
        s_probs, s_embs, logits = self.predict_prob2(model)
        s_weight = model.fc.weight.cpu()
        start_time = time.time()
        s_probs_sorted, s_pos_sorted = s_probs.sort(descending=True)
        s_pos_weight = s_weight[s_pos_sorted]
        s_pmax1 = s_probs_sorted[:,0].reshape(s_pos_sorted.shape[0],1)
        s_pmax2 = s_probs_sorted[:,1].reshape(s_pos_sorted.shape[0],1)
        s_max1 = (s_pmax1 * (1-s_pmax1)) * s_pos_weight[:,0,:]
        s_max2 = (s_pmax2 * (1-s_pmax2)) * s_pos_weight[:,1,:]
        s_Q_tensor = s_max1 - s_max2 \
        - torch.sum((s_probs_sorted[:,2:].unsqueeze(-1) * s_pos_weight[:,2:,:]),dim=1) \
        * ((s_probs_sorted[:, 0] - s_probs_sorted[:, 1]).unsqueeze(1))
        # Loss pdf
        s_embs = s_embs.requires_grad_()
        fc = deepcopy(model.fc).cpu()
        s_logits = fc(s_embs)
        max1_pseudo = s_pos_sorted[:,0]
        max2_pseudo = s_pos_sorted[:,1]
        ce_loss1 = F.cross_entropy(s_logits, max1_pseudo)
        ce_loss2 = F.cross_entropy(s_logits, max2_pseudo)
        margin_loss1 = F.multi_margin_loss(F.normalize(s_logits),max1_pseudo)
        margin_loss2 = F.multi_margin_loss(F.normalize(s_logits),max2_pseudo)
        label1 = torch.unsqueeze(max1_pseudo,dim=1)
        label2 = torch.unsqueeze(max2_pseudo,dim=1)
        onehot_label1 = torch.zeros_like(s_logits).scatter_(1,label1.long(),1)
        onehot_label2 = torch.zeros_like(s_logits).scatter_(1,label2.long(),1)
        addition_loss1 = (F.normalize(s_logits) * onehot_label1).sum() / s_pos_sorted.shape[0]
        addition_loss2 = (F.normalize(s_logits) * onehot_label2).sum() / s_pos_sorted.shape[0]
        loss1 = ce_loss1 + margin_loss1 - addition_loss1
        loss2 = ce_loss2 + margin_loss2 - addition_loss2
        grad1 = -torch.autograd.grad(outputs=loss1,inputs=s_embs,retain_graph=True)[0]
        grad2 = -torch.autograd.grad(outputs=loss2,inputs=s_embs,retain_graph=True)[0]
        grad = s_probs_sorted[:,0].unsqueeze(-1) * grad1 + s_probs_sorted[:,1].unsqueeze(-1) * grad2
        # uncertainty
        uncertainty = torch.cosine_similarity(s_Q_tensor,grad)
        return uncertainty, s_probs, start_time
