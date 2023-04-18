import torch
import sys
sys.path.append('..')
from util.utils import *


class FDA(torch.nn.Module):
    def __init__(self, l=0.1):
        super().__init__()

        self.l = l

    def _extract_ampl_phase(self, fft_im):
        # fft_im: size should be bx3xhxwx2
        fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
        return fft_amp, fft_pha

    def _low_freq_mutate(self, amp_src, amp_trg, L=0.1):
        # _, _, h, w = amp_src.size()
        _, _, h, w = amp_trg.size()
        b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # top left
        amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]  # top right
        amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]  # bottom left
        amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h, w - b:w]  # bottom right
        return amp_src

    def forward(self, src_img, trg_img):
        # exchange magnitude
        # input: src_img, trg_img

        # get fft of both source and target
        # fft_src = torch.rfft(src_img.clone(), signal_ndim=2, onesided=False)
        # fft_trg = torch.rfft(trg_img.clone(), signal_ndim=2, onesided=False)
        # fft_src = torch.fft.fft2(src_img.clone())
        # fft_trg = torch.fft.fft2(trg_img.clone())
        fft_src = torch.fft.rfft2(src_img.clone())
        fft_trg = torch.fft.rfft2(trg_img.clone())

        # extract amplitude and phase of both ffts
        amp_src, pha_src = self._extract_ampl_phase(torch.view_as_real(fft_src).clone())
        amp_trg, pha_trg = self._extract_ampl_phase(torch.view_as_real(fft_trg).clone())

        # replace the low frequency amplitude part of source with that from target
        amp_src_ = self._low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=self.l)

        # recompose fft of source
        fft_src_ = torch.zeros(torch.view_as_real(fft_src).size(), dtype=torch.float, device=fft_src.device)
        fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
        fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

        # get the recomposed image: source content, target style
        _, _, imgH, imgW = src_img.size()
        # src_in_trg = torch.irfft(fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])
        # src_in_trg = torch.fft.ifft2(fft_src_, s=[imgH, imgW])
        fft_src_ = torch.complex(fft_src_[..., 0], fft_src_[..., 1])
        src_in_trg = torch.fft.irfft2(fft_src_)

        return src_in_trg