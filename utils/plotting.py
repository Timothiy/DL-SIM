import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity

plt.switch_backend('agg')

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()


def testAndMakeCombinedPlots(net, loader, opt, idx=0):
    def PSNR_numpy(p0, p1):
        I0, I1 = np.array(p0) / 255.0, np.array(p1) / 255.0
        MSE = np.mean((I0 - I1) ** 2)
        PSNR = 20 * np.log10(1 / np.sqrt(MSE))
        return PSNR

    def SSIM_numpy(p0, p1):
        I0, I1 = np.array(p0) / 255.0, np.array(p1) / 255.0
        return structural_similarity(I0, I1, multichannel=True)

    def calcScores(img, hr=None, makeplotBool=False, plotidx=0, title=None):
        if makeplotBool:
            plt.subplot(1, 3, plotidx)
            plt.gca().axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img, cmap='gray')
        if not hr == None:
            psnr, ssim = PSNR_numpy(img, hr), SSIM_numpy(img, hr)
            if makeplotBool:
                plt.title('%s (%0.2fdB/%0.3f)' % (title, psnr, ssim))
            return psnr, ssim
        if makeplotBool:
            plt.title(r'GT ($\infty$/1.000)')

    count, mean_wf_psnr, mean_sr_psnr, mean_wf_ssim, mean_sr_ssim = 0, 0, 0, 0, 0

    for i, batch in enumerate(loader):
        lr, hr, wf = batch['sim_inputs'], batch['sim_gt'], batch['wf']
        b, c, w, h = hr.shape
        with torch.no_grad():
            sr = net(lr.to(opt.device))
        for j in range(len(lr)):  # loop over batch
            makeplotBool = (idx < 5 or (idx + 1) % opt.plotinterval == 0 or idx == opt.nepoch - 1) and count < opt.nplot
            if opt.test:
                makeplotBool = True

            sr, hr, lr = sr.data[j], hr.data[j], wf.data[j]
            sr = torch.clamp(sr, min=0, max=1)
            # gradient_map = torch.clamp(gradient_map, min=0, max=1)
            ### Common commands
            lr, sr, hr = toPIL(lr), toPIL(sr), toPIL(hr)

            if opt.scale == 2:
                lr = lr.resize((w, h), resample=Image.BICUBIC)

            if makeplotBool:
                plt.figure(figsize=(10, 5), facecolor='white')
            wf_psnr, wf_ssim = calcScores(lr, hr, makeplotBool, plotidx=1, title='WF')
            sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=2, title='SR')
            calcScores(hr, None, makeplotBool, plotidx=3)

            mean_wf_psnr += wf_psnr
            mean_sr_psnr += sr_psnr
            mean_wf_ssim += wf_ssim
            mean_sr_ssim += sr_ssim

            if makeplotBool:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.savefig('%s/combined_epoch%d_%d.tif' % (opt.out, idx + 1, count), dpi=300, bbox_inches='tight',
                            pad_inches=0)
                # gradient_map.save("%s/gradient_epoch%d_%d.tif" % (opt.out, idx + 1, count))
                plt.close()
                if opt.test:
                    lr.save('%s/lr_epoch%d_%d.tif' % (opt.out, idx + 1, count))
                    sr.save('%s/sr_epoch%d_%d.tif' % (opt.out, idx + 1, count))
                    hr.save('%s/hr_epoch%d_%d.tif' % (opt.out, idx + 1, count))

            count += 1
            if count == opt.ntest:
                break
        if count == opt.ntest:
            break

    summarystr = ""
    if count == 0:
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. wf: %0.2f dB / %0.4f, sr: %0.2f dB / %0.4f' % (
        count, mean_wf_psnr / count, mean_wf_ssim / count, mean_sr_psnr / count, mean_sr_ssim / count)
    print(summarystr)
    print(summarystr, file=opt.fid)
    opt.fid.flush()
    if opt.log and not opt.test:
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx, t1, mem, mean_sr_psnr / count, mean_sr_ssim / count, file=opt.test_stats)
        opt.test_stats.flush()


def generate_convergence_plots(opt, filename):
    fid = open(filename, 'r')
    psnrlist = []
    ssimlist = []
    losslist = []
    for line in fid:
        if 'psnr: ' in line:
            psnrlist.append(float(line.split(',')[2].split('psnr: ')[1]))
        if 'ssim: ' in line:
            ssimlist.append(float(line.split(',')[3].split('ssim: ')[1]))
        if 'total loss: ' in line:
            losslist.append(float(line.split(",")[1].split("total loss: ")[1]))

    plt.figure(figsize=(12, 5), facecolor='white')
    plt.subplot(131)
    plt.plot(psnrlist, '.-')
    plt.title('PSNR')

    plt.subplot(132)
    plt.plot(ssimlist, '.-')
    plt.title('SSIM')

    plt.subplot(133)
    plt.plot(losslist, '.-')
    plt.title('Total Loss')

    plt.savefig('%s/{}.png'.format(opt.model) % opt.out, dpi=300)
