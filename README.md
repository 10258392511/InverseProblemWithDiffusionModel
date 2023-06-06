# Inverse Problem with Score-Based Diffusion Model
Please first install `PyTorch` following the [official website](https://pytorch.org/). Then please install other 
dependencies by:
```bash
git clone https://github.com/10258392511/InverseProblemWithDiffusionModel
cd InverseProblemWithDiffusionModel
pip3 install -e .
```
### NCSN for 2D Complex-Valued MR Image Reconstruction
<figure>
    <img src="readme_images/acdc_summary.gif" alt="ACDC Summary">
    <figcaption><strong>Comparing ALD reconstruction with MAP and TV reconstruction. We computed the mean of 105 reconstructions and it outperforms distinctively over MAP and TV reconstruction under the high acceleration rate R = 40 with 4 coils.</strong></figcaption>
</figure>

### NCSN for 2D + Time Cardiac MR Reconstruction
| |<figure><img src="readme_images/GT.gif" alt="CINE127 GT blocks"> <figurecaption>GT</figurecaption></figure>| |
| :----: | :----: | :-----: |
|<figure> <img src="readme_images/lr_1e-4.gif" alt="sample with lr 1e-4"> <figurecaption>$\alpha = 10^{-4}$</figurecaption></figure>|<figure> <img src="readme_images/lr_5e-5.gif" alt="sample with lr 5e-5"> <figurecaption>$\alpha = 5 \times 10^{-5}$</figurecaption></figure>|<figure> <img src="readme_images/lr_1_75e-5.gif" alt="sample with lr 1.75e-4"> <figurecaption>$\alpha = 1.75 \times 10^{-5}$</figurecaption></figure>|
|<figure> <img src="readme_images/lr_1_5e-5.gif" alt="sample with lr 1.5e-5"> <figurecaption>$\alpha = 1.5 \times 10^{-5}$</figurecaption></figure>|<figure> <img src="readme_images/lr_1_25e-5.gif" alt="sample with lr 1.25e-4"> <figurecaption>$\alpha = 1.25 \times 10^{-5}$</figurecaption></figure>|<figure> <img src="readme_images/lr_1e-5.gif" alt="sample with lr 1e-5"> <figurecaption>$\alpha = 10^{-5}$</figurecaption></figure>|
<figurecaption><strong>CINE127 temporal unconditional real / imaginary samples with different ALD learning rate $\alpha$. $\alpha$ should be carefully tuned, otherwise meaningful samples can be missed.</strong></figurecaption>
