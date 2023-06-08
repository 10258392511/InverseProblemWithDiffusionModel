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
<table align="center" id="NCSN-3D-unconditional">
    <tbody align="center">
        <tr>
            <td></td>
            <td><figure><img src="readme_images/GT.gif" alt="CINE127 GT blocks"> <br> <figurecaption>GT</figurecaption></figure></td>
            <td></td>
        </tr>
        <tr>
            <td><figure> <img src="readme_images/lr_1e-4.gif" alt="sample with lr 1e-4"> <br> <figurecaption>$\alpha = 10^{-4}$</figurecaption></figure></td>
            <td><figure> <img src="readme_images/lr_5e-5.gif" alt="sample with lr 5e-5"> <br> <figurecaption>$\alpha = 5 \times 10^{-5}$</figurecaption></figure></td>
            <td><figure> <img src="readme_images/lr_1_75e-5.gif" alt="sample with lr 1.75e-4"> <br> <figurecaption>$\alpha = 1.75 \times 10^{-5}$</figurecaption></figure></td>
        </tr>
        <tr>
            <td><figure> <img src="readme_images/lr_1_5e-5.gif" alt="sample with lr 1.5e-5"> <br> <figurecaption>$\alpha = 1.5 \times 10^{-5}$</figurecaption></figure></td>
            <td><figure> <img src="readme_images/lr_1_25e-5.gif" alt="sample with lr 1.25e-4"> <br> <figurecaption>$\alpha = 1.25 \times 10^{-5}$</figurecaption></figure></td>
            <td><figure> <img src="readme_images/lr_1e-5.gif" alt="sample with lr 1e-5"> <br> <figurecaption>$\alpha = 10^{-5}$</figurecaption></figure></td>
        </tr>
    </tbody>
    
    
</table>
<div align="center">
    <strong>CINE127 temporal unconditional real / imaginary samples with different ALD learning rate &alpha;.</strong>
</div>

<br>
<table align="center" id="NCSN-3D-results">
    <tr>
        <th>R</th>
        <th>Algorithm</th>
        <th>Mag<span style="color: white;">.</span></th>
        <th >Phase</th>
        <th>Mag<span style="color: white;">.</span> @ T / 2</th>
        <th>Phase @ T / 2</th>
        <th>Mag<span style="color: white;">.</span> @ H / 2</th>
        <th>NRMSE</th>
        <th>SSIM</th>
    </tr>
    <tbody align="center">
        <tr>
            <td>8</td>
            <td>Original</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/original/mag.gif" alt="original mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/original/phase.gif" alt="original phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/original/half_T_mag.png" alt="original mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/original/half_T_phase.png" alt="original phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/original/half_H_mag.png" alt="original mag at half H" height=128></td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>8</td>
            <td>ZF</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ZF/mag.gif" alt="ZF mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ZF/phase.gif" alt="ZF phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ZF/half_T_mag.png" alt="ZF mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ZF/half_T_phase.png" alt="ZF phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ZF/half_H_mag.png" alt="ZF mag at half H" height=128></td>
            <td>0.439</td>
            <td>0.502</td>
        </tr>
        <tr>
            <td>8</td>
            <td>ALD Best</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ALD/mag.gif" alt="ALD best mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ALD/phase.gif" alt="ALD best phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ALD/half_T_mag.png" alt="ALD best mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ALD/half_T_phase.png" alt="ALD best phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/ALD/half_H_mag.png" alt="ALD best mag at half H" height=128></td>
            <td>0.073</td>
            <td>0.953</td>
        </tr>
        <tr>
            <td>8</td>
            <td>MAP Best</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/MAP/mag.gif" alt="MAP best mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/MAP/phase.gif" alt="MAP best phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/MAP/half_T_mag.png" alt="MAP best mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/MAP/half_T_phase.png" alt="MAP best phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_8/MAP/half_H_mag.png" alt="MAP best mag at half H" height=128></td>
            <td>0.091</td>
            <td>0.955</td>
        </tr>
        <tr>
            <td>16</td>
            <td>Original</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/original/mag.gif" alt="original mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/original/phase.gif" alt="original phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/original/half_T_mag.png" alt="original mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/original/half_T_phase.png" alt="original phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/original/half_H_mag.png" alt="original mag at half H" height=128></td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>16</td>
            <td>ZF</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ZF/mag.gif" alt="ZF mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ZF/phase.gif" alt="ZF phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ZF/half_T_mag.png" alt="ZF mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ZF/half_T_phase.png" alt="ZF phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ZF/half_H_mag.png" alt="ZF mag at half H" height=128></td>
            <td>0.512</td>
            <td>0.412</td>
        </tr>
        <tr>
            <td>16</td>
            <td>ALD Best</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ALD/mag.gif" alt="ALD best mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ALD/phase.gif" alt="ALD best phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ALD/half_T_mag.png" alt="ALD best mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ALD/half_T_phase.png" alt="ALD best phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/ALD/half_H_mag.png" alt="ALD best mag at half H" height=128></td>
            <td>0.141</td>
            <td>0.865</td>
        </tr>
        <tr>
            <td>16</td>
            <td>MAP Best</td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/MAP/mag.gif" alt="MAP best mag"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/MAP/phase.gif" alt="MAP best phase"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/MAP/half_T_mag.png" alt="MAP best mag at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/MAP/half_T_phase.png" alt="MAP best phase at half T"></td>
            <td><img src="readme_images/qualitative_2d_time_readme/R_16/MAP/half_H_mag.png" alt="MAP best mag at half H" height=128></td>
            <td>0.161</td>
            <td>0.871</td>
        </tr>
    </tbody>
</table>

<div align="center">
    <strong>Results on cardiac MR data. Our algorithms reconstruct image very close to GT under moderate acceleration rate; and still work very well under very high acceleration rate. </strong>
</div>
