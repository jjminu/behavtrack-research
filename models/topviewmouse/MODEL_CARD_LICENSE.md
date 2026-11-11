---
tags:
- computer_vision
- pose_estimation
- animal_pose_estimation
- deeplabcut
pipeline_tag: keypoint-detection
---
# MODEL CARD:

## Model Details

• SuperAnimal-TopViewMouse model developed by the [M.W.Mathis Lab](http://www.mackenziemathislab.org/) in 2023, 
trained to predict mouse topline pose from images. 
Please see [Shaokai Ye et al. 2023](https://arxiv.org/abs/2203.07436) for details.

• The there are three models:
  - `pose_model.pth` is an HRNet-w32 compatable for DLC3.0+ Pytorch code, trained on our TopViewMouse-5K dataset.
  - `detector.pt` is a Faster R-CNN that can be used as a detector for top-down detection.
  - `DLC_ma_supertopview5k_resnet_50_iteration-0_shuffle-1.tar.gz` is a DLCRNet trained on our TopViewMouse-5K dataset.


•  Full training details can be found in Ye et al. 2023.
You can use this model simply with our light-weight loading package called [DLCLibrary](https://github.com/DeepLabCut/DLClibrary). 
Here is an example useage:

```python
from pathlib import Path
from dlclibrary import download_huggingface_model

# Creates a folder and downloads the model to it
model_dir = Path("./superanimal_topviewmouse_model")
model_dir.mkdir()
download_huggingface_model("superanimal_topviewmouse_model", model_dir)
```

## Intended Use

• Intended to be used for pose tracking of lab mice videos filmed from an overhead view. The models can be used as a plug-and-
play solution if extremely high precision is not required (we benchmark the zero-shot performance in the paper). Otherwise, it is
recommended to also be used as the weights for transfer learning and fine-tuning.

• Intended for academic and research professionals working in fields related to animal behavior, neuroscience, biomechanics, and
ecology.

• Not suitable for other species and other camera views. Also not suitable for videos that look dramatically different from those we
show in the paper.

## Factors

• Based on the known robustness issues of neural networks, the relevant factors include the lighting, contrast and resolution of the
video frames. The present of objects might also cause false detections of the mice and keypoints. When two or more animals are
extremely close, it could cause the top-down detectors to only detect only one animal, if used without further fine-tuning.


## Metrics
• Mean Average Precision (mAP)

• Root Mean Square Error (RMSE)

## Evaluation Data

• The test split of TopViewMouse-5K and in the paper on two benchmarks, DLC Openfield and TriMouse


## Training Data

It consists of being trained together on the following datasets:

- **3CSI, BM, EPM, LDB, OFT** See full details at (1) and in (2). 

- **BlackMice** See full details at (3).

- **WhiteMice** Courtesy of Prof. Sam Golden and Nastacia Goodwin. See details in SIMBA (4). TriMouse See full details
at (5). 

- **DLC-Openfield** See full details at (6). 

- **Kiehn-Lab-Openfield, Swimming, and treadmill** Courtesy of Prof. Ole
Kiehn, Dr. Jared Cregg, and Prof. Carmelo Bellardita; see details at (7). 

- **MausHaus** We collected video data from five
single-housed C57BL/6J male and female mice in an extended home cage, carried out in the laboratory of Mackenzie Mathis
at Harvard University and also EPFL (temperature of housing was 20-25C, humidity 20-50%). Data were recorded at 30Hz
with 640 × 480 pixels resolution acquired with White Matter, LLC eV cameras. Annotators localized 26 keypoints across 322
frames sampled from within DeepLabCut using the k-means clustering approach (8). All experimental procedures for mice
were in accordance with the National Institutes of Health Guide for the Care and Use of Laboratory Animals and approved by
the Harvard Institutional Animal Care and Use Committee (IACUC) (n=1 mouse), and by the Veterinary Office of the Canton
of Geneva (Switzerland; license GE01) (n=4 mice).

Here is an image with examples from the datasets, the distribution of images per dataset, and the keypoint guide.

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1690986892069-I1DP3EQU14DSP5WB6FSI/modelcard-TVM.png?format=1500w" width="95%">
</p>

## Ethical Considerations

• Data was collected with IUCAC or other governmental approval. Each individual dataset used in training reports the ethics approval
they obtained.

## Caveats and Recommendations

• The model may have reduced accuracy in scenarios with extremely varied lighting conditions or atypical mouse characteristics not
well-represented in the training data. For example, this dataset only has one set of white mice, therefore it may not generalize well
to diverse settings of white lab mice.

• Please note that each training dataset was labeled by separate labs and different individuals, therefore while we map names to a
unified pose vocabulary, there will be annotator bias in keypoint placement (See Ye et al. 2023 for our Supplementary Note on
annotator bias).

• Note the dataset is primarily using C56Blk6/J mice and only some CD1 examples.

• We recommend if performance is not as good as you need it to be, first try video adaptation (see Ye et al. 2023), or fine-tune these
weights with your own labeling.

## License

Modified MIT.

Copyright 2023 by Mackenzie Mathis, Shaokai Ye, and contributors. 

Permission is hereby granted to you (hereafter "LICENSEE") a fully-paid, non-exclusive,
and non-transferable license for academic, non-commercial purposes only (hereafter “LICENSE”)
to use the "MODEL" weights (hereafter "MODEL"), subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software:

This software may not be used to harm any animal deliberately.

LICENSEE acknowledges that the MODEL is a research tool. 
THE MODEL IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL
OR THE USE OR OTHER DEALINGS IN THE MODEL.

If this license is not appropriate for your application, please contact Prof. Mackenzie W. Mathis 
(mackenzie@post.harvard.edu) and/or the TTO office at EPFL (tto@epfl.ch) for a commercial use license.

Please cite **Ye et al** if you use this model in your work https://arxiv.org/abs/2203.07436v2.

## References

1. Oliver Sturman, Lukas von Ziegler, Christa Schläppi, Furkan Akyol, Mattia Privitera, Daria Slominski, Christina Grimm, Laetitia Thieren, Valerio
Zerbi, Benjamin Grewe, et al. Deep learning-based behavioral analysis reaches human accuracy and is capable of outperforming commercial
solutions. Neuropsychopharmacology, 45(11):1942–1952, 2020.
2. Lukas von Ziegler, Oliver Sturman, and Johannes Bohacek. Videos for deeplabcut, noldus ethovision X14 and TSE multi conditioning systems
comparisons. https://doi.org/10.5281/zenodo.3608658. Zenodo, January 2020.
3. Isaac Chang. Trained DeepLabCut model for tracking mouse in open field arena with topdown view. https://doi.org/10.5281/zenodo.3955216.
Zenodo, July 2020.
4. Simon RO Nilsson, Nastacia L. Goodwin, Jia Jie Choong, Sophia Hwang, Hayden R Wright, Zane C Norville, Xiaoyu Tong, Dayu Lin, Bran-
don S. Bentzley, Neir Eshel, Ryan J McLaughlin, and Sam A. Golden. Simple behavioral analysis (simba) – an open source toolkit for computer
classification of complex social behaviors in experimental animals. bioRxiv, 2020.
5. Jessy Lauer, Mu Zhou, Shaokai Ye, William Menegas, Steffen Schneider, Tanmay Nath, Mohammed Mostafizur Rahman, Valentina Di Santo,
Daniel Soberanes, Guoping Feng, Venkatesh N. Murthy, George Lauder, Catherine Dulac, Mackenzie W. Mathis, and Alexander Mathis. Multi-
animal pose estimation, identification and tracking with deeplabcut. Nature Methods, 19:496 – 504, 2022.
6. Alexander Mathis, Pranav Mamidanna, Kevin M Cury, Taiga Abe, Venkatesh N Murthy, Mackenzie Weygandt Mathis, and Matthias Bethge. Deeplab-
cut: markerless pose estimation of user-defined body parts with deep learning. Nature neuroscience, 21:1281–1289, 2018.
7. Jared M. Cregg, Roberto Leiras, Alexia Montalant, Paulina Wanken, Ian R. Wickersham, and Ole Kiehn. Brainstem neurons that command
mammalian locomotor asymmetries. Nature neuroscience, 23:730 – 740, 2020