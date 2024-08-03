# Environment Semantic Communication: Enabling Distributed Sensing Aided Networks

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communication systems require large antenna arrays and use narrow directive beams to ensure sufficient receive signal power. However, selecting the optimal beams for these large antenna arrays incurs a significant beam training overhead, making it challenging to support applications involving high mobility. In recent years, machine learning (ML) solutions have shown promising results in reducing the beam training overhead by utilizing various sensing modalities such as GPS position and RGB images. However, the existing approaches are mainly limited to scenarios with only a single object of interest present in the wireless environment and focus only on co-located sensing, where all the sensors are installed at the communication terminal. This brings key challenges such as the limited sensing coverage compared to the coverage of the communication system and the difficulty in handling non-line-of-sight scenarios. To overcome these limitations, our paper proposes the deployment of multiple distributed sensing nodes, each equipped with an RGB camera. These nodes focus on extracting environmental semantics from the captured RGB images. The semantic data, rather than the raw images, are then transmitted to the basestation. This strategy significantly alleviates the overhead associated with the data storage and transmission of the raw images. Furthermore, semantic communication enhances the system’s adaptability and responsiveness to dynamic environments, allowing for prioritization and transmission of contextually relevant information. Experimental results on the DeepSense 6G dataset demonstrate the effectiveness of the proposed solution in reducing the sensing data transmission overhead while accurately predicting the optimal beams in realistic communication environments.


**Downloading Dataset and Code** 
1. Download Scenario 41 of Deepsense6G dataset.
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory.
4. The bounding boxes and mask from Yolov7 can be generated using 'Yolov7_bbox_and_masks_generating_code.ipynb' in the repository https://github.com/convexoptimist/Environment-Semantic-Aided-Communication/tree/main


**Code Package Content**
1. Run FCNN_on_BBoxes\unit3\main_pos_beam.py
2. Run FCNN_on_BBoxes\unit4\main_pos_beam.py
3. Run LeNet_on_Masks\unit3\main.py
4. Run LeNet_on_Masks\unit4\main.py



# License and Referencing
This code package is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. If you in any way use this code for research that results in publications, please cite our original article:

Imran, Shoaib, Gouranga Charan, and Ahmed Alkhateeb. "Environment Semantic Communication: Enabling Distributed Sensing Aided Networks." arXiv preprint arXiv:2402.14766 (2024).

If you use the DeepSense 6G dataset, please also cite our dataset article:

A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan, and N. Srinivas, “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and     Communication Dataset,” arXiv preprint arXiv:2211.09769 (2022) [Online]. Available: https://www.DeepSense6G.net





