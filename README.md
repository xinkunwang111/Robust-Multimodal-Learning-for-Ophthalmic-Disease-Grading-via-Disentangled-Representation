# Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation
Ophthalmologists often rely on multimodal data to improve diagnostic accuracy, but obtaining complete datasets is challenging due to limited medical equipment and privacy concerns. Traditional deep learning methods typically address these issues by learning representations in latent space. However, two main limitations exist in current approaches: (i) Task-irrelevant redundant information from complex modalities, such as numerous slices, leads to excessive redundancy in latent space representations, and (ii) overlapping multimodal representations make it difficult to extract unique features for each modality. To overcome these challenges, the Essence-Point and Disentangle Representation Learning (EDRL) strategy is introduced. This strategy integrates a self-distillation mechanism into an end-to-end framework to improve feature selection and disentanglement for better multimodal learning. Specifically, the Essence-Point Representation Learning module selects discriminative features that enhance disease grading performance, while the Disentangled Representation Learning module separates multimodal data into common and unique representations for each modality, reducing feature overlap and improving both robustness and interpretability in ophthalmic disease diagnosis. Experimental results on multimodal ophthalmology datasets show that the proposed EDRL strategy significantly outperforms state-of-the-art methods.




![image](https://github.com/user-attachments/assets/6c1e772e-9a77-4a4e-91bb-1187a9be753e)

### 1.dataset
Harvard 30K 
https://yutianyt.com/projects/fairvision30k/

### 2. Run
Train the end-to-end framework.
`./Run_fusion`.

 Test the checkpoint.
`./Run_test`

### 3. Environment
see the requirement.txt 
