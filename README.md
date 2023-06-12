# Driver-Workload-Recognition

This dataset is available for research purposes, and the files could be downloaded from the [download page](https://entuedu-my.sharepoint.com/:u:/g/personal/haohan_yang_staff_main_ntu_edu_sg/Eb3HVEY1ufhFs7saFVAqGUgBjyP8ZzUAIt-62PWGqB-zGQ?e=CzBa88).

## Terms & Conditions
- The dataset is the sole property of the AutoMan group at the Nanyang Technological University and is protected by copyright. The dataset shall remain the exclusive property of AutoMan.
- The End User acquires no ownership, rights, or title of any kind in all or parts regarding the dataset.
- Any commercial use of the dataset is strictly prohibited. Commercial use includes, but is not limited to: testing commercial systems; using screenshots of subjects from the dataset in advertisements, selling data or making any commercial use of the dataset, broadcasting data from the dataset.
- The End User shall not, without prior authorization of the AutoMan group, transfer in any way, permanently or temporarily, distribute or broadcast all or part of the dataset to third parties.
- The End User shall send all requests for the distribution of the dataset to the AutoMan group.
- All publications that report on research that uses the dataset should cite our publications.

H. Yang, J. Wu, Z. Hu, and C. Lv, "Real-Time Driver Cognitive Workload Recognition: Attention-Enabled Learning With Multimodal Information Fusion", IEEE Transactions on Industrial Eletronics, 2023.

## 1. Experimental Platform
  * EEG headset [(EMOTIVE EPOC Flex)](https://www.emotiv.com/)   
    Ensure good contact quality and EEG quality during the experiments.
    
    <img src="https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/contact%20quality.jpg" width="300" alt="">
    <img src="https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/EEG%20quality.jpg" width="300" alt="">
    <img src="https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/EEG%20signals.jpg" width="310" alt="">
    
  * Eye tracker [(Tobii Pro X120)](https://www.tobiipro.com/)  
    Calibrate eye gaze position  
    
    <img src="https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/calibration.jpg" width="350" alt="">
        
  * Simulator [(Carla 0.9.8)](http://carla.org/)    
  
## 2. Real-Time Validation (Demo)
Note: data streams with the z-score standardization are presented in the demo video
[![Watch the video](https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/demo.jpg)](https://youtu.be/E0blk93KIK4)

## 3. Supplemental Results (Without EEG Signals)
Current wearable EEG equipment inevitably affects driver operations to some extent. Therefore, we have carried out additional experiments for the performance comparison of various methods without using EEG signals, and the average results as well as standard deviations across the 5-fold cross validation are presented below, (a) Sunny noon. (b) Foggy dusk. (c) Rainy night.

<img src="https://github.com/yhh-IV/Driver-Workload-Recognition/blob/main/images/comparison.png" width="300" alt="">

It can be concluded that
- The recognition accuracy of the proposed method significantly surpasses other baselined models in all situations, and its standard deviation is lower than others in most cases.
- The recognition accuracy of a specific model increases with extended historical horizons, which is intuitive since a longer historical horizon generally contains more information.

Except for an overall decrease in the recognition accuracy of the model compared to when it has EEG signals, there is almost no change in the conclusion. The phenomenon indicates that the proposed model has a stronger representational capacity in a general sense.
    
   
 
   

