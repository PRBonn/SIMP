# SIMP: Long-Term Indoor Localization with Metric-Semantic Mapping using a Floor Plan Prior
 This repository contains the implementation of the following publication:


## Abstract
Object-based maps are relevant for scene understanding since they integrate geometric and semantic information of the environment, allowing autonomous robots to robustly localize and interact with on objects. In this paper, we address the task of constructing a metric-semantic map for the purpose of long-term object-based localization. 
We exploit 3D object detections from monocular RGB frames for both, the object-based map construction, and for globally localizing in the constructed map. To tailor the approach to a target environment, we propose an efficient way of generating 3D annotations to finetune the 3D object detection model.
We evaluate our map construction in an office building, and test our long-term localization approach on challenging sequences recorded in the same environment over nine months. The experiments suggest that our approach is suitable for constructing metric-semantic maps, and that our localization approach is robust to long-term changes. 
Both, the mapping algorithm and the localization pipeline can run online on an onboard computer. 
We will release an open-source C++/ROS implementation of our approach.


## Results
Our live demos for both localization and mapping on the Dingo-O platform can seen in the following video:
[![](http://img.youtube.com/vi/z5VKtl3Vyyw/0.jpg)](https://www.youtube.com/watch?v=z5VKtl3Vyyw "SIMP")

The Dingo-O is mounted with Intel NUC10i7FNK and NVidia Jetson Xavier AGX, both running Ubuntu 20.04. Both mapping and localization can be executed onboard of the Dingo.

For the quantitative evaluation, we collected data for 9 months, traversing our entire floor. Our data collection platform is a Kuka YouBot, and we recorded 4 RGB camera streams, ground truth poses and wheel odometry. We used a 30-minutes long sequence to construct an object-based map of our lab, enriching our floor plan with geometric and semantic information. We then localized on challenging sequences, spanning over 9 months, which include dynamics such a humans and changes to the furniture. 
A visualization of our 3D metric-semantic map, and qualitative results for the localization approach evaluated on our map
<p align="center">
<img src="https://github.com/PRBonn/SIMP/blob/master/resources/3dmap.png" width="384"/>
<img src="https://github.com/PRBonn/SIMP/blob/master/resources/traj.png" width="384"/>
</p>

## Code
Coming soon!

## Citation
Coming soon!


