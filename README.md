# SIMP: Long-Term Indoor Localization with Metric-Semantic Mapping using a Floor Plan Prior
 This repository contains the implementation of the following publication:


## Abstract
Object-based maps are relevant for scene understanding since they integrate geometric and semantic information of the environment, allowing autonomous robots to robustly localize and interact with on objects. 

In this paper, we address the task of constructing a metric-semantic map for the purpose of long-term object-based localization. 

We exploit 3D object detections from monocular RGB frames for both, the object-based map construction, and for globally localizing in the constructed map. To tailor the approach to a target environment, we propose an efficient way of generating 3D annotations to finetune the 3D object detection model.
We evaluate our map construction in an office building, and test our long-term localization approach on challenging sequences recorded in the same environment over nine months. The experiments suggest that our approach is suitable for constructing metric-semantic maps, and that our localization approach is robust to long-term changes. 
Both, the mapping algorithm and the localization pipeline can run online on an onboard computer. 
We will release an open-source C++/ROS implementation of our approach.


## Results
Our live demos for both localization and mapping on the Dingo-O platform can seen in the following video:
[![](http://img.youtube.com/vi/z5VKtl3Vyyw/0.jpg)](https://www.youtube.com/watch?v=z5VKtl3Vyyw "SIMP")

A visualization of our 3D metric-semantic map constructed based on recordings from our building
<img src="https://github.com/PRBonn/SIMP/blob/master/resources/3dmap.png" width="512"/>
<!-- ![3Dmap](https://github.com/PRBonn/SIMP/blob/master/resources/3dmap.png) -->  

Qualitative results for the localization approach evaluated on the constructed map above
![traj](https://github.com/PRBonn/SIMP/blob/master/resources/traj.png)


## Code
Coming soon!

## Citation
Coming soon!


