![Cluster&Probability](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/New_1.png)

This is the repository for identifying bubble cluster configurations to produce the probability distribution.

# Progress:
![RICNNv1_12/05/2017](https://github.com/chinchangkuo/RICNN_Cluster_classification/blob/master/RICNNv1.md)

![RICNNv2_02/02/2018](https://github.com/chinchangkuo/RICNN_Cluster_classification/blob/master/RICNNv2.md)

# Motivation:

In condensed matter physics one of the most important and open questions involves understanding the mechanical response of materials. With different structural features, solid materials can be classified into two main groups: crystalline materials and amorphous materials. Crystalline materials refer to substances that are arranged in a highly ordered microscopic structure. This type of material is often very rare in the natural environment, an example of which are diamonds. In the crystalline case, the mechanical response is well described by theoretical models, due to their highly ordered structure. One can easily imagine that the deformation of the crystalline structure under an applied force tends to occur near structural defects. This is because the surrounding local structure is relatively less stable. However, most materials exist in the nature are considered to be amorphous. In contrast to their crystalline counterpart, there is no long-range structural ordering. This is due to their intrinsic structural complexity, which makes it challenging to precisely predict the mechanical response under an applied force, especially as it pertains to understanding the structural transformation at the microscale. 

One of the methods used to study this essential question is to decompose an amorphous structure into sub-units (clusters) based on their stability. In the analog to the crystalline defects, recognizing the less stable local cluster is equivalent to identifying the weak zone in the larger structure. From here we can begin to understand the relationship between the structural feature and the deformation of amorphous materials under an applied force. Thus studying the stability of clusters is crucial to further our understanding of this behavior. 

In our experiment, we use bubbles to generate clusters at the air-water interface to study their stability. Bubble rafts have been used as a model system in many past research projects, due to the similarity of the particle-particle based attraction to atomic interaction[1]. Also, it provides us the capability to directly observe the microstructure which is very rare in other systems. With the benefit of using bubbles to create clusters, we are able to collect massive data sets in the form of images to analyze their stability in a controlled laboratory setting.

[1] 	L. Bragg and J. F. Nye, "A Dynamical Model of a Crystal Structure," Proc. R. Soc. Lond. A. 190 (1023), p. 474–481, 1947. 

# Previous work: 

We had focused on the stability of clusters formed by 3 large and 3 small identical bubbles. We first built a bubble cluster generator that automatically produces data in the form of images and using bubbles of two different diameters, we were able to form clusters of various sizes. From there we then applied an image processing algorithm to recognize the composition and size of each cluster, and the set of bubble cluster images that included the identical bubble diameters and compositions that were pre-selected. In principle, the stability of the cluster is associated with the occurring probability of the cluster configurations. By analyzing and comparing the probability of configurations, the relative stability of each pre-selected cluster can be obtained. Finally, we developed the second algorithm to precisely classify bubble cluster configurations based on their center coordinates. Using that information we were able to generate the probability of the occurrence for each configuration. 

To understand the relation between the energetic stability and the probability of the configurations, we collaborated with a molecular dynamics simulation group and focused on clusters composed of 3 large and 3 small bubbles. We compared the probability of cluster configurations in our experimental data with simulation results. This work has been published in Soft matter [2]. 
In order to study the impact of the cluster stability to the structural transformation at a larger scale, exploring the stability of larger clusters will be necessary. Although previously we were able to successfully deploy the classification algorithm to identify the configurations for small clusters, applying the same approach to larger clusters will not be a practical strategy. This is due to the fact that a significant increase in the number of possible configurations will result from the slight increase in cluster size, which will be very challenging for us to proceed using our existing method.

[2] 	C.-C. K. N. S. C. O. a. M. D. 1. Kai Zhang, "Stable small bubble clusters in two-dimensional," Soft Matter, vol. 13, pp. 4370-4380, 2017. 
