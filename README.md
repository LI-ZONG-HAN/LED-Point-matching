# LED-Point-matching

##Introduction
This Point matching algorithm is call [Robust point matching](https://en.wikipedia.org/wiki/Point_set_registration#Robust_point_matching) whose detail is in this paper [link](https://www.cise.ufl.edu/~anand/pdf/prrevfinal.pdf)

This algorithm is belong to non-rigid registration but only suitable for regid-body point set.The distance of any two points within the set is unchanged between two set in 3D space. For example in this case, it matches two LEDs set on network switch.

## Algorithm detail
In this algorithm, it will optimize the parameters of the affine transformation and the matching matrix.Matching matrix define the correspondence between two set. Affine transformation maps one point set to the other. In my case affine transformation include translation、scaling、rotation、projection(1-point）. 

$ python eval.py

![Matching Demo](https://github.com/LI-ZONG-HAN/LED-Point-matching/blob/master/Matching_Animation.gif)
