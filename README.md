# LED-Point-matching in 2D

## Introduction
This Point matching algorithm is call [Robust point matching](https://en.wikipedia.org/wiki/Point_set_registration#Robust_point_matching) whose detail is in this paper [link](https://www.cise.ufl.edu/~anand/pdf/prrevfinal.pdf)

This algorithm is belong to non-rigid registration but only suitable for regid-body point set.The distance of any two points within the set is unchanged after mapping in 3D space. For example in this case, it matches two LEDs set on network switch.

## Algorithm detail
In this algorithm, it will optimize the parameters of the affine transformation and the matching matrix.Matching matrix define the correspondence between two set. Affine transformation maps one point set to the other. In my case affine transformation include translation、scaling、rotation、projection(1-point).

## Optimize process
It optimize affine transformation and matching matrix respectively.Step1, fix affine transformation and caculate the matching matrix by the distance between points at different set.Step2, fix matching matrix and find the best affine transformation which minimize the cost function. Keep repeat step1 and step2 until the cost converge.

We can see the mayching process as below

![Matching Demo](https://github.com/LI-ZONG-HAN/LED-Point-matching/blob/master/Matching_Animation.gif)

For the application detail in my real case. See here [[mandarin]](https://zongsoftwarenote.blogspot.com/2018/01/point-matching-algorithm-for-rigid-body.html#more) [[Slides in English]](https://docs.google.com/presentation/d/1WqPyfLMDJUNg_eXAg0ISzHjTDkNoBHfAsP43zSk7MYU/edit#slide=id.g2f647f410e_0_66)

## How to run my code
'''
$ python eval.py
'''
