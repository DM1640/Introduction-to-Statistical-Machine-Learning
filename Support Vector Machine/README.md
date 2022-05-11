If there are two classes data, an SVM can separate them to +1 and -1 by searching the best hyperplane. Support vectors are the special points that are equally closest to the best hyperplane. They not only define the maximum margin of the best hyperplane to the data points but also determine the direction and position of the hyperplane, which means moving one of the support vectors will change the resulting hyperplane. Besides, if a point from the non-support vectors moved to the boundary of the slab where the support vectors are on, or the point was within the maximum margin, the resulting hyperplane would also change.
<br />
<br />
In hard margin SVMs, support vectors are on the boundary of the slab. However, in soft margin SVMs, support vectors are on or within the boundary of the slab.
