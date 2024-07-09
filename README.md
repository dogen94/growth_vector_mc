# growth_vector_mc
Growth Vector Monte Carlo

The motivation was to create a method for generating samples from a joint
distribution with a known uncertainty in each dimension. The uncertainty
can be given by a distribution or a vector that will scale the known data's
covariance in a centered multivariate gaussian. The generated data starts
at a known data point. It takes a step in a given direction with the step
size drawn from the aforementioned uncertainty distribution. The direction
of the step depends on the "growth vectors". These can be set to anything
in principle. However, the SVD basis vectors of the known data (and their
negated directions) were chosen. 

For a given step, the probability that
one of these growth vectors is chosen is calculated from the probability
that data exists in that direction. This is estimated by first calculating
the k-nearest neighbors of the nearest known data point to the current
position. Then the angle between these k-nearest data points to each
growth vector is estimated by inverse cosine of their scaled dot
products. If this angle is less than 45 degrees, it is counted as being
in the growth vector direction. The probability of data in a given direction
is just the total data in a given direction divided by the total in the
k-nearest neighbor hypersphere (i.e. k). The final probability drawn for
each step takes this directional probability calculated here and adjusts
for local density of the data (can be preprocessed with the k-nearest
neighbor). The idea being that if the local density is low, the confidence
in the directional probability is less. In this case the
probability of each direction is a weighted sum of a uniform probability 
for each direction and the k-nearest neighbor estimated directional
probability described before. The "low" local density is low relative to
the max local density of any point in the known data.

There is also an overstepping protection step where if the next step
would be far outside the known data, it is instead limited to the closest
point. The actual metric for this protection is if at the next step, the
distance to the closest point would be greater than that point's k-th 
nearest neighbor, the next step is instead set to the closest point.
This done with the goal to preserve sharp edges in the joint
distribution while allowing for less dense data areas to be expanded.
