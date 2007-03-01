"""
Indexable and callable. Values on base mesh ('static') must always exist,
indexing returns one of those. Also keeps two nodes and an array:


dynamic_value_registry = array maintaining values computed so far.


@node
def dynamic_covariance(dynamic_value_registry, actual_covariance):

	Computes the trapezoidal covariance upper-matrix evaluated at
	the dynamic value registry, computed conditionally in succession.
	
	That is, the first value in the dynamic registry's covariance is
	computed conditional only on the base mesh; the second value is
	computed conditional on the base mesh and the first value; and so
	on.

	
@node
def dynamic_mean(	dynamic_value_registry, actual_mean, 
					actual_covariance, dynamic_covariance):
	
	Computes the mean evaluated at the dynamic value registry, in the
	same way as dynamic covariance. However, dynamic_mean will need to
	use the _values_ in the registry in addition to the _locations_.
	
dynamic_mean and dynamic_covariance will need to be valued as special 
clases. It may make sense to combine them into the same class. They
don't need to be callable or anything fancy since they're not visible to
the user; access should be done by sensible method calls.

Oh yeah- When a new object is instantiated, it's with a clear value registry.
That includes values produced by random. Also, logp only needs to worry about 
the static values; the dynamic values can be considered as sampled from their 
prior.


"""