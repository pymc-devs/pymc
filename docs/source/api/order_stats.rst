================
Order_Statistics
================

------------
Introduction
------------

Order statistics is a way to understand and analyze data by looking at its ordered values. Imagine you have a set of numbers, and you want to know things like the smallest number, the largest number, or the middle value. The maximum, minimum and median respectively represent them. Order statistics help you find these values and more.
In this section, we'll tackle how users can find the log-probability corresponding to the nth order statistic (maximum value) using PyMC for their own Custom distributions.

Example 1: [4, 2, 6]
	Minimum: The smallest number is 2.
	Median: Since there are three elements, the median is the middle value, which is 4.
	Maximum: The largest number is 6.

Example 2: [10, 3, 8]
	Minimum: The smallest number is 3.
	Median: Since there are three elements, the median is the middle value, which is 8.
	Maximum: The largest number is 10.

Example 3: [-1, 5, 3]
	Minimum: The smallest number is -1.
	Median: Since there are three elements, the median is the middle value, which is 3.
	Maximum: The largest number is 5.

------------------------
`Max`
------------------------
Using PyMC and Pytensor, users can extract the maximum of a distribution and derive the log-probablity corresponding to this operation.

.. autofunction:: pymc.logprob.order.max_logprob
