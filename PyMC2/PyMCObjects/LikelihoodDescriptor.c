// Not going to try this until the exit bug is resolved...

#ifndef _LIKELIHOODDESCRIPTOR_C_
#define _LIKELIHOODDESCRIPTOR_C_

#include "PyMCObjects.h"

// Get and set log-likelihood
static PyObject * 
getloglike(PyObject *self, void *closure) 
{
	PyObject *loglike;
	PyObject *children, child, loglike_now;

	loglike = Py_BuildValue("f",0.0);
	children = PyObject_GetAttrString(self, "children");
	children = PyObject_GetIter(children);	

	if(PyErr_Occurred()) return NULL;
	Py_DECREF(children);
	
	while (child = PyIter_Next(iterator)) { 		
		loglike_now = PyObject_GetAttrString(child, "logp");
		Py_DECREF(child); 
		if(PyErr_Occurred()) return NULL;		

		loglike = PyNumber_InPlaceAdd(loglike, loglike_now);
		// This decref is questionable.
		Py_DECREF(loglike);
		Py_DECREF(loglike_now);
	} 
	Py_DECREF(children);
	return loglike;
}
static int 
setloglike(PyObject *self, PyObject *value, void *closure) 
{
	PyErr_SetString(PyExc_AttributeError,"Log-likelihood cannot be set.");
	return -1;
}

typedef struct PyGetSetDef { 
"loglike"; /* attribute name */ 
get_loglike; /* C function to get the attribute */ 
set_loglike; /* C function to set the attribute */ 
"Log-likelihood of a SamplingMethod"; /* optional doc string */ 
} LikelihoodDescriptor;

#endif /* _LIKELIHOODDESCRIPTOR_C_ */



