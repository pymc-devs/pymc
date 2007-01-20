#ifndef _PYMCOBJECTS_C_
#define _PYMCOBJECTS_C_

#include "Python.h"
#include "PyMCBase.h"
#include "Parameter.h"
#include "Node.h"

/* List of methods defined in the module */

static struct PyMethodDef PYMC_methods[] = {
	
	{NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initPyMCObjects) */

static char PyMCObjects_module_documentation[] = 
""
;

void
initPyMCObjects()
{
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule4("PyMCObjects", PYMC_methods,
		PyMCObjects_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);
		
	Paramtype.tp_new = PyType_GenericNew;
	Paramtype.tp_base = &PyMCBasetype;
	Paramtype.tp_methods = Param_methods;
	if(PyType_Ready(&Paramtype)<0) return;
	PyModule_AddObject(m, "Parameter", (PyObject *)&Paramtype); 

	Nodetype.tp_new = PyType_GenericNew;
	Nodetype.tp_base = &PyMCBasetype;	
	if(PyType_Ready(&Nodetype)<0) return;
	PyModule_AddObject(m, "Node", (PyObject *)&Nodetype);	 		

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyString_FromString("PyMCObjects.error");
	PyDict_SetItemString(d, "error", ErrorObject);

	/* XXXX Add constants here */
	
	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module PyMCObjects");
}

#endif /* _PYMCOBJECTS_C_ */