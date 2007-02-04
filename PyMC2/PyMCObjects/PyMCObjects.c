#ifndef _PYMCOBJECTS_C_
#define _PYMCOBJECTS_C_

#include "Python.h"
#include "PyMCBase.c"
#include "Parameter.c"
#include "Node.c"

/* List of methods defined in the module */
static struct PyMethodDef PYMC_methods[] = {
	{NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initPyMCObjects) */
static char PyMCObjects_module_documentation[] = 
"PyMC Objects: PyMCBase, Parameter, Node, RemoteProxy";

void
initPyMCObjects()
{
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule4("PyMCObjects", PYMC_methods,
		PyMCObjects_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);
		
	/* Add Parameter and Node */
	PyMCBasetype.tp_new = PyType_GenericNew; 
	if(PyType_Ready(&PyMCBasetype)<0) return;
	PyModule_AddObject(m, "PyMCBase", (PyObject *)&PyMCBasetype); 	

	RemoteProxyBasetype.tp_new = PyType_GenericNew; 	
	if(PyType_Ready(&RemoteProxyBasetype)<0) return;
	PyModule_AddObject(m, "RemoteProxyBase", (PyObject *)&RemoteProxyBasetype);	
	
	Paramtype.tp_new = PyType_GenericNew; 
	if(PyType_Ready(&Paramtype)<0) return;
	PyModule_AddObject(m, "Parameter", (PyObject *)&Paramtype); 
	
	Nodetype.tp_new = PyType_GenericNew; 
	if(PyType_Ready(&Nodetype)<0) return;
	PyModule_AddObject(m, "Node", (PyObject *)&Nodetype);	
	
	Py_INCREF(&PyMCBasetype);
	Py_INCREF(&RemoteProxyBasetype);
	Py_INCREF(&Paramtype);
	Py_INCREF(&Nodetype);

/*	Not going to try this until the exit bug is resolved.	
	if(PyType_Ready(&LogLikeDescriptor)<0) return;
	PyModule_AddObject(m, "LogLikeDescriptor", (PyObject *)&LogLikeDescriptor);	
*/
	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyString_FromString("PyMCObjects.error");
	PyDict_SetItemString(d, "error", ErrorObject);
	
	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module PyMCObjects");
}



#endif /* _PYMCOBJECTS_C_ */