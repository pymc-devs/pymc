/*

TODO: Figure out the container class and teach Parameter and Node
TODO: to deal with container parents. They should keep the actual 
TODO: container in their 'parents' dictionary, but their 
TODO: parent_pointers array should contain the PyMC objects in the
TODO: container. This shouldn't be a deep change, since Parameter
TODO: and Node already pass arguments to logp using their dictionaries
TODO: and only use their pointers for timestamp access.
TODO: Note that there are PyIter_Check and PyIter_Next, but maybe they
TODO: won't help.
*/

#ifndef _PYMCOBJECTS_C_
#define _PYMCOBJECTS_C_

#include "Python.h"
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

	PyObject* AbstractBase;
	PyTypeObject* PyMCBase;
	PyTypeObject* PurePyMCBase;	
	PyTypeObject* ParameterBase;
	PyTypeObject* NodeBase;
	PyTypeObject* ContainerBase;

	AbstractBase = (PyObject*) PyImport_ImportModule("AbstractBase");
	PyMCBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "PyMCBase");
	PurePyMCBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "PurePyMCBase");
	ParameterBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "ParameterBase");
	NodeBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "NodeBase");
	ContainerBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "ContainerBase");
	Py_DECREF(AbstractBase);

	/* Create the module and add the functions */
	m = Py_InitModule4("PyMCObjects", PYMC_methods,
		PyMCObjects_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);
	
	Paramtype.tp_new = PyType_GenericNew; 
	Paramtype.tp_base = ParameterBase;
	if(PyType_Ready(&Paramtype)<0) return;
	PyModule_AddObject(m, "Parameter", (PyObject *)&Paramtype); 
	
	Nodetype.tp_new = PyType_GenericNew; 
	Nodetype.tp_base = NodeBase;
	if(PyType_Ready(&Nodetype)<0) return;
	PyModule_AddObject(m, "Node", (PyObject *)&Nodetype);	
	
	Py_INCREF(&Paramtype);
	Py_INCREF(&Nodetype);

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	
	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module PyMCObjects");
}



#endif /* _PYMCOBJECTS_C_ */