/*

TODO: Figure out the container class and teach Parameter and Node
TODO: to deal with container parents. They should keep the actual 
TODO: container in their 'parents' dictionary, but their 
TODO: parent_pointers array should contain the PyMC objects in the
TODO: container. This shouldn't be a deep change, since Parameter
TODO: and Node already pass arguments to logp using their dictionaries
TODO: and only use their pointers for timestamp access.

TODO: Instead of passing a new 'children' set in to the PyMC objects,
TODO: make a function in Abstract Base: def newset(): return set()
TODO: and import and call the function from Parameter and Node's
TODO: constructors. Then get rid of the 'children' argument
TODO: in the decorators.

Two types of memory errors: the 'gc_collect' one

Host Name:      Anand-Patils-Computer
Date/Time:      2007-02-22 17:39:59.862 -0800
OS Version:     10.4.8 (Build 8L127)
Report Version: 4

Command: Python
Path:    /Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python
Parent:  bash [1895]

Version: 2.5a0 (2.5alpha0)

PID:    1952
Thread: 0

Exception:  EXC_BAD_ACCESS (0x0001)
Codes:      KERN_PROTECTION_FAILURE (0x0002) at 0x00000054

Thread 0 Crashed:
0   org.python.python    	0x002eae50 visit_decref + 32 (gcmodule.c:270)
1   org.python.python    	0x0026a93c tupletraverse + 76 (tupleobject.c:443)
2   org.python.python    	0x002ebd00 collect + 656 (gcmodule.c:293)
3   org.python.python    	0x002ec630 _PyObject_GC_Malloc + 64 (gcmodule.c:897)
4   org.python.python    	0x0026bc60 PyType_GenericAlloc + 80 (typeobject.c:454)
5   org.python.python    	0x0022686c BaseException_new + 44 (exceptions.c:37)

gdb output:

Program received signal EXC_BAD_ACCESS, Could not access memory.
Reason: KERN_INVALID_ADDRESS at address: 0x0a68fe40
0x002eae50 in visit_decref (op=0x28a1cb0, data=0x0) at /Users/ronald/Python/r25/Modules/gcmodule.c:270
270     /Users/ronald/Python/r25/Modules/gcmodule.c: No such file or directory.
       in /Users/ronald/Python/r25/Modules/gcmodule.c

or

Program received signal EXC_BAD_ACCESS, Could not access memory.
Reason: KERN_PROTECTION_FAILURE at address: 0x00000056
0x002eae50 in visit_decref (op=0x28a17e0, data=0x0) at /Users/ronald/Python/r25/Modules/gcmodule.c:270
270     in /Users/ronald/Python/r25/Modules/gcmodule.c


And the PyDict_SetItem one:

Host Name:      Anand-Patils-Computer
Date/Time:      2007-02-22 17:41:05.155 -0800
OS Version:     10.4.8 (Build 8L127)
Report Version: 4

Command: Python
Path:    /Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python
Parent:  bash [1895]

Version: 2.5a0 (2.5alpha0)

PID:    1956
Thread: 0

Exception:  EXC_BAD_ACCESS (0x0001)
Codes:      KERN_PROTECTION_FAILURE (0x0002) at 0x00000000

Thread 0 Crashed:
0   org.python.python    	0x0024e4a0 PyDict_SetItem + 160 (dictobject.c:617)
1   PyMCObjects.so       	0x01284e84 Parameter_getlogp + 180 (Parameter.c:404)
2   org.python.python    	0x002549f0 PyObject_GenericGetAttr + 432 (object.c:1312)
3   org.python.python    	0x002b221c PyEval_EvalFrameEx + 13580 (ceval.c:1992)
4   org.python.python    	0x002b54c0 PyEval_EvalCodeEx + 2096 (ceval.c:2833)
5   org.python.python    	0x00238c98 function_call + 360 (funcobject.c:524)
6   org.python.python    	0x0020fac8 call_function_tail + 200 (abstract.c:1861)
7   org.python.python    	0x002144c4 PyObject_CallFunction + 180 (abstract.c:1915)
8   org.python.python    	0x002549f0 PyObject_GenericGetAttr + 432 (object.c:1312)
9   org.python.python    	0x002b221c PyEval_EvalFrameEx + 13580 (ceval.c:1992)

no gdb output captured.
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