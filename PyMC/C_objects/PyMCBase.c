#ifndef _PYMCBASE_H_
#define _PYMCBASE_H_

#include "Python.h"
#include "structmember.h"
#include "PyMCObjects.h"

static PyObject *ErrorObject;


// Docstring for PyMCBase
static char PyMCBasetype__doc__[] = 
"Abstract base class.";


// Empty methods table
static struct PyMethodDef PyMCBase_methods[] = {
	{NULL,		NULL}		/* sentinel */
};

// Appease the demons
static PyTypeObject PyMCBasetype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"PyMCBase",			/*tp_name*/
	sizeof(PyMCBase),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	/* methods */
	(destructor)0,	/*tp_dealloc*/
	(printfunc)0,		/*tp_print*/
	(getattrfunc)0,	/*tp_getattr*/
	(setattrfunc)0,	/*tp_setattr*/
	(cmpfunc)0,		/*tp_compare*/
	(reprfunc)0,		/*tp_repr*/
	0,			/*tp_as_number*/
	0,		/*tp_as_sequence*/
	0,		/*tp_as_mapping*/
	(hashfunc)0,		/*tp_hash*/
	(ternaryfunc)0,		/*tp_call*/
	(reprfunc)0,		/*tp_str*/

	/* Space for future expansion */
	0L,0L,0L,0L,
	PyMCBasetype__doc__ /* Documentation string */
};


/* Some utility functions */


#endif /* _PYMCBASE_H_ */