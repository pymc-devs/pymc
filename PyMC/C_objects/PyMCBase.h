#ifndef PYMCOBJECTS_BASE
#define PYMCOBJECTS_BASE
#include "Python.h"

static PyObject *ErrorObject;

/* ----------------------------------------------------- */

/* Declarations for objects of type PyMCBase */

typedef struct {
	PyObject_HEAD
	/* XXXX Add your own stuff here */
} PyMCBaseobject;

static PyTypeObject PyMCBasetype;



/* ---------------------------------------------------------------- */

static struct PyMethodDef PyMCBase_methods[] = {
	
	{NULL,		NULL}		/* sentinel */
};

/* ---------- */


static char PyMCBasetype__doc__[] = 
"Abstract base class."
;

static PyTypeObject PyMCBasetype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"PyMCBase",			/*tp_name*/
	sizeof(PyMCBaseobject),		/*tp_basicsize*/
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

/* End of code for PyMCBase objects */
/* -------------------------------------------------------- */


#endif

