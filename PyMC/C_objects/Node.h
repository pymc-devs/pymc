#ifndef PYMCOBJECTS_NODE
#define PYMCOBJECTS_NODE

#include "Python.h"

/* ---------------------------------------------------------------- */

/* Declarations for objects of type Node */

typedef struct {
	PyObject_HEAD
	/* XXXX Add your own stuff here */
} Nodeobject;

static PyTypeObject Nodetype;

/* End of code for Parameter objects */
/* -------------------------------------------------------- */


static struct PyMethodDef Node_methods[] = {
	
	{NULL,		NULL}		/* sentinel */
};

/* ---------- */


static PyObject *
Node_getattr(Nodeobject *self, char *name)
{
	/* XXXX Add your own getattr code here */
	return Py_FindMethod(Node_methods, (PyObject *)self, name);
}

static int
Node_setattr(Nodeobject *self, char *name, PyObject *v)
{
	/* Set attribute 'name' to value 'v'. v==NULL means delete */
	
	/* XXXX Add your own setattr code here */
	return -1;
}

static char Nodetype__doc__[] = 
"Node in C"
;

static PyTypeObject Nodetype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"Node",			/*tp_name*/
	sizeof(Nodeobject),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)0,	/*tp_dealloc*/
	(printfunc)0,		/*tp_print*/
	(getattrfunc)Node_getattr,	/*tp_getattr*/
	(setattrfunc)Node_setattr,	/*tp_setattr*/
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
	Nodetype__doc__ /* Documentation string */
};
#endif