#ifndef PYMCOBJECTS_PARAMETER
#define PYMCOBJECTS_PARAMETER

#include "Python.h"
#include "PyMCBase.h"

static PyObject *ErrorObject;

/* ----------------------------------------------------- */

/* Declarations for objects of type Parameter */

typedef struct {
	PyObject_HEAD
	/* XXXX Add your own stuff here */
} Paramobject;

static PyTypeObject Paramtype;

/* ---------------------------------------------------------------- */

static char Param_revert__doc__[] = 
"Call this when rejecting a jump."
;

static PyObject *
Param_revert(Paramobject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


static char Param_random__doc__[] = 
"Sample self conditional on parents."
;

static PyObject *
Param_random(Paramobject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


static struct PyMethodDef Param_methods[] = {
	{"revert",	(PyCFunction)Param_revert,	METH_VARARGS,	Param_revert__doc__},
 {"random",	(PyCFunction)Param_random,	METH_VARARGS,	Param_random__doc__},
 
	{NULL,		NULL}		/* sentinel */
};

/* ---------- */


static PyObject *
Param_getattr(Paramobject *self, char *name)
{
	/* XXXX Add your own getattr code here */
	return Py_FindMethod(Param_methods, (PyObject *)self, name);
}

static int
Param_setattr(Paramobject *self, char *name, PyObject *v)
{
	/* Set attribute 'name' to value 'v'. v==NULL means delete */
	
	/* XXXX Add your own setattr code here */
	return -1;
}

static char Paramtype__doc__[] = 
"Parameter in C"
;

static PyTypeObject Paramtype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"Parameter",			/*tp_name*/
	sizeof(Paramobject),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)0,	/*tp_dealloc*/
	(printfunc)0,		/*tp_print*/
	(getattrfunc)Param_getattr,	/*tp_getattr*/
	(setattrfunc)Param_setattr,	/*tp_setattr*/
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
	Paramtype__doc__ /* Documentation string */
};

#endif