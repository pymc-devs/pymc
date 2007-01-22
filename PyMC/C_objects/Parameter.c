#ifndef _PARAMETER_H_
#define _PARAMETER_H_

#include "Python.h"
#include "structmember.h"
#include "PyMCBase.c"
#include "PyMCObjects.h"

// Docstring for Parameter
static char Paramtype__doc__[] = 
"Parameter in C";


// Members table for Parameter
static PyMemberDef Param_members[] = { 
{"value", T_OBJECT, offsetof(Parameter, value), 0, 
"value"}, 
{"children", T_OBJECT, offsetof(Parameter, children), 0, 
"children"},
{"__doc__", T_OBJECT, offsetof(Parameter, __doc__), 0, 
"__doc__"},
{"__name__", T_OBJECT, offsetof(Parameter, __name__), 0, 
"__name__"},
{"trace", T_OBJECT, offsetof(Parameter, trace), 0, 
"trace"},
{"isdata", T_OBJECT, offsetof(Parameter, isdata), 0, 
"isdata"},
{NULL} /* Sentinel */ 
};

static void parse_parents_of_param(Parameter *self)
{

	PyObject *parent_items; 
	PyObject *parent_now;
	PyObject *key_now;
	int i;	
	
	self->N_parents = (int) PyDict_Size(self->parents);
	parent_items =  PyDict_Items(self->parents);	

	self->node_parent_indices = malloc(sizeof(int) * self->N_parents);
	self->param_parent_indices = malloc(sizeof(int) * self->N_parents);	
	self->constant_parent_indices = malloc(sizeof(int) * self->N_parents);
	
	self->parent_pointers = malloc(sizeof(int) * self->N_parents );
	self->parent_keys = malloc(sizeof(PyObject*) * self->N_parents );
	self->parent_values = malloc(sizeof(PyObject*) * self->N_parents );
	self->parent_value_dict = PyDict_New();
	
	self->N_node_parents = 0;
	self->N_param_parents = 0;	
	self->N_constant_parents = 0;
	
	for(i=0;i<self->N_parents;i++)
	{

		key_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 0);
		parent_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 1);
		
		self->parent_pointers[i] = parent_now;
		self->parent_keys[i] = key_now;
		
		Py_INCREF(parent_now);
		Py_INCREF(key_now);

		if(PyObject_IsInstance(parent_now, (PyObject*) &Nodetype))
		{
			self->node_parent_indices[self->N_node_parents] = i;
			self->N_node_parents++;
			self->parent_values[i] = ((Node*) parent_now)->value;
			Py_INCREF(self->parent_values[i]);
		}
		else
		{	
			if(PyObject_IsInstance(parent_now, (PyObject*) &Paramtype))
			{
				self->param_parent_indices[self->N_param_parents] = i;
				self->N_param_parents++;
				self->parent_values[i] = ((Parameter*) parent_now)->value;
				Py_INCREF(self->parent_values[i]);				
			}
			else
			{
				self->constant_parent_indices[self->N_constant_parents] = i;
				self->N_constant_parents++;
				self->parent_values[i] = parent_now;
			}	
		}
		
		PyDict_SetItem(self->parent_value_dict, key_now, self->parent_values[i]);
	}
}


static void param_parent_values(Parameter *self)
{
	int index_now, i;

	for( i = 0; i < self->N_param_parents; i ++ )
	{
		index_now = self->param_parent_indices[i];
		Py_DECREF(self->parent_values[i]);
		self->parent_values[index_now] = ((Parameter*) self->parent_pointers[index_now])->value;				
		Py_INCREF(self->parent_values[i]);
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);				
	}
	
	for( i = 0; i < self->N_node_parents; i ++ )
	{
		index_now = self->node_parent_indices[i];
		Py_DECREF(self->parent_values[i]);		
		self->parent_values[index_now] = ((Node*) self->parent_pointers[index_now])->value;				
		Py_INCREF(self->parent_values[i]);		
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);				
	}	
}


static void compute_logp(Parameter *self)
{
	PyObject *new_logp, *val_tuple;	
	param_parent_values(self);
	val_tuple = PyTuple_New(1);
	PyTuple_SET_ITEM(val_tuple,0,self->value);	
	new_logp = PyObject_Call(self->logp_fun, val_tuple, self->parent_value_dict);
	Py_INCREF(new_logp);	
	Py_DECREF(self->logp);
	self->logp = new_logp;
}

//__init__
static int 
Param_init(Parameter *self, PyObject *args, PyObject *kwds) 
{
	
	static char *kwlist[] = {	"logp",  
								"name", 
								"value", 
								"parents", 
								"doc",								
								"random", 
								"trace", 
								"caching", 
								"rseed", 
								"isdata", 								
								NULL};
								
	if (! PyArg_ParseTupleAndKeywords(	args, kwds, "OOOO|OOOiOi", kwlist, 
										&self->logp_fun,
										&self->__name__, 
										&self->value, 										
										&self->parents,
										&self->__doc__,																				
										&self->random_fun, 
										&self->trace, 
										&self->caching, 
										&self->rseed, 
										&self->isdata )) 
										
	{
		return -1;
	}
										
	Py_INCREF(self->logp_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->value);
	Py_INCREF(self->parents);
	Py_XINCREF(self->__doc__);
	Py_XINCREF(self->random_fun);
	Py_XINCREF(self->trace);

	self->val_tuple = PyTuple_New(1);
	PyTuple_SET_ITEM(self->val_tuple,0,self->value);	
	
	self->cache_position = 0;
	self->logp = Py_BuildValue("");
	Py_INCREF(self->logp);	
	
	if(!PyDict_Check(self->parents)) PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
	else parse_parents_of_param(self);	
	
				
	
	return 0; 
} 


// Get and set parents
static PyObject * 
Parameter_getparents(Parameter *self, void *closure) 
{ 
	Py_XINCREF(self->parents); 
	return self->parents; 
} 
static int 
Parameter_setparents(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Parameter.parents cannot be changed after initialization."); 
	return -1; 
} 

// Get and set value
static PyObject * 
Parameter_getvalue(Parameter *self, void *closure) 
{
	Py_INCREF(self->value);
	return self->value; 
} 
static int 
Parameter_setvalue(Parameter *self, PyObject *value, void *closure) 
{
	Py_INCREF(value);
	Py_DECREF(self->value);
	self->value = value;
	self->timestamp++;
}

// Get and set logp
static PyObject * 
Parameter_getlogp(Parameter *self, void *closure) 
{
	compute_logp(self);
	Py_XINCREF(self->logp); 
	return self->logp; 
} 
static int 
Parameter_setlogp(Parameter *self, PyObject *value, void *closure) 
{
	PyErr_SetString(PyExc_TypeError, "Parameter.logp cannot be set."); 
	return -1;
}

// Get and set last_value
static PyObject * 
Parameter_getlastvalue(Parameter *self, void *closure) 
{
	if(self->cache_position > 0){
		Py_INCREF(self->value_caches[self->cache_position-1]); 
		return self->value_caches[self->cache_position-1];
	}
	else{
		PyErr_SetString(PyExc_TypeError, "Parameter.cache is empty, cannot return last_value"); 
		return NULL;
	}
} 
static int 
Parameter_setlastvalue(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Parameter.last_value cannot be set."); 
	return -1;
}

// Get and set last_logp
static PyObject * 
Parameter_getlastlogp(Parameter *self, void *closure) 
{
	if(self->cache_position > 0){
		Py_INCREF(self->logp_caches[self->cache_position-1]); 
		return self->logp_caches[self->cache_position-1];
	}
	else{
		PyErr_SetString(PyExc_TypeError, "Parameter.cache is empty, cannot return last_logp"); 
		return NULL;		
	}
} 
static int 
Parameter_setlastlogp(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Parameter.last_logp cannot be set."); 
	return -1;
}

// Get/set function table
static PyGetSetDef Param_getseters[] = { 

	{"logp", 
	(getter)Parameter_getlogp, (setter)Parameter_setlogp, 
	"logp of Parameter", 
	NULL}, 
	
	{"value", 
	(getter)Parameter_getvalue, (setter)Parameter_setvalue, 
	"value of Parameter", 
	NULL},	

	{"last_logp", 
	(getter)Parameter_getlastlogp, (setter)Parameter_setlastlogp, 
	"last logp of Parameter", 
	NULL},	
	
	{"last_value", 
	(getter)Parameter_getlastvalue, (setter)Parameter_setlastvalue, 
	"last value of Parameter", 
	NULL},	
	
	{"parents", 
	(getter)Parameter_getparents, (setter)Parameter_setparents, 
	"parents of Parameter", 
	NULL},	

	{NULL} /* Sentinel */ 
};


// Method Parameter.revert()
static char Param_revert__doc__[] = 
"Call this when rejecting a jump.";
static PyObject*
Param_revert(Parameter *self)
{
	return Py_None;
}

// Method Parameter.random()
static char Param_random__doc__[] = 
"Sample self conditional on parents.";
static PyObject*
Param_random(Parameter *self)
{
	PyObject *new_value;
	if (!self->random_fun) PyErr_SetString(PyExc_TypeError, "No random() function specified."); 
	
	param_parent_values(self);
	new_value = PyObject_Call(self->random_fun, PyTuple_New(0), self->parent_value_dict);
	Py_INCREF(new_value);
	Py_DECREF(self->value);
	self->value = new_value;
	self->timestamp++;
}


// Methods table for Parameter
static PyMethodDef Param_methods[] = {
	{"revert",	(PyCFunction)Param_revert,	METH_VARARGS,	Param_revert__doc__},
 {"random",	(PyCFunction)Param_random,	METH_VARARGS,	Param_random__doc__},
	{NULL,		NULL}		/* sentinel */
};

// Lord alone knows what this is.
static PyTypeObject Paramtype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,						/*ob_size*/
	"Parameter",			/*tp_name*/
	sizeof(Parameter),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)0,	/*tp_dealloc*/
	(printfunc)0,		/*tp_print*/
	0,//(getattrfunc)Param_getattr,	/*tp_getattr*/
	0,//(setattrfunc)Param_setattr,	/*tp_setattr*/
	(cmpfunc)0,		/*tp_compare*/
	(reprfunc)0,		/*tp_repr*/
	0,			/*tp_as_number*/
	0,		/*tp_as_sequence*/
	0,		/*tp_as_mapping*/
	(hashfunc)0,		/*tp_hash*/
	(ternaryfunc)0,		/*tp_call*/
	(reprfunc)0,		/*tp_str*/
	0, /*tp_getattro*/ 
	0, /*tp_setattro*/ 
	0, /*tp_as_buffer*/ 
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/ 
	Paramtype__doc__, /* tp_doc */ 
	0, /* tp_traverse */ 
	0, /* tp_clear */ 
	0, /* tp_richcompare */ 
	0, /*tp_weaklistoffset */
	0, /* tp_iter */ 
	0, /* tp_iternext */ 
	Param_methods, /* tp_methods */ 
	Param_members, /* tp_members */ 
	Param_getseters, /* tp_getset */ 
	(PyTypeObject*) &PyMCBasetype, /* tp_base */ 
	0, /* tp_dict */ 
	0, /* tp_descr_get */ 
	0, /* tp_descr_set */ 
	0, /* tp_dictoffset */ 
	(initproc)Param_init, /* tp_init */ 
	0, /* tp_alloc */ 
	PyType_GenericNew, /* tp_new */
};

#endif /* _PARAMETER_H_ */