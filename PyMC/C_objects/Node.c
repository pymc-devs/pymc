#ifndef _NODE_H_
#define _NODE_H_


#include "Python.h"
#include "structmember.h"
#include "PyMCBase.c"
#include "PyMCObjects.h"


// Docstring for Node
static char Nodetype__doc__[] = 
"Node in C";


// Empty methods table for Node
static PyMethodDef Node_methods[] = { 
	{NULL,		NULL}		/* sentinel */
};


// Node's member table
static PyMemberDef Node_members[] = { 
{"children", T_OBJECT, offsetof(Node, children), 0, 
"children"},
{"__doc__", T_OBJECT, offsetof(Node, __doc__), 0, 
"__doc__"},
{"__name__", T_OBJECT, offsetof(Node, __name__), 0, 
"__name__"},
{"trace", T_OBJECT, offsetof(Node, trace), 0, 
"trace"},
{NULL} /* Sentinel */ 
};

static void parse_parents_of_node(Node *self)
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
		}
		else
		{	
			if(PyObject_IsInstance(parent_now, (PyObject*) &Paramtype))
			{
				self->param_parent_indices[self->N_param_parents] = i;
				self->N_param_parents++;
				self->parent_values[i] = ((Parameter*) parent_now)->value;
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


//__init__
static int 
Node_init(Node *self, PyObject *args, PyObject *kwds) 
{	
	
	static char *kwlist[] = {	"eval",
								"name",
								"parents",
								"doc",
								"caching",
								"trace",
								NULL};
								
	if (!PyArg_ParseTupleAndKeywords(	args, kwds, "OOO|OiO", kwlist, 
										&self->eval_fun,
										&self->__name__,
										&self->parents,
										&self->__doc__,
										&self->caching,
										&self->trace)) 
	{
		return -1;
	}
										
	Py_INCREF(self->eval_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->parents);
	Py_XINCREF(self->__doc__);
	Py_XINCREF(self->trace);
	
	self->value = Py_BuildValue("");
	Py_INCREF(self->value);
	
	if(!PyDict_Check(self->parents)) PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
	else parse_parents_of_node(self);

	return 0;
}

static void node_parent_values(Node *self)
{
	int index_now, i;

	for( i = 0; i < self->N_param_parents; i++ )
	{
		index_now = self->param_parent_indices[i];
		Py_DECREF(self->parent_values[i]);
		self->parent_values[index_now] = ((Parameter*) self->parent_pointers[index_now])->value;				
		Py_INCREF(self->parent_values[i]);
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);		
	}
	
	for( i = 0; i < self->N_node_parents; i++ )
	{
		index_now = self->node_parent_indices[i];
		Py_DECREF(self->parent_values[i]);		
		self->parent_values[index_now] = ((Node*) self->parent_pointers[index_now])->value;				
		Py_INCREF(self->parent_values[i]);		
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);				
	}
}

static void compute_value(Node *self)
{
	PyObject *new_value;
	node_parent_values(self);
	new_value = PyObject_Call(self->eval_fun, PyTuple_New(0), self->parent_value_dict);
	Py_INCREF(new_value);	
	Py_DECREF(self->value);
	self->value = new_value;
}

// Get and set parents
static PyObject * 
Node_getparents(Node *self, void *closure) 
{ 
	Py_INCREF(self->parents); 
	return self->parents; 
} 
static int 
Node_setparents(Node *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Node.parents cannot be changed after initialization."); 
	return -1; 
} 

// Get and set value
static PyObject * 
Node_getvalue(Node *self, void *closure) 
{
	compute_value(self);
	Py_XINCREF(self->value); 
	return self->value; 
} 
static int 
Node_setvalue(Node *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Node.value cannot be set."); 
	return -1;
}

// Get/set function table
static PyGetSetDef Node_getseters[] = { 

	{"value", 
	(getter)Node_getvalue, (setter)Node_setvalue, 
	"value of Node", 
	NULL}, 
	
	{"parents", 
	(getter)Node_getparents, (setter)Node_setparents, 
	"parents of Node", 
	NULL},	

	{NULL} /* Sentinel */ 
}; 


// Ancient Egyptian heiroglyphics
static PyTypeObject Nodetype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"Node",			/*tp_name*/
	sizeof(Node),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)0,	/*tp_dealloc*/
	(printfunc)0,		/*tp_print*/
	0,//(getattrfunc)Node_getattr,	/*tp_getattr*/
	0,//(setattrfunc)Node_setattr,	/*tp_setattr*/
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
	Nodetype__doc__, /* tp_doc */ 
	0, /* tp_traverse */ 
	0, /* tp_clear */ 
	0, /* tp_richcompare */ 
	0, /*tp_weaklistoffset */
	0, /* tp_iter */ 
	0, /* tp_iternext */ 
	Node_methods, /* tp_methods */ 
	Node_members, /* tp_members */ 
	Node_getseters, /* tp_getset */ 
	(PyTypeObject*) &PyMCBasetype, /* tp_base */ 
	0, /* tp_dict */ 
	0, /* tp_descr_get */ 
	0, /* tp_descr_set */ 
	0, /* tp_dictoffset */ 
	(initproc)Node_init, /* tp_init */ 
	0, /* tp_alloc */ 
	PyType_GenericNew, /* tp_new */			
};

#endif /* _NODE_H_ */