#ifndef _NODE_H_
#define _NODE_H_


#include "Python.h"
#include "structmember.h"
#include "PyMCBase.c"
#include "PyMCObjects.h"


//__init__
static int 
Node_init(Node *self, PyObject *args, PyObject *kwds) 
{	
	int i;
	static char *kwlist[] = {	"eval",
								"name",
								"parents",
								"children",
								"doc",
								"trace",
								NULL};
								
	if (!PyArg_ParseTupleAndKeywords(	args, kwds, "OOOO|OO", kwlist, 
										&self->eval_fun,
										&self->__name__,
										&self->parents,
										&self->children,
										&self->__doc__,
										&self->trace)) 
	{
		return -1;
	}
	
	if(!self->__doc__) self->__doc__ = self->__name__;
										
	Py_INCREF(self->eval_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->parents);
	Py_XINCREF(self->__doc__);
	Py_XINCREF(self->trace);
	
	self->value = Py_BuildValue("");
	for(i=0;i<2;i++)
	{
		self->value_caches[i] = Py_BuildValue("");
		Py_INCREF(self->value_caches[i]);
	}
	Py_INCREF(self->value);
	
	if(!PyDict_Check(self->parents)) PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
	else parse_parents_of_node(self);
	
	self->timestamp = 0;
	compute_value(self);
	node_cache(self);	
	
	Py_INCREF(self->children);	
	PyObject_CallMethodObjArgs(self->children, Py_BuildValue("s","clear"), NULL, NULL);	

	return 0;
}

static void Node_dealloc(Node* self)
{
	
	int i, index_now;
	
	Py_DECREF(self->value);
	Py_DECREF(self->eval_fun);
	
	Py_DECREF(self->parents);
	Py_DECREF(self->children);
	Py_DECREF(self->__doc__);
	Py_DECREF(self->__name__);
	Py_DECREF(self->trace);
	Py_DECREF(self->parent_value_dict);
	
	for( i = 0; i < self->N_parents; i++ )
	{
		Py_DECREF(self->parent_pointers[i]);
		Py_XDECREF(self->parent_values[i]);
		Py_DECREF(self->parent_keys[i]);
	}
	
	for(i=0;i<2;i++) Py_DECREF(self->value_caches[i]);
	
	free(self->parent_keys);
	free(self->parent_pointers);
	free(self->parent_values);
	free(self->pymc_parent_indices);
	free(self->constant_parent_indices);
	
}


static void parse_parents_of_node(Node *self)
{

	PyObject *parent_items; 
	PyObject *parent_now;
	PyObject *key_now;
	int i, j;	
	
	self->N_parents = (int) PyDict_Size(self->parents);
	parent_items =  PyDict_Items(self->parents);	

	self->pymc_parent_indices = malloc(sizeof(int) * self->N_parents);
	self->constant_parent_indices = malloc(sizeof(int) * self->N_parents);
	
	self->parent_pointers = malloc(sizeof(int) * self->N_parents );
	self->parent_keys = malloc(sizeof(PyObject*) * self->N_parents );
	self->parent_values = malloc(sizeof(PyObject*) * self->N_parents );
	self->parent_value_dict = PyDict_New();
	
	self->N_pymc_parents = 0;
	self->N_constant_parents = 0;
	
	for(i=0;i<self->N_parents;i++)
	{

		key_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 0);
		parent_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 1);
		
		self->parent_pointers[i] = parent_now;
		self->parent_keys[i] = key_now;
		
		Py_INCREF(parent_now);
		Py_INCREF(key_now);

		if(PyObject_IsInstance(parent_now, (PyObject*) &PyMCBasetype))
		{
			self->pymc_parent_indices[self->N_pymc_parents] = i;
			self->N_pymc_parents++;
			self->parent_values[i] = PyObject_GetAttrString(self->parent_pointers[i], "value");
			PyObject_CallMethodObjArgs(PyObject_GetAttrString(self->parent_pointers[i], "children"), Py_BuildValue("s","add"), self, NULL);							
		}
		else
		{	
			self->constant_parent_indices[self->N_constant_parents] = i;
			self->N_constant_parents++;
			self->parent_values[i] = parent_now;
			//Py_INCREF(self->parent_values[i]);				
		}
		
		PyDict_SetItem(self->parent_value_dict, key_now, self->parent_values[i]);
	}
	for(i=0;i<2;i++) 
	{
		self->parent_timestamp_caches[i] = malloc(sizeof(int) * self->N_parents );
		for(j=0;j<self->N_parents;j++) self->parent_timestamp_caches[i][j] = -1;
	}
}

static void node_parent_values(Node *self)
{
	int index_now, i;

	for( i = 0; i < self->N_pymc_parents; i++ )
	{
		index_now = self->pymc_parent_indices[i];
		Py_DECREF(self->parent_values[index_now]);			
		self->parent_values[index_now] = PyObject_GetAttrString(self->parent_pointers[index_now], "value");
		//Py_INCREF(self->parent_values[index_now]);		
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);
	}
}

static void compute_value(Node *self)
{
	node_parent_values(self);
	Py_DECREF(self->value);
	self->value = PyObject_Call(self->eval_fun, PyTuple_New(0), self->parent_value_dict);
}

static int node_check_for_recompute(Node *self)
{
	int i,j,recompute,index_now;
	//PyObject *timestamp;
	recompute = 0;
	for(i=0;i<2;i++)
	{
		for(j=0;j<self->N_pymc_parents;j++)
		{
			index_now = self->pymc_parent_indices[j];
			//timestamp = PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp");
			//if(self->parent_timestamp_caches[i][index_now] != PyInt_AS_LONG(timestamp));
			if(self->parent_timestamp_caches[i][index_now] != downlow_gettimestamp((Parameter *)(self->parent_pointers[index_now])))
			{
				recompute = 1;
				break;
			}
			//Py_DECREF(timestamp);
		}

		if(recompute==0) return i;
	}
	return -1;
}

static void node_cache(Node *self)
{
	int j, index_now;
	//PyObject *timestamp;
	Py_INCREF(self->value);
	Py_DECREF(self->value_caches[1]);
	self->value_caches[1] = self->value_caches[0];
	self->value_caches[0] = self->value;
	
	self->timestamp++;
	
	for(j=0;j<self->N_pymc_parents;j++)
	{
		index_now = self->pymc_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		//timestamp = PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp");
		//self->parent_timestamp_caches[0][index_now] = (int) PyInt_AS_LONG(timestamp);
		self->parent_timestamp_caches[0][index_now] = downlow_gettimestamp((Parameter *)(self->parent_pointers[index_now]));
		//Py_DECREF(timestamp);
	}
}

// Get and set value
static PyObject * 
Node_getvalue(Node *self, void *closure) 
{
	int i;
	i=node_check_for_recompute(self);
	//i=-1;
	
	if(i<0) 
	{
		compute_value(self);
		node_cache(self);		
	}
	
	else
	{
		Py_INCREF(self->value_caches[i]);
		Py_DECREF(self->value);
		self->value = self->value_caches[i];
	}
	
	Py_INCREF(self->value); 
	//return Py_BuildValue("(O,[O,O])",self->value, self->value_caches[0], self->value_caches[1]); 
	return self->value;
} 
static int 
Node_setvalue(Node *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Node.value cannot be set."); 
	return -1;
}


// Get and set timestamp
static PyObject * 
Node_gettimestamp(Node *self, void *closure) 
{
	PyObject *timestamp;
	timestamp = Py_BuildValue("i", self->timestamp);
	return timestamp;
}

static int 
Node_settimestamp(Node *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Node.timestamp cannot be set."); 
	return -1;
}

// Get/set function table
static PyGetSetDef Node_getseters[] = { 

	{"value", 
	(getter)Node_getvalue, (setter)Node_setvalue, 
	"value of Node", 
	NULL},
	
	{"timestamp", 
	(getter)Node_gettimestamp, (setter)Node_settimestamp,
	"timestamp of Node",
	NULL}, 

	{NULL} /* Sentinel */ 
}; 


// Type declaration
static PyTypeObject Nodetype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,				/*ob_size*/
	"Node",			/*tp_name*/
	sizeof(Node),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)Node_dealloc,	/*tp_dealloc*/
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