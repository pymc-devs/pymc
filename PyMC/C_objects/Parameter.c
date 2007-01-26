#ifndef _PARAMETER_H_
#define _PARAMETER_H_

#include "Python.h"
#include "structmember.h"
#include "PyMCBase.c"
#include "PyMCObjects.h"

//__init__
static int 
Param_init(Parameter *self, PyObject *args, PyObject *kwds) 
{
	int i;
	static char *kwlist[] = {	"logp",  
								"name", 
								"value", 
								"parents", 
								"children",
								"doc",								
								"random", 
								"trace",  
								"rseed", 
								"isdata", 								
								NULL};
								
	if (! PyArg_ParseTupleAndKeywords(	args, kwds, "OOOOO|OOOOi", kwlist, 
										&self->logp_fun,
										&self->__name__, 
										&self->value, 										
										&self->parents,
										&self->children,
										&self->__doc__,																				
										&self->random_fun, 
										&self->trace, 
										&self->rseed, 
										&self->isdata )) 
										
	{
		return -1;
	}
	
	// Initialize optional arguments.
	if(!self->__doc__) self->__doc__ = self->__name__;
	if(!self->isdata) self->isdata = 0;
	
	// Incref all PyObjects so they don't go away.
	Py_INCREF(self->logp_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->value);
	Py_INCREF(self->parents);
	Py_XINCREF(self->__doc__);
	Py_XINCREF(self->random_fun);
	Py_XINCREF(self->trace);
	Py_INCREF(self->children);	
	
	// Initialize PyObjects that aren't passed in.
	self->val_tuple = PyTuple_New(1);
	PyTuple_SET_ITEM(self->val_tuple,0,self->value);	
	
	self->logp = Py_BuildValue("");
	Py_INCREF(self->logp);
		
	for(i=0;i<2;i++)
	{
		self->logp_caches[i] = Py_BuildValue("");
		Py_INCREF(self->logp_caches[i]);
	} 
	self->last_value = Py_BuildValue("");
	Py_INCREF(self->last_value);		
	
	if(!PyDict_Check(self->parents)) PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
	else parse_parents_of_param(self);
	
	self->timestamp = 0;
	self->reverted = 0;
	compute_logp(self);
	param_cache(self);

	PyObject_CallMethodObjArgs(self->children, Py_BuildValue("s","clear"), NULL, NULL);		
	
	return 0; 
} 

static void parse_parents_of_param(Parameter *self)
{

	PyObject *parent_items; 
	PyObject *parent_now;
	PyObject *key_now;
	int i, j;
	
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
			PyObject_CallMethodObjArgs(((Node*) parent_now)->children, Py_BuildValue("s","add"), self, NULL);
		}
		else
		{	
			if(PyObject_IsInstance(parent_now, (PyObject*) &Paramtype))
			{
				self->param_parent_indices[self->N_param_parents] = i;
				self->N_param_parents++;
				self->parent_values[i] = ((Parameter*) parent_now)->value;
				Py_INCREF(self->parent_values[i]);
				PyObject_CallMethodObjArgs(((Parameter*) parent_now)->children, Py_BuildValue("s","add"), self, NULL);				
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
	for(i=0;i<2;i++) 
	{
		self->parent_timestamp_caches[i] = malloc(sizeof(int) * self->N_parents );
		for(j=0;j<self->N_parents;j++) self->parent_timestamp_caches[i][j] = -1;
	}
}


static void param_parent_values(Parameter *self)
{
	int index_now, i;
	void *closure_arg;

	for( i = 0; i < self->N_param_parents; i ++ )
	{
		index_now = self->param_parent_indices[i];
		Py_DECREF(self->parent_values[i]);
		//self->parent_values[index_now] = ((Parameter*) self->parent_pointers[index_now])->value;
		PyObject_GetAttrString(self->parent_pointers[index_now], "value");
		Py_INCREF(self->parent_values[i]);
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);				
	}
	
	for( i = 0; i < self->N_node_parents; i ++ )
	{
		index_now = self->node_parent_indices[i];
		Py_DECREF(self->parent_values[i]);		
		//self->parent_values[index_now] = Node_getvalue(((Node*) self->parent_pointers[index_now]), closure_arg);	
		PyObject_GetAttrString(self->parent_pointers[index_now], "value");
		Py_INCREF(self->parent_values[i]);		
		PyDict_SetItem(self->parent_value_dict, self->parent_keys[index_now], self->parent_values[index_now]);				
	}	
}


static void compute_logp(Parameter *self)
{
	PyObject *new_logp, *val_tuple;	
	param_parent_values(self);
	PyTuple_SET_ITEM(self->val_tuple,0,self->value);	
	new_logp = PyObject_Call(self->logp_fun, self->val_tuple, self->parent_value_dict);
	Py_INCREF(new_logp);	
	Py_DECREF(self->logp);
	self->logp = new_logp;
}

static int param_check_for_recompute(Parameter *self)
{
	int i,j,recompute,index_now;
	recompute = 1;
	for(i=0;i<2;i++)
	{
		if(self->timestamp_caches[i]==self->timestamp)
		{
			recompute = 0;
			for(j=0;j<self->N_node_parents;j++)
			{
				index_now = self->node_parent_indices[j];
				if(self->parent_timestamp_caches[i][index_now] != (int) PyInt_AS_LONG(PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp")))
				//if(self->parent_timestamp_caches[i][index_now]! =((Node*) self->parent_pointers[index_now])->timestamp)
				{
					recompute = 1;
					break;
				}
			}
			
			if(recompute==0)
			{
				for(j=0;j<self->N_param_parents;j++)
				{
					index_now = self->param_parent_indices[j];
					if(self->parent_timestamp_caches[i][index_now]!=(int) PyInt_AS_LONG(PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp")))
					//if(self->parent_timestamp_caches[i][index_now]!=((Parameter*) self->parent_pointers[index_now])->timestamp)
					{
						recompute = 1;
						break;
					}
				}				
			}
		}
		if(recompute==0)
		{
			return i;
		}
	}
	return -1;
}

static void param_cache(Parameter *self)
{
	int j, index_now, dummy;
	Py_INCREF(self->logp);
	Py_DECREF(self->logp_caches[1]);
	self->logp_caches[1] = self->logp_caches[0];
	self->logp_caches[0] = self->logp;
	
	self->timestamp_caches[1] = self->timestamp_caches[0];
	self->timestamp_caches[0] = self->timestamp;
	
	for(j=0;j<self->N_node_parents;j++)
	{
		index_now = self->node_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		//dummy = ((Node*) self->parent_pointers[index_now])->timestamp;
		//self->parent_timestamp_caches[0][index_now] = dummy;
		self->parent_timestamp_caches[0][index_now] = (int) PyInt_AS_LONG(PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp"));
	}
	
	for(j=0;j<self->N_param_parents;j++)
	{
		index_now = self->param_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		//dummy = ((Parameter*) self->parent_pointers[index_now])->timestamp;
		//self->parent_timestamp_caches[0][index_now] = dummy;
		self->parent_timestamp_caches[0][index_now] = (int) PyInt_AS_LONG(PyObject_GetAttrString(self->parent_pointers[index_now], "timestamp"));
	}
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
	if(self->isdata == 1) PyErr_SetString(PyExc_AttributeError, "Data objects' values cannot be set.");
	else{	
		Py_DECREF(self->last_value);
		Py_INCREF(value);	
		self->last_value = self->value;
		self->value = value;
		if(self->reverted <1) self->timestamp++;
		else{
			self->timestamp += 2;
			self->reverted = 0;
		}
		return 0;
	}
}

// Get and set logp
static PyObject * 
Parameter_getlogp(Parameter *self, void *closure) 
{
	int i;
	i=param_check_for_recompute(self);
	
	if(i<0) 
	{
		compute_logp(self);
		param_cache(self);
	}
	
	else
	{
		Py_INCREF(self->logp_caches[i]);
		Py_DECREF(self->logp);
		self->logp = self->logp_caches[i];
	}
	
	Py_INCREF(self->logp); 
	//return Py_BuildValue("(O,[O,O])",self->logp, self->logp_caches[0], self->logp_caches[1]); 
	return self->logp;
} 

static int 
Parameter_setlogp(Parameter *self, PyObject *value, void *closure) 
{
	PyErr_SetString(PyExc_TypeError, "Parameter.logp cannot be set."); 
	return -1;
}

// Get and set timestamp
static PyObject * 
Parameter_gettimestamp(Parameter *self, void *closure) 
{
	PyObject *timestamp;
	timestamp = Py_BuildValue("i", self->timestamp);
	
	Py_INCREF(timestamp); 
	return timestamp;
}
static int 
Parameter_settimestamp(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_TypeError, "Parameter.timestamp cannot be set."); 
	return -1;
}

// Get/set function table
static PyGetSetDef Param_getseters[] = { 

	{"timestamp", 
	(getter)Parameter_gettimestamp, (setter)Parameter_settimestamp, 
	"timestamp of Parameter", 
	NULL}, 

	{"logp", 
	(getter)Parameter_getlogp, (setter)Parameter_setlogp, 
	"logp of Parameter", 
	NULL}, 
	
	{"value", 
	(getter)Parameter_getvalue, (setter)Parameter_setvalue, 
	"value of Parameter", 
	NULL},	

	{NULL} /* Sentinel */ 
};


// Method Parameter.revert()
static PyObject*
Param_revert(Parameter *self)
{
	Py_INCREF(self->last_value);
	Py_DECREF(self->value);
	
	self->value = self->last_value;
	self->timestamp--;	
	Parameter_getlogp(self, self);
	
	self->reverted = 1;
	
	return Py_None;
}

// Method Parameter.random()
static PyObject*
Param_random(Parameter *self)
{
	PyObject *new_value;
	if (!self->random_fun) PyErr_SetString(PyExc_TypeError, "No random() function specified.");
	
	param_parent_values(self);
	new_value = PyObject_Call(self->random_fun, PyTuple_New(0), self->parent_value_dict);
	Py_INCREF(new_value);
	Py_DECREF(self->last_value);
	self->last_value = self->value;
	self->value = new_value;
	if(self->reverted <1) self->timestamp++;
	else{
		self->timestamp += 2;
		self->reverted = 0;
	}
	
	Py_INCREF(self->value);
	return self->value;
}

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