#ifndef _PARAMETER_H_
#define _PARAMETER_H_

#include "Python.h"
#include "structmember.h"
#include "PyMCObjects.h"

//__init__
static int 
Param_init(Parameter *self, PyObject *args, PyObject *kwds) 
{
	PyObject *AbstractBase, *newset_fun;
	int i;
	void *closure;
	static char *kwlist[] = {	"logp",  
								"name", 
								"value", 
								"parents", 
								"doc",								
								"random", 
								"trace",  
								"rseed", 
								"isdata", 								
								NULL};
								
	if (! PyArg_ParseTupleAndKeywords(	args, kwds, "OOOO|OOOOi", kwlist, 
										&self->logp_fun,
										&self->__name__, 
										&self->value, 										
										&self->parents,
										&self->__doc__,																				
										&self->random_fun, 
										&self->trace, 
										&self->rseed, 
										&self->isdata )) 
										
	{
		return -1;
	}
	
	AbstractBase = (PyObject*) PyImport_ImportModule("AbstractBase");
	self->PyMCBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "PyMCBase");
	self->ParameterBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "ParameterBase");
	self->NodeBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "NodeBase");	
	self->PurePyMCBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "PurePyMCBase");
	self->ContainerBase = (PyTypeObject*) PyObject_GetAttrString(AbstractBase, "ContainerBase");

	// This is annoying.
	newset_fun = (PyObject*) PyObject_GetAttrString(AbstractBase, "new_set");
	self->set_iter_getter = (PyObject*) PyObject_GetAttrString(AbstractBase, "set_iter");
	self->set_length_getter = (PyObject*) PyObject_GetAttrString(AbstractBase, "set_length");
	
	if(PyErr_Occurred()) return -1;		
	
	self->children =  PyObject_Call(newset_fun, PyTuple_New(0), NULL);
	if(PyErr_Occurred()) return -1;		
	
	Py_DECREF(newset_fun);
	Py_DECREF(AbstractBase);	
	
	// Initialize optional arguments.
	if(!self->__doc__) self->__doc__ = self->__name__;
	if(!self->isdata) self->isdata = 0;
	if(!self->trace) self->trace = Py_True;
	
	self->reverted = 1;
		
	for(i=0;i<2;i++)
	{
		self->logp_caches[i] = Py_None;
		Py_INCREF(self->logp_caches[i]);
	} 
	self->last_value = Py_None;
	Py_INCREF(self->last_value);		
	
	if(!PyDict_Check(self->parents)){
		PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
		return -1;	
	}
	
	if(!PyFunction_Check(self->logp_fun)){
		PyErr_SetString(PyExc_TypeError, "Argument logp must be a function.");
		return -1;
	}
	
	// Guard against subclassing
	if(!PyObject_TypeCheck(self, &Paramtype)){
		PyErr_SetString(PyExc_TypeError, "Parameter cannot be subclassed");
		return -1;
	}

	// Incref all PyObjects so they don't go away.
	Py_INCREF(self->logp_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->value);
	Py_INCREF(self->parents);
	Py_INCREF(self->__doc__);
	Py_INCREF(self->random_fun);
	Py_INCREF(self->trace);
	Py_INCREF(self->rseed);
	Py_INCREF(self->children);
	
	// Initialize PyObjects that aren't passed in.
	self->val_tuple = PyTuple_New(1);
	PyTuple_SET_ITEM(self->val_tuple,0,self->value);	

	parse_parents_of_param(self);
	if(PyErr_Occurred()) return -1;		
	
	for(i=0;i<2;i++) self->timestamp_caches[i] = -1;
	self->timestamp = 0;
	self->max_timestamp = self->timestamp;
	
	self->logp = Py_None;
	Py_INCREF(Py_None);
	Parameter_getlogp(self, closure);
	if(PyErr_Occurred()) return -1;

	if(PyErr_Occurred()) return -1;		
	param_cache(self);			

	PyObject_CallMethodObjArgs(self->children, Py_BuildValue("s","clear"), NULL, NULL);		
	if(PyErr_Occurred()) return -1;	
	return 0; 
} 

static void Param_dealloc(Parameter *self)
{
	int i;
	
	Py_XDECREF(self->value);
	Py_XDECREF(self->last_value);
	Py_XDECREF(self->logp);
	Py_XDECREF(self->logp_fun);
	Py_XDECREF(self->parents);
	Py_XDECREF(self->children);
	Py_XDECREF(self->__doc__);
	Py_XDECREF(self->__name__);
	Py_XDECREF(self->random_fun);
	Py_XDECREF(self->trace);
	Py_XDECREF(self->rseed);
	Py_XDECREF(self->val_tuple);	
	Py_XDECREF(self->parent_value_dict);
	Py_XDECREF(self->PyMCBase);
	Py_XDECREF(self->PurePyMCBase);	
	Py_XDECREF(self->ContainerBase);
	Py_XDECREF(self->set_iter_getter);
	Py_XDECREF(self->set_length_getter);
	
	
	for(  i = 0; i < self->N_dict_parents; i ++ )
	{
		Py_XDECREF(self->parent_values[i]);
		Py_XDECREF(self->dict_parents[i]);
		Py_XDECREF(self->parent_keys[i]);
	}
	
	for(i=0; i<self->N_parents; i++)
	{
		Py_XDECREF(self->ultimate_parents[i]);
	}

	PyObject *logp_caches[2];	
	for(i=0;i<2;i++) Py_XDECREF(self->logp_caches[i]);
	
	free(self->parent_keys);
	free(self->ultimate_parents);
	free(self->parent_values);
	free(self->dict_parents);

	free(self->pymc_parent_indices);
	free(self->constant_parent_indices);

	free(self->pure_parent_indices);
	free(self->node_parent_indices);
	free(self->param_parent_indices);
}

static void parse_parents_of_param(Parameter *self)
{

	PyObject *parent_items; 
	PyObject *parent_now;
	PyObject *key_now;
	PyObject *contained_parent;
	PyObject *iter;
	int i, j, OK;	

	//printf("\n\n%p:\n",self);	
	// Number of parents in dictionary (may include containers)
	self->N_dict_parents = (int) PyDict_Size(self->parents);
	parent_items =  PyDict_Items(self->parents);
	
	// Count total number of parents (unpack containers)
	self->N_parents = self->N_dict_parents;
	for(i=0;i<self->N_dict_parents;i++)
	{
		parent_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 1);
		if(PyObject_IsInstance(parent_now, (PyObject*) self->ContainerBase))
		{
			//printf("Found container: %p\n",parent_now);
			self->N_parents -= 1;
			PyTuple_SET_ITEM(self->val_tuple,0,PyObject_GetAttrString(parent_now,"pymc_objects"));
			if(PyErr_Occurred()) return;
			//printf("Length: %i\n",PyInt_AS_LONG(PyObject_Call(self->set_length_getter, self->val_tuple, NULL)));
			self->N_parents += (int) PyInt_AS_LONG(PyObject_Call(self->set_length_getter, self->val_tuple, NULL));
			if(PyErr_Occurred()) return;			
		}
	}
	//printf("N_dict_parents: %i\n",self->N_dict_parents);
	//printf("N_parents: %i\n",self->N_parents);

	// The indices of the parents in the dictionary that are PyMC objects or containers
	self->pymc_parent_indices = malloc(sizeof(int) * self->N_dict_parents);
	
	// The indices of the parents in the dictionary that are constants
	self->constant_parent_indices = malloc(sizeof(int) * self->N_dict_parents);
	

	// The indices of the ultimate parents that are parameters
	self->param_parent_indices = malloc(sizeof(int) * self->N_parents);

	// The indices of the ultimate parents that are nodes
	self->node_parent_indices = malloc(sizeof(int) * self->N_parents);
	
	// The indices of the ultimate parents that are pure PyMC objects
	self->pure_parent_indices = malloc(sizeof(int) * self->N_parents);
	
	// Ultimate parents
	self->ultimate_parents = malloc(sizeof(PyObject*) * self->N_parents);
		
	// Keys and values of parents in the dictionary
	self->parent_keys = malloc(sizeof(PyObject*) * self->N_dict_parents );
	self->parent_values = malloc(sizeof(PyObject*) * self->N_dict_parents );
	self->dict_parents = malloc(sizeof(PyObject*) * self->N_dict_parents );
	
	self->parent_value_dict = PyDict_New();
	
	self->N_pymc_parents = 0;
	self->N_constant_parents = 0;
	
	self->N_param_parents = 0;
	self->N_node_parents = 0;
	self->N_pure_parents = 0;
	
	self->ultimate_index = 0;
	for(i=0;i<self->N_dict_parents;i++)
	{
		
		//PyTuple_GetItem and PyDict_GetItem return borrowed references
		key_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 0);
		parent_now = PyTuple_GetItem(PyList_GetItem(parent_items, i), 1);
		if(PyErr_Occurred()) return;				
		self->parent_keys[i] = key_now;
		self->dict_parents[i] = parent_now;
		Py_INCREF(Py_None);
		self->parent_values[i] = Py_None;		
		
		Py_INCREF(parent_now);
		Py_INCREF(key_now);
		
		// If this parent is a contiainer:
		if(PyObject_IsInstance(parent_now, (PyObject*) self->ContainerBase))
		{
			//printf("Found container %p\n",parent_now);
			// Remember that it has a value attribute
			self->pymc_parent_indices[self->N_pymc_parents] = i;
			self->N_pymc_parents++;
			
			// And unpack its contents
			PyTuple_SET_ITEM(self->val_tuple,0,PyObject_GetAttrString(parent_now,"pymc_objects"));
			iter = PyObject_Call(self->set_iter_getter, self->val_tuple, NULL);
			if(PyErr_Occurred()) return;		
			//printf("iter: %p",iter);
			if(PyErr_Occurred()) 
			{
				//printf("Error occurred\n");
				return;
			}
			while(contained_parent = PyIter_Next(iter))
			{
				//printf("Unpacking container\n");
				Py_INCREF(contained_parent);
				OK = param_fileitem(self, contained_parent, NULL, -1);
				if(OK<0) return;
			}
			Py_DECREF(iter);
		}
		
		// Otherwise, sort this parent.
		else 
		{
			OK=param_fileitem(self, parent_now, key_now, i);
			if(OK<0) return;			
		}
	}
	
	// Malloc timestamp caches
	for(i=0;i<2;i++) 
	{
		self->parent_timestamp_caches[i] = malloc(sizeof(int) * self->N_parents );
		for(j=0;j<self->N_parents;j++) self->parent_timestamp_caches[i][j] = -1;
	}
	Py_DECREF(parent_items);
	//printf("dict parents: ");
	//for(i=0;i<self->N_dict_parents;i++) printf("%p ",self->dict_parents[i]);
	//printf("\nUltimate index: %i\n",self->ultimate_index);
	//printf("ultimate parents: ");
	//for(i=0;i<self->N_parents;i++) printf("%p ",self->ultimate_parents[i]);	
}

static int param_fileitem(Parameter *self, PyObject *parent_now, PyObject *key_now, int i)
{
	int claimed;
	
	claimed = 0;
	// If it's a pure object:
	if(PyObject_IsInstance(parent_now, (PyObject*) self->PurePyMCBase))
	{
		// Remember that it has a value attribute
		if(i>=0) 
		{
			self->pymc_parent_indices[self->N_pymc_parents] = i;			
			self->N_pymc_parents++;	
		}
		
		
		// Remember that it has a timestamp attribute
		self->pure_parent_indices[self->N_pure_parents] = self->ultimate_index;
		self->ultimate_parents[self->ultimate_index] = parent_now;
		self->N_pure_parents++;
		
		// Add self to its children
		PyObject_CallMethodObjArgs(PyObject_GetAttrString(parent_now, "children"), Py_BuildValue("s","add"), self, NULL);
		claimed=1;
	}
	// If it's a C parameter:
	if(PyObject_IsInstance(parent_now, (PyObject*) self->ParameterBase) && !claimed)
	{
		//printf("Found a parameter %p: %i\n",parent_now, i);
		// Remember value
		if(i>=0) 
		{
			self->pymc_parent_indices[self->N_pymc_parents] = i;
			self->N_pymc_parents++;								
		}
		
		
		// Remember timestamp
		self->param_parent_indices[self->N_param_parents] = self->ultimate_index;
		self->ultimate_parents[self->ultimate_index] = parent_now;		
		self->N_param_parents++;
		
		//Add self to children
		PyObject_CallMethodObjArgs(PyObject_GetAttrString(parent_now, "children"), Py_BuildValue("s","add"), self, NULL);
		claimed=1;				
	}
	// If it's a C node:
	if(PyObject_IsInstance(parent_now, (PyObject*) self->NodeBase) && !claimed)
	{
		// Remember value
		if(i>=0) {
			self->pymc_parent_indices[self->N_pymc_parents] = i;
			self->N_pymc_parents++;
		}
		
		
		// Remember timestamp
		self->node_parent_indices[self->N_node_parents] = self->ultimate_index;
		self->ultimate_parents[self->ultimate_index] = parent_now;		
		self->N_node_parents++;
		
		// Add self to children
		PyObject_CallMethodObjArgs(PyObject_GetAttrString(parent_now, "children"), Py_BuildValue("s","add"), self, NULL);
		claimed=1;
	}
	// If it's something else
	if(!claimed)
	{
		//printf("Found a constant %p: %i\n",parent_now, i);
		// Remember that it has no value attribute
		if(i>=0) 
		{
			self->constant_parent_indices[self->N_constant_parents] = i;			
			// Add it to the parent value dictionary
			Py_DECREF(self->parent_values[i]);		
			self->parent_values[i] = parent_now;
			PyDict_SetItem(self->parent_value_dict, key_now, self->parent_values[i]);
		}
		
		self->ultimate_parents[self->ultimate_index] = parent_now;		
		self->N_constant_parents++;
	}
	self->ultimate_index++;
	if(PyErr_Occurred()) return -1;
	else return 0;
}

static void param_parent_values(Parameter *self)
{
	//FIXME: This seems to be balanced on a knife-edge between not working and memory leaks.
	//FIXME: Try to fix it. Check out crash report when the decrefs are uncommented.
	int index_now, i;
	for(i=0;i<self->N_pymc_parents;i++)
	{
		index_now = self->pymc_parent_indices[i];
		Py_DECREF(self->parent_values[index_now]);
		self->parent_values[index_now] = PyObject_GetAttrString(self->dict_parents[index_now],"value");
		PyDict_SetItem(	self->parent_value_dict, self->parent_keys[index_now], 
						self->parent_values[index_now]);
	}
}

static int param_check_for_recompute(Parameter *self)
{
	int i,j,recompute,index_now;
	PyObject *timestamp;
	recompute = 1;
	
	// Go through both cache levels
	for(i=0;i<2;i++)
	{
		// Check for a mismatch in C parameter parents
		if(self->timestamp_caches[i]==self->timestamp)
		{
			recompute = 0;
			for(j=0;j<self->N_param_parents;j++)
			{
				index_now = self->param_parent_indices[j];
				if(	self->parent_timestamp_caches[i][index_now] != 
					((Parameter*) self->ultimate_parents[index_now])->timestamp)
				{
					recompute = 1;
					break;
				}
			}
			if(recompute==0)
			{
				// Check for a mismatch in C node parents
				for(j=0;j<self->N_node_parents;j++)
				{
					index_now = self->node_parent_indices[j];
					if(	self->parent_timestamp_caches[i][index_now] != 
						((Node*) self->ultimate_parents[index_now])->timestamp)
					{
						recompute = 1;
						break;
					}
				}
			}
			
			if(recompute==0)
			{
				// Check for mismatch in pure parameter and node parents
				for(j=0;j<self->N_pure_parents;j++)
				{
					index_now = self->pure_parent_indices[j];		
					timestamp = PyObject_GetAttrString(self->ultimate_parents[index_now], "timestamp");
					if(self->parent_timestamp_caches[i][index_now] != (int) PyInt_AS_LONG(timestamp));
					{
						recompute = 1;
						Py_DECREF(timestamp);
						break;
					}
					Py_DECREF(timestamp);
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
	int j, index_now;
	Py_INCREF(self->logp);
	Py_DECREF(self->logp_caches[1]);
	PyObject *timestamp;
	self->logp_caches[1] = self->logp_caches[0];
	self->logp_caches[0] = self->logp;
	
	self->timestamp_caches[1] = self->timestamp_caches[0];
	self->timestamp_caches[0] = self->timestamp;

	// Record timestamps of C parameter parents
	for(j=0;j<self->N_param_parents;j++)
	{
		index_now = self->param_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		self->parent_timestamp_caches[0][index_now] = ((Parameter*)(self->ultimate_parents[index_now]))->timestamp;
	}
	
	// Record timestamps of C node parents
	for(j=0;j<self->N_node_parents;j++)
	{
		index_now = self->node_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		self->parent_timestamp_caches[0][index_now] = ((Node*)(self->ultimate_parents[index_now]))->timestamp;
	}
	
	// Record timestamps of pure parameter and node parents
	for(j=0;j<self->N_pure_parents;j++)
	{
		index_now = self->pure_parent_indices[j];
		self->parent_timestamp_caches[1][index_now] = self->parent_timestamp_caches[0][index_now];
		timestamp = PyObject_GetAttrString(self->ultimate_parents[index_now], "timestamp");
		self->parent_timestamp_caches[0][index_now] = (int) PyInt_AS_LONG(timestamp);
		Py_DECREF(timestamp);
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
	if(self->isdata == 1) 
	{
		PyErr_SetString(PyExc_AttributeError, "Data objects' values cannot be set.");
		return -1;
	}
	else{
		self->reverted = 0;	
		Py_DECREF(self->last_value);
		self->last_value = self->value;
		Py_INCREF(value);	
		self->value = value;
		self->max_timestamp++;
		self->timestamp = self->max_timestamp;
		return 0;
	}
}

// Get and set logp
static PyObject * 
Parameter_getlogp(Parameter *self, void *closure) 
{
	int i;
	PyObject *new_logp;
	param_parent_values(self);	
	if(PyErr_Occurred()) return NULL;	
	i=param_check_for_recompute(self);
	//i=-1;
	
	if(i<0) 
	{
		PyTuple_SET_ITEM(self->val_tuple,0,self->value);		
		new_logp =  PyObject_Call(self->logp_fun, self->val_tuple, self->parent_value_dict);
		if(PyErr_Occurred()) return NULL;
		else{
			Py_DECREF(self->logp);		
			self->logp = new_logp;
			param_cache(self);			
		}
	}
	
	else
	{
		Py_INCREF(self->logp_caches[i]);
		Py_DECREF(self->logp);
		self->logp = self->logp_caches[i];
	}

	if(PyErr_Occurred()) return NULL;
	
	Py_INCREF(self->logp); 
	//return Py_BuildValue("(O,i)",self->logp, i); 
	return self->logp;
} 

static int 
Parameter_setlogp(Parameter *self, PyObject *value, void *closure) 
{
	PyErr_SetString(PyExc_AttributeError, "Parameter.logp cannot be set."); 
	return -1;
}


// Get and set timestamp
static PyObject * 
Parameter_gettimestamp(Parameter *self, void *closure) 
{
	PyObject *timestamp;
	timestamp = Py_BuildValue("i", self->timestamp);
	return timestamp;
}
static int 
Parameter_settimestamp(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_AttributeError, "Parameter.timestamp cannot be set."); 
	return -1;
}


// Get and set isdata
static PyObject * 
Parameter_getisdata(Parameter *self, void *closure) 
{
	PyObject *isdata;
	isdata = Py_BuildValue("i", self->isdata);
	return isdata;
}
static int 
Parameter_setisdata(Parameter *self, PyObject *value, void *closure) 
{ 
	PyErr_SetString(PyExc_AttributeError, "Parameter.isdata cannot be set."); 
	return -1;
}


// Get/set function table
static PyGetSetDef Param_getseters[] = { 
	
	{"isdata", 
	(getter)Parameter_getisdata, (setter)Parameter_setisdata, 
	"Flag indicating whether Parameter is data or no", 
	NULL}, 

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
	if(self->reverted == 1)
	{
		PyErr_SetString(PyExc_RuntimeError, "Parameter objects' values must be changed between calls to revert().");
		return NULL;		
	}
	self->reverted = 1;
	Py_INCREF(self->last_value);
	Py_DECREF(self->value);
	
	self->value = self->last_value;
	self->timestamp--;
	
	Py_INCREF(Py_None);
	return Py_None;
}

// Method Parameter.random()
static PyObject*
Param_random(Parameter *self)
{
	PyObject *new_value;
	
	if(self->isdata == 1) 
	{
		PyErr_SetString(PyExc_AttributeError, "Data objects' values cannot be set.");
		return NULL;
	}
	else{
		if (!self->random_fun | (self->random_fun == Py_None)) 
		{
			PyErr_SetString(PyExc_AttributeError, "No random() function specified.");
			return NULL;
		}		
	}
	
	param_parent_values(self);
	new_value = PyObject_Call(self->random_fun, PyTuple_New(0), self->parent_value_dict);
	if(PyErr_Occurred()) return NULL;
	else{
		self->reverted = 0;		
		Py_DECREF(self->last_value);
		self->last_value = self->value;
		self->value = new_value;
		self->max_timestamp++;
		self->timestamp = self->max_timestamp;
		
		Py_INCREF(self->value);
		return self->value;
	}
}

// Type declaration
static PyTypeObject Paramtype = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,						/*ob_size*/
	"Parameter",			/*tp_name*/
	sizeof(Parameter),		/*tp_basicsize*/
	0,				/*tp_itemsize*/
	(destructor)Param_dealloc,	/*tp_dealloc*/
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
	0, /* tp_base */ 
	0, /* tp_dict */ 
	0, /* tp_descr_get */ 
	0, /* tp_descr_set */ 
	0, /* tp_dictoffset */ 
	(initproc)Param_init, /* tp_init */ 
	0, /* tp_alloc */ 
	PyType_GenericNew, /* tp_new */
};

#endif /* _PARAMETER_H_ */