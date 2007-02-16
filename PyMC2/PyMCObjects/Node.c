#ifndef _NODE_H_
#define _NODE_H_


#include "Python.h"
#include "structmember.h"
#include "PyMCObjects.h"


//__init__
static int 
Node_init(Node *self, PyObject *args, PyObject *kwds) 
{	
	PyObject *AbstractBase, *newset_fun;
	int i;
	void *closure;
		
	static char *kwlist[] = {	"eval",
								"name",
								"parents",
								"doc",
								"trace",
								NULL};
								
	if (!PyArg_ParseTupleAndKeywords(	args, kwds, "OOO|OO", kwlist, 
										&self->eval_fun,
										&self->__name__,
										&self->parents,
										&self->__doc__,
										&self->trace)) 
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
	Py_DECREF(newset_fun);	
	
	if(!self->__doc__) self->__doc__ = self->__name__;
	if(!self->trace) self->trace = Py_BuildValue("i",1);
	
	for(i=0;i<2;i++)
	{
		self->value_caches[i] = Py_BuildValue("");
	}
	
	if(!PyDict_Check(self->parents)){
		PyErr_SetString(PyExc_TypeError, "Argument parents must be a dictionary.");
		return -1;	
	}
	
	if(!PyFunction_Check(self->eval_fun)){
		PyErr_SetString(PyExc_TypeError, "Argument eval must be a function.");
		return -1;
	}
	
	// Guard against subclassing
	if(!PyObject_TypeCheck(self, &Nodetype)){
		PyErr_SetString(PyExc_TypeError, "Node cannot be subclassed");
		return -1;
	}	
	
	Py_INCREF(self->eval_fun);
	Py_INCREF(self->__name__);
	Py_INCREF(self->parents);
	Py_INCREF(self->children);
	Py_XINCREF(self->__doc__);
	Py_XINCREF(self->trace);	
	
	parse_parents_of_node(self);
	if(PyErr_Occurred()) return -1;		
	
	self->timestamp = 0;
	self->value = Py_None;
	Py_INCREF(Py_None);	
	Node_getvalue(self, closure);
	if(PyErr_Occurred()) return -1;		
	
	
	Py_INCREF(self->children);	
	PyObject_CallMethodObjArgs(self->children, Py_BuildValue("s","clear"), NULL, NULL); 

	if (PyErr_Occurred()) return -1;

	return 0;
}

static void Node_dealloc(Node* self)
{
	
	int i, index_now;
	
	Py_XDECREF(self->value);
	Py_XDECREF(self->eval_fun);
	
	Py_XDECREF(self->parents);
	Py_XDECREF(self->children);
	Py_XDECREF(self->__doc__);
	Py_XDECREF(self->__name__);
	Py_XDECREF(self->trace);
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
	
	for(i=0;i<2;i++) Py_XDECREF(self->value_caches[i]);
	
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


static void parse_parents_of_node(Node *self)
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
				OK = node_fileitem(self, contained_parent, NULL, -1);
				if(OK<0) return;
			}
			Py_DECREF(iter);
		}
		
		// Otherwise, sort this parent.
		else 
		{
			OK=node_fileitem(self, parent_now, key_now, i);
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

static int node_fileitem(Node *self, PyObject *parent_now, PyObject *key_now, int i)
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



static void node_parent_values(Node *self)
{
	int index_now, i;
	for(i=0;i<self->N_pymc_parents;i++)
	{
		index_now = self->pymc_parent_indices[i];
		Py_XDECREF(self->parent_values[index_now]);
		self->parent_values[index_now] = PyObject_GetAttrString(self->dict_parents[index_now],"value");
		PyDict_SetItem(	self->parent_value_dict, self->parent_keys[index_now], 
						self->parent_values[index_now]);
	}
}


static int node_check_for_recompute(Node *self)
{
	int i,j,recompute,index_now;
	PyObject *timestamp;
	recompute = 1;
	
	// Go through both cache levels
	for(i=0;i<2;i++)
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

		if(recompute==0)
		{
			return i;
		}
	}
	return -1;
}


static void node_cache(Node *self)
{
	int j, index_now;
	Py_INCREF(self->value);
	Py_DECREF(self->value_caches[1]);
	PyObject *timestamp;
	self->value_caches[1] = self->value_caches[0];
	self->value_caches[0] = self->value;
	
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
Node_getvalue(Node *self, void *closure) 
{
	int i;
	PyObject *new_value;
	node_parent_values(self);	
	i=node_check_for_recompute(self);
	if(PyErr_Occurred()) return NULL;					
	//i=-1;
	
	if(i<0) 
	{		
		new_value = PyObject_Call(self->eval_fun, PyTuple_New(0), self->parent_value_dict);
		if(PyErr_Occurred()) return NULL;				
		else{
			Py_DECREF(self->value);
			self->value = new_value;
			self->timestamp++;
			node_cache(self);					
		}

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
	PyErr_SetString(PyExc_AttributeError, "Node.value cannot be set."); 
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
	PyErr_SetString(PyExc_AttributeError, "Node.timestamp cannot be set."); 
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
	0, /* tp_base */ 
	0, /* tp_dict */ 
	0, /* tp_descr_get */ 
	0, /* tp_descr_set */ 
	0, /* tp_dictoffset */ 
	(initproc)Node_init, /* tp_init */ 
	0, /* tp_alloc */ 
	PyType_GenericNew, /* tp_new */			
};

#endif /* _NODE_H_ */