#ifndef _PYMCOBJECTS_H_
#define _PYMCOBJECTS_H_


/*****************
 *
 *	PYMCBASE
 *
 *****************/

// Declarations for dummy object PyMCBase
typedef struct {
	PyObject_HEAD
	PyObject *value;
} PyMCBase;

static PyTypeObject PyMCBasetype;

static char PyMCBasetype__doc__[] = 
"The base PyMC object. Parameter and Node inherit from this class.\n\n"

"PyMCBase cannot be instantiated.\n\n"

"See also Parameter and Node,"
"as well as parameter(), node(), and data().";




/*******************
 *
 *	REMOTEPROXYBASE
 *
 *******************/

// Declarations for dummy object RemoteProxyBase.
typedef struct {
	PyObject_HEAD
	PyObject *value;
} RemoteProxyBase;

static PyTypeObject RemoteProxyBasetype;

static char RemoteProxyBasetype__doc__[] = 
"The base remote proxy object. Cannot be instantiated.\n\n"

"See also RemoteProxy.";



/*****************
 *
 *	NODE
 *
 *****************/

// Declarations for objects of type Node
// Docstring for Node
static char Nodetype__doc__[] = 
"A PyMC object whose value is determined by its parents.\n"
"A subclass of PyMCBase.\n\n"

"Externally-accessible attributes:\n\n"

"\tvalue:\t\tValue conditional on parents. Retrieved from cache when possible,\n"
"\t\t\trecomputed only when necessary.\n\n"

"\tparents :\tA dictionary containing parents of self with\n"
"\t\t\tparameter names. Parents can be any type.\n\n"

"\tchildren :\tA set containing children of self.\n"
"\t\t\tChildren must be PyMC objects.\n\n"

"\ttrace:\t\tSelf's trace object.\n\n"

"To instantiate: see node()\n\n"

"See also Parameter and PyMCBase,\n"
"as well as parameter() and data().";

typedef struct {
	PyObject_HEAD
	PyObject *value;
	PyObject *eval_fun;
	int timestamp;
	PyObject *parents;
	PyObject *children;
	PyObject *__doc__;
	PyObject *__name__;
	PyObject *trace;

	int N_parents;

	int N_pymc_parents;
	int N_constant_parents;
	int N_proxy_parents;

	int *pymc_parent_indices;
	int *constant_parent_indices;
	int *proxy_parent_indices;
	
	PyObject **parent_pointers;
	PyObject **parent_keys;
	PyObject **parent_values;
	PyObject *parent_value_dict;

	
	PyObject *value_caches[2];
	int timestamp_caches[2];
	int *parent_timestamp_caches[2];
} Node;

static PyTypeObject Nodetype;
static int Node_init(Node *self, PyObject *args, PyObject *kwds);

static void parse_parents_of_node(Node *self);

static void node_parent_values(Node *self);

static int node_check_for_recompute(Node *self);
static void node_cache(Node *self);

static PyObject *Node_getvalue(Node *self, void *closure);
static int Node_setvalue(Node *self, PyObject *value, void *closure);

static PyObject* Node_gettimestamp(Node *self, void *closure);
static int Node_settimestamp(Node *self, PyObject *value, void *closure);

static void Node_dealloc(Node* self);


// Empty methods table for Node
static PyMethodDef Node_methods[] = { 
	{NULL,		NULL}		/* sentinel */
};


// Node's member table
static PyMemberDef Node_members[] = { 
{"parents", T_OBJECT, offsetof(Node, parents), RO, 
"parents"},	
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





/*****************
 *
 *	PARAMETER
 *
 *****************/

// Docstring for Parameter
static char Paramtype__doc__[] = 
"A PyMC object whose value is not completely determined by its parents.\n"
"Includes both unknown parameters and data.\n"
"Subclass of PyMCBase.\n\n"

"Externally-accessible attributes:\n\n"

"\tlogp :\t\tLog-probability of self's current value conditional on parents.\n"
"\t\t\tRetrieved from cache when possible, recomputed only when necessary.\n\n"

"\tvalue:\t\tCurrent value. Can be any type. Read-only if self is data.\n\n"

"\tparents:\tA dictionary containing parents of self with\n"
"\t\t\tparameter names. Parents can be any type.\n\n"

"\tchildren:\tA set containing children of self.\n"
"\t\t\tChildren must be PyMC objects.\n\n"

"\ttrace:\t\tSelf's trace object.\n\n"

"Externally-accessible methods:\n\n"

"\trevert():\tReturn value to last value, decrement timestamp.\n\n"

"\trandom():\tIf random_fun is defined, this draws a value for self.value from\n"
"\t\t\tself's distribution conditional on self.parents. Used for\n"
"\t\t\tmodel averaging.\n\n"

"To instantiate as unknown parameter: see parameter().\n"
"To instantiate as data: see data().\n\n"

"See also PyMCBase and Node,\n"
"as well as node()";

// Declarations for objects of type Parameter 
typedef struct {
	PyObject_HEAD
	PyObject *value;
	PyObject *last_value;
	PyObject *logp;
	PyObject *logp_fun;
	int timestamp;
	int max_timestamp;
	int reverted;
	PyObject *parents;
	PyObject *children;
	PyObject *__doc__;
	PyObject *__name__;
	PyObject *random_fun;
	PyObject *trace;
	PyObject *rseed;
	PyObject *val_tuple;	
	int isdata;
	
	int N_parents;

	int N_pymc_parents;
	int N_constant_parents;
	int N_proxy_parents;

	int *pymc_parent_indices;
	int *constant_parent_indices;
	int *proxy_parent_indices;
	
	PyObject **parent_pointers;
	PyObject **parent_keys;
	PyObject **parent_values;
	PyObject *parent_value_dict;

	PyObject *logp_caches[2];	
	int timestamp_caches[2];
	int *parent_timestamp_caches[2];
} Parameter;

static PyTypeObject Paramtype;
static int Param_init(Parameter *self, PyObject *args, PyObject *kwds);

static void parse_parents_of_param(Parameter *self);

static void param_parent_values(Parameter *self);

static int param_check_for_recompute(Parameter *self);
static void param_cache(Parameter *self);

static PyObject *Parameter_getvalue(Parameter *self, void *closure);
static int Parameter_setvalue(Parameter *self, PyObject *value, void *closure);

static PyObject *Parameter_getlogp(Parameter *self, void *closure);
static int Parameter_setlogp(Parameter *self, PyObject *value, void *closure);

static PyObject* Parameter_gettimestamp(Parameter *self, void *closure);
static int Parameter_settimestamp(Parameter *self, PyObject *value, void *closure);

static PyObject * Parameter_getisdata(Parameter *self, void *closure);
static int Parameter_setisdata(Parameter *self, PyObject *value, void *closure);

static char Param_random__doc__[] = "Sample self conditional on parents.";
static PyObject* Param_random(Parameter *self);

static char Param_revert__doc__[] = "Call this when rejecting a jump.";
static PyObject* Param_revert(Parameter *self);

static void Param_dealloc(Parameter *self);



// Members table for Parameter
static PyMemberDef Param_members[] = { 
{"parents", T_OBJECT, offsetof(Parameter, parents), RO, 
"parents"},
// {"last_value", T_OBJECT, offsetof(Parameter, last_value), RO, 
// "last_value"},
{"rseed", T_OBJECT, offsetof(Parameter, rseed), 0, 
"rseed"},
{"children", T_OBJECT, offsetof(Parameter, children), 0, 
"children"},
{"__doc__", T_OBJECT, offsetof(Parameter, __doc__), 0, 
"__doc__"},
{"__name__", T_OBJECT, offsetof(Parameter, __name__), 0, 
"__name__"},
{"trace", T_OBJECT, offsetof(Parameter, trace), 0, 
"trace"},
{NULL} /* Sentinel */ 
};


// Methods table for Parameter
static PyMethodDef Param_methods[] = {
	{"revert",	(PyCFunction)Param_revert,	METH_VARARGS,	Param_revert__doc__},
 {"random",	(PyCFunction)Param_random,	METH_VARARGS,	Param_random__doc__},
	{NULL,		NULL}		/* sentinel */
};



static int downlow_gettimestamp(Parameter *self)
{return self->timestamp;}

#endif /* _PYMCOBJECTS_H_ */
