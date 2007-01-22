#ifndef _PYMCOBJECTS_H_
#define _PYMCOBJECTS_H_

// Declarations for dummy object PyMCBase
typedef struct {
	PyObject_HEAD
	PyObject *value;
} PyMCBase;
static PyTypeObject PyMCBasetype;

// Declarations for objects of type Parameter 
typedef struct {
	PyObject_HEAD
	PyObject *value;
	PyObject *logp;
	PyObject *logp_fun;
	int timestamp;
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

	int N_node_parents;
	int N_param_parents;
	int N_constant_parents;

	int *node_parent_indices;
	int *param_parent_indices;
	int *constant_parent_indices;
	
	PyObject **parent_pointers;
	PyObject **parent_keys;
	PyObject **parent_values;
	PyObject *parent_value_dict;

	int caching;
	PyObject *value_caches[2];
	PyObject *logp_caches[2];	
	int timestamp_caches[2];
	int parent_timestamp_caches[2];
	int cache_position;
} Parameter;
static PyTypeObject Paramtype;

// Declarations for objects of type Node
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

	int N_node_parents;
	int N_param_parents;
	int N_constant_parents;

	int *node_parent_indices;
	int *param_parent_indices;
	int *constant_parent_indices;
	
	PyObject **parent_pointers;
	PyObject **parent_keys;
	PyObject **parent_values;
	PyObject *parent_value_dict;


	int caching;	
	PyObject *value_caches[2];
	int timestamp_caches[2];
	int parent_timestamp_caches[2];
} Node;
static PyTypeObject Nodetype;


#endif /* _PYMCOBJECTS_H_ */
