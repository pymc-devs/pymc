class value_desc(object):
	def __init__(self):
		self.__value = 0
	def __get__(self,instance,owner):
		print 'self' , self
		print 'instance' , instance
		print 'owner' , owner
		return self.__value
	def __set__(self,instance,value):
		self.__value = value
	def __call__(self):
		return self.__value ** 2
		
		
class ParameterHolder(object):
	def __init__(self):
		self.__value = 3
	def getval(self):
		return self.__value
	def setval(self,new_value):
		self.__value = new_value
	value = value_desc()
	
B=ParameterHolder()
A=value_desc()