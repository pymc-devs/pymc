# Glossary

A glossary of common terms used throughout the PyMC3 documentation and examples.

:::::{glossary}
[Term with external reference](https://www.youtube.com/watch?v=6dc7JgR8eI0)
  Terms are defined within this glossary directive. The term id is defined as the non
  indented line, and can be text alone (like {term}`second term`) or also include a link
  to an external reference.

Second term
  Definitions can have as many lines as desired, and should be written in markdown. Definitions
  can contain any markdown formatting for MyST to parse, this includes basic markdown like **bold**
  as well as MyST directives and roles like {fa}`fort awesome,style=fab`
Functional Programming
  Functional programming is a programming style that prefers the use of basic functions with explicit and distinct inputs and outputs.
  This contrasts with functions or methods that depend on variables that are not explicitly passed as an input (such as accessing `self.variable` inside a method) or that alter the inputs or other state variables in-place, instead of returning new distinct variables as outputs.
Dispatching
  Choosing which function or method implementation to use based on the type of the input variables (usually just the first variable). For some examples, see Python's documentation for the [singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch) decorator.
:::::
