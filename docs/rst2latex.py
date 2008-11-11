#!/usr/bin/env python

"""
A minimal front end to the Docutils Publisher, producing LaTeX.
"""

try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

from docutils.parsers.rst.roles import register_canonical_role
from docutils import nodes
from docutils.writers.latex2e import LaTeXTranslator
from docutils.parsers.rst.directives import _directives
from docutils.core import publish_cmdline, default_description


# Define LaTeX math node:
class latex_math(nodes.Element):
    tagname = '#latex-math'
    def __init__(self, rawsource, latex):
        nodes.Element.__init__(self, rawsource)
        self.latex = latex

# Register role:
def latex_math_role(role, rawtext, text, lineno, inliner,
                    options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    node = latex_math(rawtext, latex)
    return [node], []
register_canonical_role('latex-math', latex_math_role)


# Register directive:
def latex_math_directive(name, arguments, options, content, lineno,
                         content_offset, block_text, state, state_machine):
    latex = ''.join(content)
    node = latex_math(block_text, latex)
    return [node]
latex_math_directive.arguments = None
latex_math_directive.options = {}
latex_math_directive.content = 1
_directives['latex-math'] = latex_math_directive


# Add visit/depart methods to HTML-Translator:
def visit_latex_math(self, node):
    inline = isinstance(node.parent, nodes.TextElement)
    if inline:
        self.body.append('$%s$' % node.latex)
    else:
        self.body.extend(['\\begin{equation*}\\begin{split}',
                          node.latex,
                          '\\end{split}\\end{equation*}'])
def depart_latex_math(self, node):
    pass
LaTeXTranslator.visit_latex_math = visit_latex_math
LaTeXTranslator.depart_latex_math = depart_latex_math


description = ('Generates LaTeX documents from standalone reStructuredText '
               'sources.  ' + default_description)

publish_cmdline(writer_name='latex', description=description)
