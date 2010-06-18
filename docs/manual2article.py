"""Substitute chapters -> sections, sections -> subsections, etc..."""

import re

def manual2article(text):
    text = re.sub(r"\\subsubsection\*?", r"\\paragraph", text)
    text = re.sub(r"\\subsection\*?", r"\\subsubsection", text)
    text = re.sub(r"\\section\*?", r"\\subsection", text)
    text = re.sub("section ","Section~",text)
    text = re.sub("Section ","Section~",text)
    text = re.sub(r"\\chapter", r"\\section", text)
    text = re.sub("chapter ","Section~",text)    
    text = re.sub("Chapter ","Section~",text)
    text = re.sub(" chapter", " section",text)
    text = re.sub("figure[ ]+", "figure~",text)
    text = re.sub("Figure[ ]+", "Figure~",text)
    text = re.sub("table[ ]+", "table~",text)
    text = re.sub("Table[ ]+", "Table~",text)
    for code_alias in ['module','texttt','function','file','class']:
        text = re.sub(r"\\%s{"%code_alias, "\code{", text)
    text = re.sub(r'^T',r'\\top',text)
    
    # Dedent all code blocks
    re.DOTALL=True
    codeblock = re.compile(r'\\begin{verbatim}[^&]*?\\end{verbatim}')
    codeblocks = codeblock.findall(text)
    for match in codeblocks:
        if match.count(r'\\begin')>0:
            continue
        text = text.replace(match, match.replace('\n\t','\nTAB').replace('\n    ','\nTAB').replace('\n   ','\nTAB').replace('TAB',''))
    
    # Destroy all boxes - This seems to remove more than expected. 
    boxblock = re.compile(r'\\fbox{[^#]*}')
    boxblocks = boxblock.findall(text)
    for match in boxblocks:
        text = text.replace(match, '')
    
    # Convert verbatim blocks to CodeInput blocks
    text = re.sub(r'\\begin{verbatim}',r'\\begin{CodeInput}',text)
    text = re.sub(r'\\end{verbatim}',r'\\end{CodeInput}',text)
    

    # Proglang and pkg
    text = re.sub(r"\\pdfbookmark.*", r"", text)
    for pkgname in ['PyMC','NumPy','SciPy','PyTables','Matplotlib','Pylab','Pyrex','WinBUGS','JAGS','Hierarchical Bayes Compiler','pyTables','pydot','IPython','nose','gfortran','gcc','Enthought Python Distribution','Gnu Compiler Collection','GCC','g77','subversion']:
        text = re.sub(pkgname, r'\\pkg{%s}'%pkgname,text)
    for proglang in ['Python','Fortran']:
        text = re.sub(' ' + proglang, r' \\proglang{%s}'%proglang, text)
        
        
    # Convert hyperlinks to citations, or make them explicit, or remove them.
    text = text.replace(r'\href{http://www.python.org/doc/}{\proglang{Python} documentation}',r'\proglang{Python} documentation \citep{python}')
    text = text.replace(r'\href{http://code.google.com/p/pymc/downloads/list}{download} page', r'download page at \href{http://code.google.com/p/pymc/downloads/list}')
    text = text.replace(r'\href{http://docs.python.org/glossary.html#term-decorator}{decorator}', r'decorator \citep{python}')
    text = text.replace(r'\href{http://code.google.com/p/pymc/wiki/Benchmarks}{the wiki page on benchmarks}', r'\pkg{PyMC}\'s wiki page on benchmarks at \href{http://code.google.com/p/pymc/wiki/Benchmarks}')
    text = text.replace(r'\href{http://www.cosc.canterbury.ac.nz/greg.ewing/python/\pkg{Pyrex}/}{\pkg{Pyrex}} ', r'\pkg{Pyrex} \citep{pyrex}')
    text = text.replace(r'\href{http://en.wikipedia.org/wiki/Topological_sort}{topological sorting}', r'topological sorting')
    text = text.replace(r'\href{http://lib.stat.cmu.edu/R/CRAN/}{R statistical package}', r'\proglang{R} language \citep{R}')
    text = text.replace(r'\href{http://www-fis.iarc.fr/coda/}{CODA module}',r'\pkg{CODA} module \citep{coda}')
    text = text.replace(r'\href{pymc@googlegroups.com}{mailing list}', r'mailing list at \href{pymc@googlegroups.com}')
    text = text.replace(r'\href{http://code.google.com/p/pymc/w/list}{wiki page}', r'wiki page at \href{http://code.google.com/p/pymc/w/list}')
    text = text.replace(r'\href{http://www.map.ox.ac.uk}{Malaria Atlas Project}', r'Malaria Atlas Project')
    text = text.replace(r'{R}',r'{r}')
    text = text.replace(r'(\cite{dawidmarkov,Jordan:2004p5439})', r'\citep{dawidmarkov,Jordan:2004p5439}')
    text = text.replace(r'\cite{',r'\citet{')

    return text

if __name__ == '__main__':
    import sys
    f = open(sys.argv[1], 'r')
    text = f.read()
    print manual2article(text)
