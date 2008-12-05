"""Substitute chapters -> sections, sections -> subsections, etc..."""

import re

def manual2article(text):
    text = re.sub(r"\\subsubsection\*?", r"\\paragraph", text)
    text = re.sub(r"\\subsection\*?", r"\\subsubsection", text)
    text = re.sub(r"\\section\*?", r"\\subsection", text)
    text = re.sub(r"\\chapter", r"\\section", text)
    text = re.sub(r"\\hypertarget\{.*", r"", text)
    text = re.sub(r"\\pdfbookmark.*", r"", text)
    return text

if __name__ == '__main__':
    import sys
    f = open(sys.argv[1], 'r')
    text = f.read()
    print manual2article(text)
