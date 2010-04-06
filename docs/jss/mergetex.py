import sys, re

fname=sys.argv[1]

text = file(fname).read()

pattern = re.compile(r'\\input{.*}')
matches = pattern.findall(text)

for m in matches:
    basename = re.sub(r'\\input{','',m)
    basename = re.sub(r'}','',basename)
    text = text.replace(m, file(basename+'.tex').read())
    file(fname,'w').write(text)