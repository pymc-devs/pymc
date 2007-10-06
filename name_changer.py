import os
    
name_replacements = [
('Stochastic', 'Stochastic'),
('@stochastic', '@stochastic'),
('stoch', 'stoch'),
('stoch', 'stoch'),
('Functional', 'Functional'), 
('@functional', '@functional'),
('functl', 'functl'),
('Node', 'Functional'),
('nodes', 'functls'),
('nodes', 'functls'),
('StepMethod', 'StepMethod'),
('step_method', 'step_method'),
('step method', 'step method')]
    
for dirname, dirs, fnames in os.walk('/Users/anand/renearch/PyMC'):
    
    os.chdir(dirname)

    for fname in fnames:

        if fname[-3:]=='.py' or fname[-4:] in ['.tex', '.pyx', '.png', '.pdf'] and not fname=='name_changer.py':

            for pair in name_replacements:

                fname_new = fname.replace(pair[0], pair[1])
                os.rename( fname, fname_new )
                fname = fname_new

            f = file(fname, 'r')
            f_new = file(fname+'_copy','w')

            for line in f:

                new_line = line

                for pair in name_replacements:
                    new_line = new_line.replace(pair[0], pair[1])

                f_new.write( new_line )
            f.close
            f_new.close
            os.remove(fname)
            os.rename(fname+'_copy',fname)