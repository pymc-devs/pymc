import os

dir_to_change = '/Users/anand/renearch/PyMC'
    
name_replacements = [
('Parameter', 'Stochastic'),
('parameter', 'stoch'),
('param', 'stoch'),
('Node', 'Deterministic'), 
('node', 'dtrm'),
('PyMCBase', 'Node'),
('pymc_object', 'node'),
('pymc object', 'node'),
('SamplingMethod', 'StepMethod'),
('sampling_method', 'step_method'),
('sampling method', 'step method')]

    
for dirname, dirs, fnames in os.walk(dir_to_change):
    
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