%SYS_PYTHON% setup.py build --compiler=mingw32 
%SYS_PYTHON% setup.py install --prefix=%PREFIX%
if errorlevel 1 exit 1


for %%x in (libgcc_s_sjlj-1.dll libgfortran-3.dll libquadmath-0.dll) do (
   copy %SYS_PREFIX%\Scripts\%%x %SP_DIR%\pymc\
   if errorlevel 1 exit 1
)
