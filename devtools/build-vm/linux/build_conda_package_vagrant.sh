export PATH=$HOME/miniconda/bin:$PATH

git clone https://github.com/pymc-devs/pymc.git
cd pymc
git checkout tags/2.3.4  # To checkout specific release for packaging. 
conda build conda
