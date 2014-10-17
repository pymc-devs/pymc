# Prepare a vagrant CentOS 6.5 VM for building OpenMM
# Needs latest version of vagrant to auto-download the chef package
#vagrant init chef/centos-6.5
#vagrant up
#vagrant ssh


# Download and enable the EPEL RedHat EL extras repository
mkdir ~/Software
cd Software
sudo yum install wget -y
wget http://dl.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm
sudo rpm -i epel-release-6-8.noarch.rpm

sudo yum update -y

# Several of these come from the EPEL repo
sudo yum install cmake28 graphviz perl flex bison rpm-build texlive texlive-latex ghostscript gcc gcc-c++ git vim -y gcc-gfortran

# Probably can't use RHEL6 version of doxygen because it's very old.
wget http://ftp.stack.nl/pub/users/dimitri/doxygen-1.8.7.src.tar.gz
rpmbuild -ta doxygen-1.8.7.src.tar.gz
sudo rpm -i ~/rpmbuild/RPMS/x86_64/doxygen-1.8.7-1.x86_64.rpm
rm ~/rpmbuild -r


sudo yum clean headers
sudo yum clean packages


# Install Conda
cd ~/Software
wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh
bash Miniconda-3.7.0-Linux-x86_64.sh -b

export PATH=$HOME/miniconda/bin:$PATH
conda install --yes jinja2 sphinx conda-build binstar
