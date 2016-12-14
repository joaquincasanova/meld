#!/bin/bash
#sudo apt-get update
#sudo apt-get upgrade

#need most of these for the meld code to work.
#system dependent. tough shit
#sudo apt install emacs
#sudo apt install python-dev
#sudo apt install python-pip
#sudo apt install git
#sudo apt install python-opencv
#sudo apt install texlive-latex-base
#sudo apt install python-tk					
#sudo apt install qt-sdk
#sudo apt install qtcreator
#sudo apt install python-traits
#sudo apt install python-traitsui

# prerequisites
#sudo apt-get install build-essential tcsh libtool-bin libtool automake gfortran libglu1-mesa-dev libfreetype6-dev uuid-dev libxmu-dev libxmu-headers libxi-dev  libx11-dev libxt-dev libjpeg62-dev libxaw7-dev liblapack-dev
#sudo apt-get install gcc-4.8 g++-4.8 libgfortran-4.8-dev
#sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
#sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

#sudo pip install --upgrade numpy
#sudo pip install --upgrade scipy
#sudo pip install --upgrade matplotlib
#sudo pip install --upgrade scikit-learn
#sudo pip install --upgrade pandas
#sudo pip install --upgrade mayavi
#sudo pip install --upgrade PySurfer
#sudo pip install --upgrade mne
#sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc2-cp27-none-linux_x86_64.whl
cd ~/Downloads
wget http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt-x11-gpl-4.11.4.tar.gz
tar -xvzf PyQt*
mv PyQt-x11-gpl-4.11.4 ~/

wget https://sourceforge.net/projects/pyqt/files/sip/sip-4.18.1/sip-4.18.1.tar.gz
tar -xvzf sip*
mv sip-4.18.1 ~/

wget ftp://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/dev/freesurfer-Linux-centos6_x86_64-dev.tar.gz
tar -xvzf freesurfer*
mv freesurfer ~/

wget ftp://surfer.nmr.mgh.harvard.edu/pub/dist/fs_supportlibs/prebuilt/centos6_x86_64/centos6-x86_64-packages.tar.gz
tar -xzvf centos6-x86_64-packages.tar.gz
mv centos6-x86_64-packages ~/

#need to download after registering, can't wget:
tar -xvzf MNE-*.tar.gz
mv ~/MNE-2.7.0-3106-Linux-x86_64 ~

cd ~/centos6-x86_64-packages/
./setup.sh
c
d ~/sip-4.18.1
sudo python configure.py
sudo make
sudo make install

cd ~/PyQt-x11-gpl-4.11.4
sudo python configure.py

cd ~/freesurfer

cd ~/MNE-2.7.0-3106-Linux-x86_64
MNE_ROOT=/home/jcasa/MNE-2.7.0-3106-Linux-x86_64
source $MNE_ROOT/bin/mne_setup_sh

cd ~

echo "QT_API=pyqt" >> ~/.bashrc
echo "ETS_TOOLKIT=qt4" >> ~/.bashrc
echo "MNE_ROOT=/home/jcasa/MNE-2.7.0-3106-Linux-x86_64" >> ~/.bashrc
echo 'source $MNE_ROOT/bin/mne_setup_sh' >> ~/.bashrc
echo "FREESURFER_HOME=/home/jcasa/freesurfer" >> ~/.bashrc
echo 'source $FREESURFER_HOME/SetUpFreeSurfer.sh' >> ~/.bashrc
echo 'SUBJECTS_DIR=/home/jcasa/mne_data/openfmri/subjects' >> ~/.bashrc

source ~/.bashrc
