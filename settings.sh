rm /etc/apt/sources.list
cp ./sources.list /etc/apt/sources.list
apt-get update
apt-get install sudo
sudo apt install ffmpeg -y
sudo apt-get install espeak-ng -y
# apt-get install python3-opencv -y
sudo apt-get install tmux -y
apt-get install git-lfs
sudo -v ; curl https://rclone.org/install.sh | sudo bash
~/miniconda3/bin/conda init bash


