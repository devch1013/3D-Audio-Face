# python agent.py
cd /home/elicer/talk2yourself/face_module/LDT/Results;zip -r download.zip objs
rclone copy /home/elicer/talk2yourself/face_module/LDT/Results/download.zip onedrive:rclone
rm /home/elicer/talk2yourself/face_module/LDT/Results/download.zip