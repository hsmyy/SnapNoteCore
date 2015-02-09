REMOTE_DIR=/home/ubuntu/ocr-test-env/
PEM_PATH=/home/xxy/.ssh/android.pem

cd Debug
make

scp -r -i $PEM_PATH image_process ubuntu@192.168.64.31:$REMOTE_DIR/bin

cd ../ansible
deploy.sh deploy.yml
