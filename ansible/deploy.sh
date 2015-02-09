#!/bin/bash
if [ -z $1 ];then
    echo 'usage deploy [playbook.yml]'
else
    ansible-playbook -i hosts -u ubuntu $1
fi
