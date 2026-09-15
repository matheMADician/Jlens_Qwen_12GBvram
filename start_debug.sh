# debug時要先執行這個檔案
# 用完記得 exit，不然會一直扣錢

#!/bin/bash
salloc \
	--account=MST113234 \
	--partition=dev \
	--nodes=1 \
	--ntasks=1 \
	--cpus-per-task=8 \
	--gres=gpu:1