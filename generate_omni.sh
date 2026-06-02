python scripts/data/generate_omni_waypoint_json.py \
  --video-folder "/data0/lujia/VLNData/InternData-N1/vln_n1/traj_data,/data1/xiangchen/traj_data/R2R,/data1/xiangchen/traj_data/RxR" \
  --output /data1/xiangchen/traj_data/omni_waypoint_train.json \
  --max-samples 20000000 \
  --num-history-images 20 \
  --waypoint-number 5 \
  --trajectory-stride 3 \
  --step-scale 0.3