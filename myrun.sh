folder_path=./data/door_pre4055/

if [ ! -d folder_path"logs" ]; then
    mkdir -p $folder_path'logs'
fi



# # # LSTM BiLSTM CNN+LSTM
# for model_name in RNN

# do
# python -u run.py \
#   --choose_model $model_name \
#   --folder_path $folder_path  >$folder_path'logs/'$model_name'.log'
# done


# test
# LSTM BiLSTM CNN+LSTM CNN
for model_name in RNN

do
python -u test.py \
  --choose_model $model_name \
  --folder_path $folder_path  >$folder_path'logs/'$model_name'.log'
done