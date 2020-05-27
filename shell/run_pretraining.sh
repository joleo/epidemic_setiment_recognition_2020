python ../Further_pretraining/run_pretraining.py \
--input_file=../data/tf_pretrain_data.tfrecord \
--output_dir=../data/pretrain_weight/pretraining_output \
--do_train=True \
--do_eval=True \
--bert_config_file=/data/data01/liyang099/com/weight/chinese/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/data/data01/liyang099/com/weight/chinese/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt \
--train_batch_size=32 \
--eval_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=50 \
--num_train_steps=100000 \
--num_warmup_steps=10000 \
--learning_rate=5e-5

