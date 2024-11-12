python train.py --config_file config-mscoco-cnnrnn.yaml --embed_size 512 --batch_size 64 --learning_rate 0.001
python train.py --config_file config-mscoco-cnnattn.yaml --embed_size 512 --num_layers 2 --batch_size 64 --learning_rate 0.001
python train.py --config_file config-mscoco-vitcnnattn.yaml --embed_size 512 --num_layers 2 --batch_size 64 --learning_rate 0.001