conda activate SC4001
for f in configs/*.yaml; do
    python3 train.py --config $f
done