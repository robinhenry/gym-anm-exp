# Activate Conda environment
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate anm
echo "Running in Conda env:" 
which python

# Run for random seeds
for seed in {1..$2}
do
   python train.py $1 -s $seed &
done
