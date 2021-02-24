# Activate Conda environment
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate anm
echo "Running in Conda env:" 
which python

# Run for 2 different random seeds
for seed in {5..5}
do
   python train.py $1 -s $seed &
done

