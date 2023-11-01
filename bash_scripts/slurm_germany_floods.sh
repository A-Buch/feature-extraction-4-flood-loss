#!/bin/bash
### process feature extraction for households in Germany

#SBATCH --nodes=1
#SBATCH --constraint="cascadelake"
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --output=/storage/vast-gfz-hpc-01/home/abuch/Feature_Selection_Pipeline/feature-selection-pipeline/log/%x_%A.log
#SBATCH --error=/storage/vast-gfz-hpc-01/home/abuch/Feature_Selection_Pipeline/feature-selection-pipeline/log/%x_%A.log
#SBATCH --time=00-04:00:00


aoi_and_floodtype=$1
year=$2


source ./variables_shellscript.sh

module load $python_version
source $venv_dir

cd $project_basedir/scripts
srun python -u feature_selection_regression_germany_commercial_floods.py ${aoi_and_floodtype} ${year}

echo "Finished run"

deactivate

