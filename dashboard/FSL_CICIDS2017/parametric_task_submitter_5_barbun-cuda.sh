#!/bin/bash
#SBATCH -p barbun-cuda        	# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ituzun         	    # Kullanici adi
#SBATCH -J agent5_torch_experiment     # Gonderilen isin ismi
#SBATCH --dependency singleton
#SBATCH -o experiment_5.out    	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        	# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                	# Gorev kac node'da calisacak?
#SBATCH -n 1                	# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20  	# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=1-23:00:00      	# Sure siniri koyun.
#SBATCH --mail-user=e2036234@ceng.metu.edu.tr
#SBATCH --mail-type=ALL

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate torch-env
conda init

cd /truba/home/$USER/agent5/FSL_IDS/dashboard/FSL_CICIDS2017/
./run_experiments.sh $1 $2 $3 > run_experiments.log 2>&1
