#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=cannon-orig.%j
#SBATCH --error=cannon-orig-err.%j
#SBATCH --time=00:04:00
#SBATCH --partition=develgpus

#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/no_comp/result_no_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/no_comp/result_no_comp_2048.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/no_comp/result_no_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/no_comp/result_no_comp_8192.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/no_comp/iresult_no_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/comp/result_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/comp/result_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/comp/result_comp_2048.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/comp/result_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_brain/comp/result_comp_8192.txt

#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/no_comp/result_no_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/no_comp/result_no_comp_2048.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/no_comp/result_no_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/no_comp/result_no_comp_8192.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/no_comp/result_no_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/comp/result_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/comp/result_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/comp/result_comp_2048.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/comp/result_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/msg_sppm/comp/result_comp_8192.txt

# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/no_comp/result_no_comp_1024.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/no_comp/result_no_comp_2048.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/no_comp/result_no_comp_4096.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/no_comp/result_no_comp_8192.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/no_comp/result_no_comp_16384.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/comp/result_comp_16384.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/comp/result_comp_1024.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/comp/result_comp_2048.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/comp/result_comp_4096.txt
# srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/num_plasma/comp/result_comp_8192.txt

#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/no_comp/result_no_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/no_comp/result_no_comp_2048.txt
srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/no_comp/result_no_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/no_comp/result_no_comp_8192.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/no_comp/result_no_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 16384 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/comp/result_comp_16384.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 1024 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/comp/result_comp_1024.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 2048 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/comp/result_comp_2048.txt
srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 4096 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/comp/result_comp_4096.txt
#srun /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/build/Cannons_Algorithm_Comp 8192 /p/project/icei-hbp-2022-0013/vogel6/cmake-tut/test/ergebnisse/obs_spitzer/comp/result_comp_8192.txt
