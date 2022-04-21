# OpenMPI needs an extra argument to run as root and needs to be able
# to oversubscribe CPUs to pass the MPI smoke test
set(BLT_MPI_COMMAND_APPEND "--allow-run-as-root --oversubscribe" CACHE PATH "") 
# No CUDA on github-hosted runners
set(ENABLE_CUDA "OFF" CACHE PATH "")
