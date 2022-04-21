# OpenMPI needs an extra argument to run as root
set(BLT_MPI_COMMAND_APPEND "--allow-run-as-root" CACHE PATH "") 
# No CUDA on github-hosted runners
set(ENABLE_CUDA "OFF" CACHE PATH "")
