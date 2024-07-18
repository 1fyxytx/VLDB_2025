# SIGMOD_2025_Round3

IndexC.cpp is the source code of Our method BatchPL. To run this program, you need to do the following steps:

1. Download the datasets from "http://konect.cc/networks/", "https://networkrepository.com/index.php", and "https://law.di.unimi.it/", which have the same format as test.graph.
2. Run the key.sh file.

The explanations of some parameters are listed as follows:

choice: 0 - build the 2-hop labeling by PSL in a static graph; 1 - BatchPL in edge deletion scenario; 3 - BatchPL in edge insertion scenario.
threads: the number of computing cores.
cp: the distribution of vertices to be insert/delete edges
Dcnt: the number of inserted/deleted edges
