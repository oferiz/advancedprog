WILL WORK ONLY ON A WINDOWS MACHINE

Usage example (text for file "input.txt"):
1
30 0.8 0.2
15 0.7 0.2 0.1

The first line represents how many nodes that have 7 subspaces to start with (can be only 0, 1 or 2).
The following lines relate to spaces that have 49 subspaces. The first of these lines says that for the first 30 nodes picked to the independent set, each node has 80% chance to be picked from the nodes with the lowest degree in the graph and 20% chance to be picked from the nodes with the second lowest degree in the graph (from the nodes that can be added to the set). The last line says that for the next 15 nodes picked to the independent set, each node has 70% chance to be picked from the nodes with the lowest degree in the graph, 20% chance to be picked from the nodes with the second lowest degree in the graph and 10% chance to be picked from the nodes with the third lowest degree in the graph. You may input any amount of such lines and in each line you may print any amount of propabilities (that will add-up to 100%).
If there are more spaces that have 49 subspaces left that can be added to the set, the program will pick them by degree order.


