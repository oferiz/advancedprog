// AdvancedProg.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <ctime> 
#include <vector> 
#include <list>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;

/*
 * In order to generate all of the vector spaces I define a canonical
 * representation of a vector space.
 *
 * A CANONICAL REPRESENTATION of a vector space of bit vectors of dimension 3 is
 * an array of the 7 non-zero vectors in the space: {vec1, ..., vec7} where
 * vec1, ..., vec3 is a basis and vec1 is the smallest vector of the space 
 * lexicographicaly, vec2 and the smallest after vec1, and vec3 is the smallest
 * after vec1, vec2, vec1 + vec2. Also:
 * vec4 = vec1 + vec2, vec5 = vec2 + vec3, vec6 = vec1 + vec2,
 * vec7 = vec1 + vec2 + vec3.
 *
 * A CANONICAL REPRESENTATION of a vector space of bit vectors of dimension 2 is
 * simply the 3 non-zero vectors of the space arranged in lexicographical order
 */


/* 
 * simple implementation of a BINARY TREE in order to store vector spaces for
 * quick search of spaces
 */

typedef struct BinaryNode {
  struct BinaryNode* left;
  struct BinaryNode* right;
} BinaryNode;

BinaryNode* createTree(){
	BinaryNode* tree = new BinaryNode;
	tree->left = NULL;
	tree->right = NULL;
	return tree;
}

void freeTree(BinaryNode* tree){
	if(tree == NULL)
		return;
	freeTree(tree->left);
	freeTree(tree->right);
	delete tree;
}

void addToTree(BinaryNode* tree, bool** space){
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 7; j++){
			if(space[i][j] == 0){
				if(tree->left == NULL)
					tree->left = createTree();
				tree = tree->left;
			}
			else { //space[i][j] == 1
				if(tree->right == NULL)
					tree->right = createTree();
				tree = tree->right;
			}
		}
}

bool searchInTree(BinaryNode* tree, bool** space){
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 7; j++){
			if(space[i][j] == 0){
				if(tree->left == NULL)
					return false;
				tree = tree->left;
			}
			else { //space[i][j] == 1
				if(tree->right == NULL)
					return false;
				tree = tree->right;
			}
		}
	return true;
}

//how many spaces held in this tree (how many leafs)
int sizeOfTree(BinaryNode* tree){
	if(tree == NULL)
		return 0;
	if(tree->left == NULL && tree->right == NULL)
		return 1;
	return sizeOfTree(tree->left) + sizeOfTree(tree->right);
}

//checks if the two trees contain mutual vector space
bool doTreesIntersect(BinaryNode* tree1, BinaryNode* tree2){
	//simply check if trees have a mutual route from root to a leaf
	if(tree1 == NULL || tree2 == NULL)
		return false;
	if(tree1->left == NULL && tree1->right == NULL &&
		tree2->left == NULL && tree2->right == NULL)//both leafs
		return true;
	return doTreesIntersect(tree1->left, tree2->left) ||
		doTreesIntersect(tree1->right, tree2->right);
}

/* 
 *implementation of a graph specialized for this project, to evantually
 * find maximal independent sets in the graph
 */

typedef struct GraphNode {
	list<bool**> node;//the list of spaces of this node (which are all
						//shifts of some space)
	BinaryNode* subspacesTree;//the tree of all the subspaces of this
								//node
	int degree;//number of neighbours
	list<struct GraphNode*> neighbours;
	bool dead;//relevant for maximal independent set algorithm
} GraphNode;

typedef struct Graph {
	vector<GraphNode*> nodes;
} Graph;

Graph* createGraph(){
	Graph* graph = new Graph;
	graph->nodes = vector<GraphNode*>();
	return graph;
}

GraphNode* createNode(list<bool**> node, BinaryNode* subspacesTree){
	GraphNode* graph_node = new GraphNode;
	graph_node->node = node;
	graph_node->subspacesTree = subspacesTree;
	graph_node->degree = 0;
	graph_node->dead = false;
	return graph_node;
}

void addNode(Graph* graph, GraphNode* node){
	graph->nodes.push_back(node);
}

void addEdge(GraphNode* node1, GraphNode* node2){
	node1->degree++;
	node1->neighbours.push_back(node2);
	node2->degree++;
	node2->neighbours.push_back(node1);
}

//comparator of nodes (by degree) for the sort function below
struct compareNodes
{
    bool operator() (const GraphNode* node1, const GraphNode* node2)
    {
        return (node1->degree < node2->degree);
    }
};

void sortNodesByDegree(Graph* graph){
	sort(graph->nodes.begin(), graph->nodes.end(), compareNodes());
}


/* AUXILIARY VECTORS COMPARISONS */

/*
 * returns true if vec1 is lexicographically smaller than or equal to vec2
 */
bool smallerThan(bool* vec1, bool* vec2){
	for(int i = 0; i < 7; i++){
		if(vec1[i] == 0 &&  vec2[i] == 1)
			return true;
		if(vec1[i] == 1 &&  vec2[i] == 0)
			return false;
	}
	return true;
}

/*
 * returns true if vec1 and vec2 are equal
 */
bool areEqualVectors(bool* vec1, bool* vec2){
	for(int i = 0; i < 7; i++)
		if(vec2[i] != vec1[i])
			return false;
	return true;
}

/*
 * returns true if vec1 + vec2 == vec3
 */
bool addEquals(bool* vec1, bool* vec2, bool* vec3){
	for(int i = 0; i < 7; i++)
		if((vec1[i] ^ vec2[i]) != vec3[i])
			return false;
	return true;
}

/*
 * returns true vec is the zero vector
 */
bool isZeroVector(bool* vec){
	for(int i = 0; i < 7; i++)
		if(vec[i] != 0)
			return false;
	return true;
}


/* AUXILIARY SPACE OPERATIONS */

//for dimension 3 space
bool** allocateSpace3(){
	bool** space = new bool*[7];
	for(int i = 0; i < 7; i++)
		space[i] = new bool[7];
	return space;
}

//for dimension 2 space
bool** allocateSpace2(){
	bool** space = new bool*[7];
	for(int i = 0; i < 3; i++)
		space[i] = new bool[7];
	return space;
}

void freeSpace3(bool** space){
	for(int i = 0; i < 7; i++)
		delete[] space[i];
	delete[] space;
}

void freeSpace2(bool** space){
	for(int i = 0; i < 3; i++)
		delete[] space[i];
	delete[] space;
}

/*
 * copies the given space to a newly allocated dynamic space
 */
//for dimension 3 space
bool** replicateSpace3(bool** space){
	bool** copy_space = allocateSpace3();
	for(int i = 0; i < 7; i++)
		for(int j = 0; j < 7; j++)
			copy_space[i][j] = space[i][j];
	return copy_space;
}

//for dimension 2 space
bool** replicateSpace2(bool** space){
	bool** copy_space = allocateSpace2();
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 7; j++)
			copy_space[i][j] = space[i][j];
	return copy_space;
}

/*
 * shift each vector in the space 1 bit to the right circularly to receive
 * a new vector space (for dimension 3 space)
 */
bool** shiftSpace(bool** space){
	bool** shifted_space = replicateSpace3(space);
	for(int i = 0; i < 7; i++)
		for(int j = 0; j < 7; j++)
			shifted_space[i][j] = space[i][(j - 1 + 7) % 7];
	return shifted_space;
}

//for dimension 3 space
void printSpace3(bool** space, ostream& stream){
	for(int i = 0; i < 7; i++){
		stream << "vec" << i+1 << ": ";
		for(int j = 0; j < 7; j++)
			stream << space[i][j];
		stream << endl;
	}
	stream << endl;
}

//for dimension 2 space
void printSpace2(bool** space, ostream stream){
	stream << endl;
	for(int i = 0; i < 3; i++){
		stream << "vec" << i+1 << ": ";
		for(int j = 0; j < 7; j++)
			stream << space[i][j];
		stream << endl;
	}
}


/*
 * expects an array which contains a basis in the first enteries,
 * and calculates the rest of the vectors from the basis
 */
//for dimension 3 space
void completeToSpace3(bool** space){
	for(int i = 0; i < 7; i++){
		space[3][i] = space[0][i] ^ space[1][i];
		space[4][i] = space[1][i] ^ space[2][i];
		space[5][i] = space[0][i] ^ space[2][i];
		space[6][i] = space[0][i] ^ space[1][i] ^ space[2][i];
	}
}

//for dimension 2 space
void completeToSpace2(bool** space){
	for(int i = 0; i < 7; i++)
		space[2][i] = space[0][i] ^ space[1][i];
}

/*
 * receives a space as an array of its vectors and rearranges the
 * vectors to receive the canonical form described abovFe
 */
//for dimension 3 space
void toCanonical3(bool** space){
	//minimal vector first

	//find minimal vector
	int min_ind = 0;
	for(int i = 1; i < 7; i++)
		if(smallerThan(space[i], space[min_ind]))
			min_ind = i;
	//move minimal vector to be first
	for(int i = 0; i < 7; i++){
		bool tmp = space[0][i];
		space[0][i] = space[min_ind][i];
		space[min_ind][i] = tmp;
	}

	//second minimal vector is second
	
	//find minimal vector starting from index 1
	min_ind = 1;
	for(int i = 2; i < 7; i++)
		if(smallerThan(space[i], space[min_ind]))
			min_ind = i;
	//move second minimal vector to be second
	for(int i = 0; i < 7; i++){
		bool tmp = space[1][i];
		space[1][i] = space[min_ind][i];
		space[min_ind][i] = tmp;
	}

	//third minimal vector which is not the sum of the first two is third

	//find minimal vector which is not the sum of the first two starting
	//from index 2
	min_ind = 2;
	if(addEquals(space[0], space[1], space[2]))
		min_ind = 3;
	for(int i = 3; i < 7; i++)
		if(smallerThan(space[i], space[min_ind]) &&
			!addEquals(space[0], space[1], space[i]))
			min_ind = i;
	//move second minimal vector to be second
	for(int i = 0; i < 7; i++){
		bool tmp = space[2][i];
		space[2][i] = space[min_ind][i];
		space[min_ind][i] = tmp;
	}

	//after we have the basis we can complete to full canonical space
	completeToSpace3(space);
}

//for dimension 2 space
void toCanonical2(bool** space){
	//minimal vector first

	//find minimal vector
	int min_ind = 0;
	for(int i = 1; i < 3; i++)
		if(smallerThan(space[i], space[min_ind]))
			min_ind = i;
	//move minimal vector to be first
	for(int i = 0; i < 7; i++){
		bool tmp = space[0][i];
		space[0][i] = space[min_ind][i];
		space[min_ind][i] = tmp;
	}

	//second minimal vector is second, and third minimal is third
	
	if(smallerThan(space[2], space[1]))
		//switch vectors
		for(int i = 0; i < 7; i++){
			bool tmp = space[1][i];
			space[1][i] = space[2][i];
			space[2][i] = tmp;
		}
}

/*
 * returns true if the two given spaces have a joint subspace of
 * dimension 2
 */
bool hasJointSubspace(bool** space1, bool** space2){
	int count = 0;//counts how many joint vectors the spaces share
	for(int i = 0; i < 7; i++)
		for(int j = 0; j < 7; j++)
			if(areEqualVectors(space1[i], space2[j]))
				count++;

	if(count >= 2)
		return true;
	return false;
}

/*
 * check if the given array represents a valid vector space of dimension 3
 * IN ThE CANONICAL FORM DESCRIBED ABOVE
 */
bool checkValid(bool** space){
	if(isZeroVector(space[0]) || !smallerThan(space[0], space[1]) ||
			!smallerThan(space[1], space[2]) ||
			addEquals(space[0], space[1], space[2])||
			areEqualVectors(space[0], space[1]) ||
			areEqualVectors(space[1], space[2]))
		return false;

	for(int i = 3; i < 7; i++){
		if(!smallerThan(space[0], space[i]))
			return false;
	}

	for(int i = 3; i < 7; i++){
		if(!smallerThan(space[1], space[i]))
			return false;
	}

	//start from i = 4 to skip the vector cur[0] + cur[1]
	for(int i = 4; i < 7; i++){
		if(!smallerThan(space[2], space[i]))
			return false;
	}

	return true;
}

/*
 * the recursive function for the wrapper below
 * @param cur the space being generated currently
 * @param vec_ind the index in the main array (points on the vector). this
 * goes upto  because only a bsis is generated
 * @param bit_ind the bit index in the vector (upto 7)
 */
void createSpacesRec(list<bool**>& lst, bool** cur, int vec_ind, int bit_ind){
	
	if(bit_ind == 7){//finished generating a whole vector
		if(vec_ind == 2){//finished generating 3 vector which is the whole basis
			completeToSpace3(cur);
			if(!checkValid(cur))//if not a valid canonical space
				return;

			//found a valid canonical space. add to list
			bool** space = replicateSpace3(cur);
			lst.push_back(space);
			return;
		}

		vec_ind ++;
		bit_ind = 0;
	}

	//set current bit to 0 or 1 and then make the recursive call for the next bit
	cur[vec_ind][bit_ind] = 0;
	createSpacesRec(lst, cur, vec_ind, bit_ind + 1);

	cur[vec_ind][bit_ind] = 1;
	createSpacesRec(lst, cur, vec_ind, bit_ind + 1);
}

/*
 * creates all of the vector spaces of dimension 3 without repetitions and stores them
 * in the given list
 */
void createSpaces(list<bool**>& lst){//recursion wrapper
	bool** cur_space = allocateSpace3();//help variable for the recursion function
	createSpacesRec(lst, cur_space, 0, 0);
	freeSpace3(cur_space);
}

/*
 * receives a space of dimension 3 and returns a list with all of the 7 subspaces
 * of dimension 2 of this space
 */
list<bool**> getAllSubspaces(bool** space){
	list<bool**> lst;

	//first subspace
	bool** subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[0][i];
		subspace[1][i] = space[1][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//second subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[1][i];
		subspace[1][i] = space[2][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//third subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[0][i];
		subspace[1][i] = space[2][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//fourth subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[0][i];
		subspace[1][i] = space[4][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//fifth subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[1][i];
		subspace[1][i] = space[5][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//sixth subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[2][i];
		subspace[1][i] = space[3][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	//seventh subspace
	subspace = allocateSpace2();
	for(int i = 0; i < 7; i++){
		subspace[0][i] = space[3][i];
		subspace[1][i] = space[5][i];
	}
	completeToSpace2(subspace);
	toCanonical2(subspace);
	lst.push_back(subspace);

	return lst;
}

/*
 * checks if the two spaces of dimension 3 are equal. this function assumes
 * canonical representation of the spaces
 */
bool areEqualSpaces(bool** space1, bool** space2){
	//sufficient to check equality of the three basis vectors
	return areEqualVectors(space1[0], space2[0]) &&
		areEqualVectors(space1[1], space2[1]) &&
		areEqualVectors(space1[2], space2[2]);
}


/* Main stages of the program */

Graph* Preprocessing(){
	cout<<"Preprocessing initiated..." << endl << endl;
	//create the list of all spaces of dimension 3
	list<bool**> lst;
	createSpaces(lst);
	//cout<<"	Found all spaces of dimension 3. Amount: " << lst.size() <<
	//		endl << endl;

	//create a list of lists, where each list represents a node which is
	//a cycle. a cycle is all of the spaces which are shifts of some space
	list<list<bool**>> nodes;
	{
		//also store all spaces so far in a binary tree to quickly search
		//if some space had already been added to the list of lists
		BinaryNode* tree = createTree();

		list<bool**>::const_iterator iter_lst;
		for (iter_lst = lst.begin(); iter_lst != lst.end(); ++iter_lst) {
			bool** space = *iter_lst;
			if(searchInTree(tree, space))//space already added before
				continue;

			//else add this spaces with all of its shifts as a new list
			list<bool**> node;
			bool** cur_shifted = replicateSpace3(space);
			node.push_back(cur_shifted);
			addToTree(tree, cur_shifted);
			for(int i = 1; i < 7; i++){

				cur_shifted = shiftSpace(cur_shifted);
				toCanonical3(cur_shifted);
				if(!searchInTree(tree, cur_shifted)){
					node.push_back(cur_shifted);
					addToTree(tree, cur_shifted);
				}
			}

			nodes.push_back(node);
		}
	}

	//for each list in variable "nodes" we hold a binary tree which will
	//contain all of the subspaces of dimension 2 of the spaces in that node
	list<BinaryNode*> nodes_subspaces;
	{
		list<list<bool**>>::const_iterator iter_nodes;
		for (iter_nodes = nodes.begin(); iter_nodes != nodes.end(); ++iter_nodes){
			list<bool**> node = *iter_nodes;
			BinaryNode* tree = createTree();

			//iterate through all spaces in this node and add all of their subspaces
			//to the tree
			list<bool**>::const_iterator iter_spaces;
			for (iter_spaces = node.begin(); iter_spaces != node.end(); ++iter_spaces){
				bool** space = *iter_spaces;
				list<bool**> subspaces = getAllSubspaces(space);

				//add all subspaces to tree
				list<bool**>::const_iterator iter_subspaces;
				for (iter_subspaces = subspaces.begin(); iter_subspaces != subspaces.end(); ++iter_subspaces){
					bool** subspace = *iter_subspaces;
					addToTree(tree, subspace);
				}
			}

			nodes_subspaces.push_back(tree);
		}

		//cout<<"	Done calculating all merged nodes. Amount: " <<
				//nodes.size() << endl;
	}

	//Build the graph
	//cout << endl << "	Building the graph..." << endl << endl;
	Graph* graph = createGraph();
	{
		//build nodes:
		list<BinaryNode*>::const_iterator iter_trees = nodes_subspaces.begin();
		list<list<bool**>>::const_iterator iter_nodes = nodes.begin();
		while(iter_trees != nodes_subspaces.end()){
			GraphNode* graph_node = createNode(*iter_nodes, *iter_trees);
			addNode(graph, graph_node);

			++iter_trees;
			++iter_nodes;
		}

		//build vertices:
		vector<GraphNode*> graph_nodes = graph->nodes;
		for(int i = 0; i < (int)(graph_nodes.size()); i++){
			for(int j = i + 1; j < (int)(graph_nodes.size()); j++){
				GraphNode* node1 = graph_nodes[i];
				GraphNode* node2 = graph_nodes[j];
				//add edge if nodes share a subspace
				if(doTreesIntersect(node1->subspacesTree, node2->subspacesTree)){
					addEdge(node1, node2);
				}
			}
		}

		//sort nodes by degree
		sortNodesByDegree(graph);
	}

	cout << "Preprocessing Done." << endl << endl;

	return graph;
}

void Statistics1(Graph* graph){
	//print statistics:: how many subspaces of dimension 2 do the
	//nodes have? for each possible amount of subspaces (7 to 49) print
	//how many nodes have that many subspaces
	for(int i = 7; i <= 49; i++){
		int count = 0;//counts nodes with i subspaces
		bool** example;//will hold an example vector space which
					//belongs to a node with i subspaces

		for(int j = 0; j < (int)(graph->nodes.size()); j++){
			BinaryNode* subspacesTree = graph->nodes[j]->subspacesTree;
			list<bool**> nodeSpaces = graph->nodes[j]->node;
			if(sizeOfTree(subspacesTree) == i){
				if(count == 0)
					example = (nodeSpaces).front();
				count++;
			}
		}

		if(count > 0){
			cout << endl << "There are " << count << " nodes with " <<
				i << " subspaces. Example space:";
			printSpace3(example, cout);
		}
	}
}

void Statistics2(Graph* graph, int n){
	//print statistics: space from each node which n subspaces
	int count = 0;
	vector<GraphNode*> graph_nodes = graph->nodes;
	for(int i = 0; i < (int)(graph_nodes.size()); i++)
		if(sizeOfTree(graph_nodes[i]->subspacesTree) == n)
			count++;

	ofstream file;
	file.open ("output.txt");

	file << "Amount: " << count << endl;
		
	for(int i = 0; i < (int)(graph_nodes.size()); i++)
		if(sizeOfTree(graph_nodes[i]->subspacesTree) == n){
			printSpace3(graph_nodes[i]->node.front(), file);
			if(sizeOfTree(graph_nodes[i]->subspacesTree) != 7)
					file << "(Together with all of its 7 shifts)" << endl;
			file << "Degree of the node in the graph: " << graph_nodes[i]->neighbours.size() << endl;
		}

	file.close();
}

void Execution(Graph* graph){
	cout << "Execution initiated..." << endl << endl;

	//clear file opnened in previous execution
	system("if exist set_size*.txt del set_size*.txt");

	//read input
	ifstream myfile ("input.txt");
	if (!myfile.is_open()){
		cout << "ERROR: Unable to open file \"input.txt\"";
		return;
	}

	int amount7; //how many nodes that has 7 subspaces to take (0, 1 or 2)
	vector<int> nodes_amounts;//each int represents the amount of nodes in a batch
	vector<vector<double>> probabilities;//each vector of int represents the probabilites
										//of that batch (read Readme.txt)

	//fill the above variables from the input file
	string line;
	istringstream buffer;

	getline (myfile,line);
	buffer = istringstream(line);
	buffer >> amount7;

	if(amount7 < 0 || amount7 > 2){
		cerr << "ERROR: First input line must be either 0, 1 or 2" << endl;
		return;
	}

	while(!myfile.eof()){
		getline (myfile,line);
		buffer = istringstream(line);
		//read amount int
		int amount;
		buffer >> amount;
		nodes_amounts.push_back(amount);
		//read prababilities line
		vector<double> ith_probabilities;
		while(!buffer.eof()){
			double probability;
			buffer >> probability;
			ith_probabilities.push_back(probability);
		}
		probabilities.push_back(ith_probabilities);
	}

	//aggregate probabilites for easier use of random function later. example:
	//(0.3, 0.1, 0.6) will be transformed to (0.3, 0.4, 1)
	for(int i = 0; i < (int)(probabilities.size()); i++)
		for(int j = 1; j < (int)(probabilities[i].size()); j++){
			probabilities[i][j] = probabilities[i][j-1] + probabilities[i][j];
			if(j == (int)(probabilities[i].size()) - 1)
				probabilities[i][j] = 1.0;
		}

	myfile.close();

	vector<GraphNode*> graph_nodes = graph->nodes;
	int max = 0;//size of maximal independent set found so far
	while(true){
		//first, in the sorted array of nodes, there are clusters of "sub-arrays"
		//of nodes that share the same degree. rearrange these nodes with themselves
		//in a random way via Knuth's shuffle algorithm

		//for each such "cluster" also filll its ends in these data variables for the
		//upcoming execution
		vector<int> clusters_beginnings;
		vector<int> clusters_ends;
		vector<int> clusters_inds;//for upcoming execution

		int start = 0; int end = 0;
		while(end < (int)(graph_nodes.size())){
			//determine the two "sub-array"'s ends (start+end)
			while(end < (int)(graph_nodes.size()) - 1 &&
					graph_nodes[end+1]->degree == graph_nodes[start]->degree)
				end++;

			//now that we have the ends, insert them to the data variables
			//(only relevant to spaces with 49 subspaces which start from index 69)
			if(start >= 69){
				clusters_beginnings.push_back(start);
				clusters_ends.push_back(end);
				clusters_inds.push_back(start);
			}

			//now perform the Knuth's shuffle in the array between indices "start" to "end"
			for(int i = start; i < end; i++){
				//choose random idex between i and end, and swap it with i
				int ind = i + (rand() % (int)(end - i + 1));
				GraphNode* tmp = graph_nodes[i];
				graph_nodes[i] = graph_nodes[ind];
				graph_nodes[ind] = tmp;
			}

			start = end + 1;
			end = end + 1;
		}

		//now search for a maximal independent set according to rules (read Readme.txt)
		list<GraphNode*> set;

		//add the nodes with 7 subspaces first
		for(int i = 0; i < amount7; i++){
			set.push_back(graph_nodes[i]);
			//kill of the node's neighbours
			list<GraphNode*>::const_iterator iter;
			for (iter = graph_nodes[i]->neighbours.begin();
					iter != graph_nodes[i]->neighbours.end(); ++iter)
				(*iter)->dead = true;
		}

		for(int i = 0; i < (int)(nodes_amounts.size()); i++){
			for(int j = 0; j < nodes_amounts[i]; j++){
				//make sure all clusters indices point to alive nodes
				for(int k = 0; k < (int)(clusters_beginnings.size()); k++)
					while(clusters_inds[k] < clusters_ends[k] && graph_nodes[clusters_inds[k]]->dead)
						clusters_inds[k]++;

				//determine randomly from where to take the next node (0 = lowest degree,
				// 1 = second lowest and so on
				int ind = 0;
				double rnd = (double)rand()/RAND_MAX;
				while(rnd > probabilities[i][ind])
					ind++;

				//find the index of the cluster from all the clusters that aren't exhusted
				int real_ind;
				int latest = -1;//latest found index of cluster which isn't exhusted
				for(real_ind = 0; real_ind  < (int)(clusters_beginnings.size()); real_ind ++){
					if(clusters_inds[real_ind] < clusters_ends[real_ind]){
						ind--;
						latest = real_ind;
					}
					if(ind < 0)
						break;
				}

				if(latest == -1)//all clusters exhusted
					goto set_ready;

				if(real_ind == (int)(clusters_beginnings.size()))
					real_ind = latest;

				//add to the set the next node from the chosen cluster
				int node_ind = clusters_inds[real_ind];
				set.push_back(graph_nodes[node_ind]);
				clusters_inds[real_ind]++;
				//kill of the node's neighbours
				list<GraphNode*>::const_iterator iter;
				for (iter = graph_nodes[node_ind]->neighbours.begin();
						iter != graph_nodes[node_ind]->neighbours.end(); ++iter)
					(*iter)->dead = true;
			}
		}

		//now try to add more nodes if possible by degree order
		for(int i = 0; i < (int)(clusters_beginnings.size()); i++){
			for(int j = clusters_inds[i]; j <= clusters_ends[i]; j++){
				if(graph_nodes[j]->dead)//has a vertex with a member of the set
					continue;
				//else add it to the set and kill all neighbours
				set.push_back(graph_nodes[j]);
				list<GraphNode*>::const_iterator iter;
				for (iter = graph_nodes[j]->neighbours.begin(); iter != graph_nodes[j]->neighbours.end(); ++iter)
					(*iter)->dead = true;
			}
		}

set_ready:

		if(max < (int)(set.size())){
			max = (int)(set.size());
			//write set to file
			ofstream file;
			file.open (string("set_size_") + to_string((long long)max) + string(".txt"));
		

			list<GraphNode*>::const_iterator iter;
			for (iter = set.begin(); iter != set.end(); ++iter){
				printSpace3((*iter)->node.front(), file);
				if(sizeOfTree((*iter)->subspacesTree) != 7)
					file << "(Together with all of its 7 shifts)" << endl;
				file << "Degree of the node in the graph: " << (*iter)->neighbours.size() << endl;
			}

			file.close();

			cout << "Largest independent set found so far is of size: " << set.size()
				<< " (written to file...)" << endl;
		}

		//clear set and revive all nodes
		set = list<GraphNode*>();
		for(int i = 0; i < (int)(graph_nodes.size()); i++)
			graph_nodes[i]->dead = false;
	}
	
	cout<<"Execution Done."<<endl;
}

int main()
{
	srand ((unsigned int)(time(NULL)));

	Graph* graph = Preprocessing();
	//Statistics1(graph);
	//Statistics2(graph, 49);
	Execution(graph);

	cin.get();
	return 0;
}
