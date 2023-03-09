#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"
#include "nbody_alloc.h"

extern node_t* root;

/* draw recursively the content of a node */
void draw_node(node_t* n) {
#ifndef DISPLAY
  return;
#else
  if(!n)
    return;

#if DRAW_BOXES
  int x1 = POS_TO_SCREEN(n->x_min);
  int y1 = POS_TO_SCREEN(n->y_min);
  int x2 = POS_TO_SCREEN(n->x_max);
  int y2 = POS_TO_SCREEN(n->y_max);
  draw_rect(x1, y1, x2, y2);
#endif

  if(n->particle) {
    int x = POS_TO_SCREEN(n->particle->x_pos);
    int y = POS_TO_SCREEN(n->particle->y_pos);
    draw_point (x,y);
  }
  if(n->children) {
#if 0
    /* draw a red point that represents the center of the node */
    int x = POS_TO_SCREEN(n->x_center);
    int y = POS_TO_SCREEN(n->y_center);
    draw_red_point (x,y);
#endif

    int i;
    for(i=0; i<4; i++) {
      draw_node(&n->children[i]);
    }
  }

#endif
}

 
 // CODE COPIED FROM https://www.geeksforgeeks.org/level-order-tree-traversal/
 // ALL CREDITS DUE FOR LEVEL ORDER TRAVERSAL ALGORITHM
 // MODIFIED HERE TO SUIT 4-NARY TRAVERSAL
 // Main motivation in using queue rather than recursive traversal: O(n^2) vs O(n)

int getNumNodes(node_t* root, int numParticles)
{
   int rear, front;
    node_t** queue = NULL;

    if(numParticles>=130)
    {
      queue = (struct node**)malloc(
    sizeof(struct node_t*) * numParticles* 4);
    }
    else
    {
      queue = (struct node**)malloc(
    sizeof(struct node_t*) * 1000);
    }
    
    front = rear = 0;
    node_t* temp_node = root;
 
    int countNodes = 0;
    // Identify number of nodes early on ?
    while (temp_node && queue != NULL) {

        countNodes++;
  
      if(temp_node->children) {
        int i;
        for(i=0; i<4; i++) {
          enQueue(queue, &rear, &temp_node->children[i]);
        }
      }
 
        /*Dequeue node and make it temp_node*/
        temp_node = deQueue(queue, &front);
    }

     free(queue);
     return countNodes;
}

/* Given a binary tree, print its nodes in level order
   using array for implementing queue */
double* printLevelOrder(node_t* root, int numParticles, int numNodes)
{
    
    // Queue takes a very large allocation of memory (4*nparticles!) or crashes
    // The crashing is due to the number of nodes being at least four times the number of particles..
    // Makes very low level of particles (< 130) crash for some reason ?
    // Anyways, if the number of particles is low, queue buffer is 1000 nodes wide to stop crashing 
    // For low number of particles, I should stick to single CPU anyways and not call at all...
    
    //double check = root->n_particles + 0.0;
    //printf("Double check//Num of nodes: %f\n", check);
    // First Pass, get total number of nodes O(n)

    //double* serializedTree = (double*) malloc(sizeof(double) * numParticles * 14);
    
    
    
    int rear, front;
    node_t** queue = NULL;

    // Second Pass, allocate array with known number of nodes, and save it O(n)
    double* serializedTree = (double*) malloc(sizeof(double) * numNodes * 15);
    printf("Nodes:  %d \n",numNodes);
   
     queue = NULL;
     int indexSerializedTree = 0;
    
    queue = (struct node**)malloc(
    sizeof(struct node_t*) * numNodes); // memleak ???
    
    front = rear = 0;
    node_t* temp_node = root;
 
    // Identify number of nodes early on ?
    while (temp_node && queue != NULL) {
        //printf("nparticles %d \n",temp_node->n_particles);
          // For the nodes, could have a hasParticle check
        serializedTree[indexSerializedTree] =  temp_node->n_particles +0.0; // casting to double
        serializedTree[indexSerializedTree+1] = temp_node->x_min;
        serializedTree[indexSerializedTree+2] = temp_node->x_max;
        serializedTree[indexSerializedTree+3] = temp_node->y_min;
        serializedTree[indexSerializedTree+4] = temp_node->y_max;
        serializedTree[indexSerializedTree+5] = temp_node->depth + 0.0;
        serializedTree[indexSerializedTree+6] = temp_node->mass;
        serializedTree[indexSerializedTree+7] = temp_node->x_center;
        serializedTree[indexSerializedTree+8] = temp_node->y_center;
          if(temp_node->particle) {
        // Don't re-invent the wheel
        // get particle and node info
        // Serialize properly
        particle_t*p = temp_node->particle;
      
        serializedTree[indexSerializedTree+9] = 1.0;
        serializedTree[indexSerializedTree+10] = p->x_pos;
        serializedTree[indexSerializedTree+11] = p->y_pos;
        serializedTree[indexSerializedTree+12] = p->x_vel;
        serializedTree[indexSerializedTree+13] = p->y_vel;
        serializedTree[indexSerializedTree+14] = p->mass;
      }
      else
      {
        serializedTree[indexSerializedTree+9] = 0.0;
      }

      indexSerializedTree += 15;

      //printf("particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);

      if(temp_node->children) {
        int i;
        for(i=0; i<4; i++) {
          enQueue(queue, &rear, &temp_node->children[i]);
        }
      }
        //printf("%d ", temp_node->x_center);
 
        /*Dequeue node and make it temp_node*/
        temp_node = deQueue(queue, &front);
    }

    
    //printf("Particles:  %d \n",countingValue1);
    // Freeing allocated memory
    free(queue);
    
    //free(temp_node);
    return serializedTree;
}


void enQueue(node_t** queue, int* rear,
            node_t* new_node)
{
    queue[*rear] = new_node;
    (*rear)++;
}

node_t* deQueue(node_t** queue, int* front)
{
    (*front)++;
    return queue[*front - 1];
}




// END OF COPIED CODE

/* Deserialize array, and obtain a tree of particles*/
/* Should return a pointer to root */
/* Freeing memory ? */
/* num of particles changing between iterations*/
void array_to_tree(int numNodes, int numParticles, double* serializedTree)
{
  // First step: Transform serializedTree into array of particles
  node_t*nodeHeap;
  nodeHeap = malloc(sizeof(node_t)*numNodes);
   int i;
  // // Proceeding in a reverse loop (from numParticles to 0)
  // // to setup root relationships with children
  for(i=(numNodes-1);i>=0;i--)
  {
    node_t *temp_node = &nodeHeap[i];
    temp_node->n_particles  = serializedTree[i*15];
    temp_node->x_min = serializedTree[i*15+1];
    temp_node->x_max = serializedTree[i*15+2];
    temp_node->y_min = serializedTree[i*15+3];
    temp_node->y_max = serializedTree[i*15+4];
    temp_node->depth = serializedTree[i*15+5]; 
    temp_node->mass = serializedTree[i*15+6];
    temp_node->x_center = serializedTree[i*15+7];
    temp_node->y_center = serializedTree[i*15+8];

    // if(i*4<numNodes)
    // {
    //   temp_node->children = malloc(sizeof(node_t*)*4);
    //   temp_node->children[0] = nodeHeap[i*4+1];
    //   temp_node->children[1] = nodeHeap[i*4+2];
    //   temp_node->children[2] = nodeHeap[i*4+3];
    //   temp_node->children[3] = nodeHeap[i*4+4];
    // }

    //insert_particle(particle, root);
  }
  //free(particlesHeap);
  // Return root
  //return nodeHeap[0];
  // Second step: Recursively initialize Tree from array of particles
  // using 4-nary heap property, and return array of particles from function
  // Root can be deduced to be at index zero
}

/* print recursively the particles of a node */
void print_particles(FILE* f, node_t*n) {
  if(!n) {
    return;
  }

  if(n->particle) {
    particle_t*p = n->particle;
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
  if(n->children) {
    int i;
    for(i=0; i<4; i++) {
      print_particles(f, &n->children[i]);
    }
  }
}



/* Initialize a node */
void init_node(node_t* n, node_t* parent, double x_min, double x_max, double y_min, double y_max) {
  n->parent = parent;
  n->children = NULL;
  n->n_particles = 0;
  n->particle = NULL;
  n->x_min = x_min;
  n->x_max = x_max;
  n->y_min = y_min;
  n->y_max = y_max;
  n->depth = 0;

  int depth=1;
  while(parent) {
    if(parent->depth < depth) {
      parent->depth = depth;
      depth++;
    }
    parent = parent->parent;
  }

  n->mass= 0;
  n->x_center = 0;
  n->y_center = 0;

  assert(x_min != x_max);
  assert(y_min != y_max);
}


/* Compute the position of a particle in a node and return
 * the quadrant in which it should be placed
 */
int get_quadrant(particle_t* particle, node_t*node) {
  double x_min = node->x_min;
  double x_max = node->x_max;
  double x_center = x_min+(x_max-x_min)/2;

  double y_min = node->y_min;
  double y_max = node->y_max;
  double y_center = y_min+(y_max-y_min)/2;

  assert(particle->x_pos>=node->x_min);
  assert(particle->x_pos<=node->x_max);
  assert(particle->y_pos>=node->y_min);
  assert(particle->y_pos<=node->y_max);

  if(particle->x_pos <= x_center) {
    if(particle->y_pos <= y_center) {
      return 0;
    } else {
      return 2;
    }
  } else {
    if(particle->y_pos <= y_center) {
      return 1;
    } else {
      return 3;
    }
  }
}

/* inserts a particle in a node (or one of its children)  */
void insert_particle(particle_t* particle, node_t*node) {
#if 0
  assert(particle->x_pos >= node->x_min);
  assert(particle->x_pos <= node->x_max);
  assert(particle->y_pos >= node->y_min);
  assert(particle->y_pos <= node->y_max);

  assert(particle->node == NULL);
#endif
  if(node->n_particles == 0 &&
     node->children == NULL) {
    assert(node->children == NULL);

    /* there's no particle. insert directly */
    /* Case for the root node !!!! */
    node->particle = particle;
    node->n_particles++;

    node->x_center = particle->x_pos;
    node->y_center = particle->y_pos;
    node->mass = particle->mass;

    // Create a link particle(node) <--> node
    particle->node = node;
    assert(node->children == NULL);
    return;
  } else {
    /* There's already a particle */

    if(! node->children) {
      /* there's no children yet */
      /* create 4 children and move the already-inserted particle to one of them */
      //assert(node->x_min != node->x_max);
      node->children = alloc_node();
      double x_min = node->x_min;
      double x_max = node->x_max;
      double x_center = x_min+(x_max-x_min)/2;

      double y_min = node->y_min;
      double y_max = node->y_max;
      double y_center = y_min+(y_max-y_min)/2;

      init_node(&node->children[0], node, x_min, x_center, y_min, y_center);
      init_node(&node->children[1], node, x_center, x_max, y_min, y_center);
      init_node(&node->children[2], node, x_min, x_center, y_center, y_max);
      init_node(&node->children[3], node, x_center, x_max, y_center, y_max);

      /* move the already-inserted particle to one of the children */
      particle_t*ptr = node->particle;
      //assert(ptr->node == node);
      int quadrant = get_quadrant(ptr, node);
      node->particle = NULL;
      ptr->node = NULL;

      insert_particle(ptr, &node->children[quadrant]);
    }

    /* insert the particle to one of the children */
    int quadrant = get_quadrant(particle, node);
    node->n_particles++;

    //assert(particle->node == NULL);
    insert_particle(particle, &node->children[quadrant]);

    /* update the mass and center of the node */
    double total_mass = 0;
    double total_x = 0;
    double total_y = 0;
    int i;
    for(i=0; i<4; i++) {
      total_mass += node->children[i].mass;
      total_x += node->children[i].x_center*node->children[i].mass;
      total_y += node->children[i].y_center*node->children[i].mass;
    }
    node->mass = total_mass;
    node->x_center = total_x/total_mass;
    node->y_center = total_y/total_mass;
#if 0
    assert(node->particle == NULL);
    assert(node->n_particles > 0);
#endif
  }
}

/*
  Place particles in their initial positions.
*/
void all_init_particles(int num_particles, particle_t *particles)
{
  int    i;
  double total_particle = num_particles;

  for (i = 0; i < num_particles; i++) {
    particle_t *particle = &particles[i];
#if 0
    particle->x_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
    particle->y_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
    particle->x_vel = particle->y_pos;
    particle->y_vel = particle->x_pos;
#else
    particle->x_pos = i*2.0/nparticles - 1.0;
    particle->y_pos = 0.0;
    particle->x_vel = 0.0;
    particle->y_vel = particle->x_pos;
#endif
    particle->mass = 1.0 + (num_particles+i)/total_particle;
    particle->node = NULL;

    //insert_particle(particle, root);
  }
}


struct memory_t mem_node;

void init_alloc(int nb_blocks) {
  mem_init(&mem_node, 4*sizeof(node_t), nb_blocks);
}

/* allocate a block of 4 nodes */
node_t* alloc_node() {
  node_t*ret = mem_alloc(&mem_node);
  return ret;
}

void free_root(node_t*root) {
  free_node(root);
  mem_free(&mem_node, root);
}

void free_node(node_t* n) {
  if(!n) return;

  if(n->children) {
    //assert(n->n_particles > 0);
    int i;
    for(i=0; i<4; i++) {
      free_node(&n->children[i]);
    }
    mem_free(&mem_node, n->children);
  }
}
