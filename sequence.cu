// Elapsed Real Time for input-4.txt: 0m1.124s

#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

// Input sequence of integers.
int *vList;

// Number of integers on the list.
int vCount = 0;

// Capacity of the list of integers.
int vCap = 0;

// Target sum.
int target_sum;

// General function to report a failure and exit.
static void fail( char const *message ) {
  fprintf( stderr, "%s\n", message );
  exit( 1 );
}

// Print out a usage message, then exit.
static void usage() {
  printf( "usage: sequence <target_sum> [report]\n" );
  exit( 1 );
}

// Read the list of values.
__host__ void readList() {
  // Set up initial list and capacity.
  vCap = 5;
  vList = (int *) malloc( vCap * sizeof( int ) );

  // Keep reading as many values as we can.
  int v;
  while ( scanf( "%d\n", &v ) == 1 ) {
    // Grow the list if needed.
    if ( vCount >= vCap ) {
      vCap *= 2;
      vList = (int *) realloc( vList, vCap * sizeof( int ) );
    }

    // Store the latest value in the next array slot.
    vList[ vCount++ ] = v;
  }
}


//Kernel function to find subsequences that sum to the target
__global__ void checkSum(int *d_vList, int *d_results, int vCount_d, int target_sum, bool report) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (idx < vCount_d) {
    int local_count = 0;

    // Iterate through all subsequences starting at idx.
    int sequenceCount = 0;
    for (int k = idx; k >= 0; k--) {
      sequenceCount += d_vList[k];


      // Check if the local sum matches the target.
      if (sequenceCount == target_sum) {
        local_count++;
        if (report) {
          printf("I’m thread %d. Local count: %d. Sequence found at: %d-%d.\n", idx, local_count, k, idx);
        }
      }
    }

    if (report && local_count == 0){
      printf("I’m thread %d. Local count: 0.\n", idx);
    }
    // Store the local count in the results array.
    d_results[idx] = local_count;
  }
}

int main( int argc, char *argv[] ) {
  if ( argc < 2 || argc > 3 )
    usage();

  if ( sscanf( argv[ 1 ], "%d", &target_sum ) != 1)
    usage();

  bool report = false;
  if ( argc == 3 ) {
    if ( strcmp( argv[ 2 ], "report" ) != 0 )
      usage();
    report = true;
  }

  readList();



  // Implementation start....

  // Add code to allocate memory on the device and copy over the list.
  int *d_vList, *d_results;
  cudaMalloc((void **)&d_vList, vCount * sizeof(int));

  // Add code to copy the list over to the device.
  cudaMalloc((void **)&d_results, vCount * sizeof(int));

  // Add code to allocate space on the device to hold the results.
  cudaMemcpy(d_vList, vList, vCount * sizeof(int), cudaMemcpyHostToDevice);

  // Block and grid dimensions.
  int threadsPerBlock = 100;
  // Round up for the number of blocks we need.
  int blocksPerGrid = (vCount + threadsPerBlock - 1) / threadsPerBlock;

  // Run our kernel on these block/grid dimensions (you'll need to add some parameters)
  checkSum<<<blocksPerGrid, threadsPerBlock>>>(d_vList, d_results, vCount, target_sum, report);
  if (cudaGetLastError() != cudaSuccess)
    fail("Failure in CUDA kernel execution.");

  // Add code to copy results back to the host, add up all the per-thread counts 
  // and report one global total count.
  int *h_results = (int *) malloc(vCount * sizeof(int));
  cudaMemcpy(h_results, d_results, vCount * sizeof(int), cudaMemcpyDeviceToHost);
  int total_count = 0;
  for (int i = 0; i < vCount; i++) {
    total_count += h_results[i];
  }
  printf("Total count: %d\n", total_count);

  // Free memory on the device and the host.
  cudaFree(d_vList);
  cudaFree(d_results);
  free(vList);
  free(h_results);

  cudaDeviceReset();
  return 0;
}
