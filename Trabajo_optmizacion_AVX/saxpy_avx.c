#define LIM 16
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>

float randfloat() 
{ 
    float r= (float)(rand());
    float div =(float)(1-RAND_MAX);
    return r/div; 
} 




int main(){
  
float *y,*x,*yc,a;
int i,j;
a=2.0;
y  = (float*) malloc(sizeof(float)*LIM);
yc = (float*) malloc(sizeof(float)*LIM);
x  = (float*) malloc(sizeof(float)*LIM);

for (i=0;i<LIM;i++)
{x[i]=-randfloat();
 y[i]=-randfloat();
  yc[i]=y[i];
}



 printf("vector y inicial \n ");
for (i=0;i<LIM;i++)
  {//printf("%f ",x[i]);
   printf("%f ",y[i]);
  }
  printf("\n ");

//saxpy
for (i=0;i<LIM;i++)
 y[i]=a*x[i]+y[i];
  
 printf("vector y tras saxpy escalar: \n ");

for (i=0;i<LIM;i++)
  {//printf("%f ",x[i]);
   printf("%f ",y[i]);
  }
 
for (i=0;i<LIM;i++)
  y[i]=yc[i];

  printf("\n ");

//Poner aqui version de saxpy con AVX


__m256 *y_avx = (__m256*) y;
__m256 *x_avx = (__m256*) x;
__m256 a_avx = _mm256_set1_ps(a);

for (i=0;i<LIM;i+=8){
    *y_avx = _mm256_add_ps(*y_avx, _mm256_mul_ps(*x_avx, a_avx));
    y_avx++;x_avx++;
}



 printf("vector y tras saxpy avx: \n ")  ;

for (i=0;i<LIM;i++)
  {//printf("%f ",x[i]);
   printf("%f ",y[i]);
  }

  free(y);
  free(yc);
  free(x);
}