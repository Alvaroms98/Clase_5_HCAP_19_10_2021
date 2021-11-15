#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <string.h>
#include <malloc.h>

void Print_matrix(double C[], int n);
void To_blocked(double A[], int n, int b);
void From_blocked(double C[], int n, int b);
void Blocked_mat_mult(double *A, double *B, double *C,  int n_bar,int tb);
void Mult_add(double *A, double *B, double *C,int i_bar, int j_bar, int k_bar, int n_bar, int tb);
void Mult_add_avx(double *A, double *B, double *C,int i_bar, int j_bar, int k_bar, int n_bar, int tb);
void matprod(double *A, double *B, double *C, int n);
void print_double_avx(__m256d *vec, int fila, int columna);

#define ALIGN 16


int main( int argc, char *argv[] ) {

  int  n=3200,i,k, tb=16;
  clock_t tic,toc;
  int nb=n/tb; //numero de bloques
  
  
  double *A       = (double *) _mm_malloc(n*n*sizeof(double),ALIGN);
  double *B       = (double *) _mm_malloc(n*n*sizeof(double),ALIGN);
  double *C       = (double *) _mm_malloc(n*n*sizeof(double),ALIGN);
  /* Reservamos memoria para los datos */



  /* Lo probamos */
  int j;
   for( j=0; j<n; j++ ) {
      for( i=0; i<n; i++ ) {
         A[i+j*n] = ((double) rand()/ RAND_MAX);
      }
   }
   for( j=0; j<n; j++ ) {
      for( i=0; i<n; i++ ) {
         B[i+j*n] = ((double) rand()/ RAND_MAX);
      }
   } 
   tic = clock();
   matprod(A,B,C,n);
   toc = clock();
   printf("\n Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

   //Print_matrix(C, n);

   for( j=0; j<n; j++ ) {
      for( i=0; i<n; i++ ) {
         C[i+j*n] = 0.0;
      }
   }
       /*a bloques con copia*/  
   tic = clock();
   To_blocked(A, n, tb);
   To_blocked(B, n, tb);
   Blocked_mat_mult(A,B,C,nb,tb);
   From_blocked(C, n, tb);
   toc = clock();
   printf("\n Elapsed bloques con copia: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

   //Print_matrix(C, n);       
   _mm_free(A);
   _mm_free(B);
   _mm_free(C);
      return 0;
}





void Print_matrix(double C[], int n) {
   int i, j;

   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         printf("%.2e ", C[i+j*n]);
      printf("\n");
   }
}  /* Print_matrix */

void To_blocked(double A[], int n, int b) {
   int i, j;
   int i_bar, j_bar;  // index block rows and block cols
   int n_bar = n/b;
   double *T, *a_p, *t_p;

   T = malloc(n*n*sizeof(double));
   if (T == NULL) {
      fprintf(stderr, "Can't allocate temporary in To_blocked\n");
      exit(-1);
   }

   // for each block in A
   t_p = T;
   for (j_bar = 0; j_bar < n_bar; j_bar++)
      for (i_bar = 0; i_bar < n_bar; i_bar++) {

         // Copy block into contiguous locations in T
         a_p = A + (i_bar*b + j_bar*b*n);
         for (j = 0; j < b; j++, a_p += (n-b)) 
            for (i = 0; i < b; i++) {
               *t_p++ = *a_p++;
            }
      }   

   memcpy(A, T, n*n*sizeof(double));

   free(T);
}  /* To_blocked */

void From_blocked(double C[], int n, int b) {
   int i, j;
   int i_bar, j_bar;  // index blocks of C
   int n_bar = n/b;
   double *T, *c_p, *t_p;

   T = malloc(n*n*sizeof(double));
   if (T == NULL) {
      fprintf(stderr, "Can't allocate temporary in To_blocked\n");
      exit(-1);
   }

   // for each block of C
   c_p = C;
   for (j_bar = 0; j_bar < n_bar; j_bar++)
      for (i_bar = 0; i_bar < n_bar; i_bar++) {

         // Copy block into correct locations in T
         t_p = T + (i_bar*b + j_bar*b*n);
         for (j = 0; j < b; j++, t_p += (n-b))
            for (i = 0; i < b; i++) {
               *t_p++ = *c_p++;
            }
      }

   memcpy(C, T, n*n*sizeof(double));
   free(T);
}  /* From_bloc */

void Blocked_mat_mult(double *A, double *B, double *C,  int n_bar,int tb)
{
   int i_bar, j_bar, k_bar;  // index block rows and columns

   for (j_bar = 0; j_bar < n_bar; j_bar++)
      for (k_bar = 0; k_bar < n_bar; k_bar++) {
       //  Zero_C(i_bar, j_bar);
         for (i_bar = 0; i_bar < n_bar; i_bar++) 
            Mult_add_avx(A, B, C, i_bar, j_bar, k_bar,n_bar,tb);
      }
}  /* Blocked_mat_mult */

/*-------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read a matrix from stdin
 * In arg:    n:  order of matrix
 * Out arg:   A:  the matrix
 */
 void Mult_add(double *A, double *B, double *C,int i_bar, int j_bar, int k_bar, int n_bar, int tb) 
 {
   int b_sqr=tb*tb;
   double *c_p = C+(i_bar + j_bar*n_bar)*b_sqr;;
   double *a_p = A + (i_bar + k_bar*n_bar)*b_sqr;
   double *b_p = B + (k_bar+ j_bar*n_bar )*b_sqr;
   int i, j, k;
   for (j = 0; j < tb; j++)
      for (k = 0; k < tb; k++) 
         for (i = 0; i < tb; i++)
             c_p[i+j*tb] += a_p[i+k*tb]*(b_p[k+j*tb]);
      //      *(c_p + i*tb + j) += 
      //         (*(a_p + i*tb+k))*(*(b_p + k*tb + j));
}  /* Mult_add */ 


void Mult_add_avx(double *A, double *B, double *C,int i_bar, int j_bar, int k_bar, int n_bar, int tb) 
{
    int b_sqr=tb*tb;
    double *c_p = C +( i_bar + j_bar*n_bar)*b_sqr;
    double *a_p = A + (i_bar + k_bar*n_bar)*b_sqr;
    double *b_p = B + (k_bar + j_bar*n_bar)*b_sqr;

    __m256d a_vec,tmp,b_vec;
    double suma = 0.0;

   int i, j, k;
    for (j = 0; j < tb; j++){
        for (i = 0; i < tb; i++){
            for (k = 0; k < tb; k+=4){
                a_vec = _mm256_set_pd(a_p[i+(k+3)*tb], a_p[i+(k+2)*tb], a_p[i+(k+1)*tb], a_p[i+(k+0)*tb]);
                b_vec = _mm256_load_pd(&b_p[k+j*tb]);

                tmp = _mm256_mul_pd(a_vec, b_vec);

                tmp = _mm256_hadd_pd(tmp,tmp);

                suma += tmp[0] + tmp[2];
            }
            c_p[i+j*tb] += suma;
            suma = 0.0;
        }
    }
}

void print_double_avx(__m256d *vec, int fila, int columna){
   double elem [4];
   memcpy(elem,vec,sizeof(double)*4);
   printf("avx[%d,%d]: %f %f %f %f\n", fila,columna,elem[0],elem[1],elem[2],elem[3]);
}



void matprod(double *A, double *B, double *C, int n)
{
    int i,j,k;
    for (j=0; j<n; j++) 
         for(k=0;k<n; k++)
             for(i=0;i<n; i++)
       {
        C[i+j*n]= C[i+j*n]+ A[i+k*n]* B[k+j*n];
    }
}