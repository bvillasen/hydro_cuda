#define  B_W 10
#define  B_H 10
#define  B_D 10

extern "C"   // ensure function name to be exactly "vadd"
{

  // __global__ void setBounderies( 
  //        double* cnsv_1, double* cnsv_2, double* cnsv_3, double* cnsv_4, double* cnsv_5,
  //        double* iFlx_1, double* iFlx_2, double* iFlx_3, double* iFlx_4, double* iFlx_5,
  //        double* times ){
  //   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  //   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  //   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  //   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  //




  __device__ void load_bounderies( const int nWidth, const int nHeight, const int nDepth,
    const int t_j, const int t_i, const int t_k, const int tid, double *out_l, double *out_r,
    double *out_d, double *out_u, double *out_b, double *out_t){
      int t_j = blockIdx.x*blockDim.x + threadIdx.x;
      int t_i = blockIdx.y*blockDim.y + threadIdx.y;
      int t_k = blockIdx.z*blockDim.z + threadIdx.z;
      int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

      int boundId;
      //X BOUNDERIES
      boundId = t_i + t_k*nHeight;
      if ( t_j==0 )        out_l[ boundId ] = tid;
      if ( t_j==nWidth-1 ) out_r[ boundId ] = tid;

      //Y BOUNDERIES
      boundId = t_j + t_k*nWidth;
      if  ( t_i==0 )         out_d[ boundId ] = tid;
      if  ( t_i==nHeight-1 ) out_u[ boundId ] = tid;

      //Z BOUNDERIES
      boundId = t_j + t_i*nWidth;
      if ( t_k==0 )         out_b[ boundId ] = tid;
      if ( t_k==nDepth-1 )  out_t[ boundId ] = tid;

    }

__device__ void write_bounderies( const int nWidth, const int nHeight, const int nDepth,
  const int t_j, const int t_i, const int t_k, const int tid, double *out_l, double *out_r,
  double *out_d, double *out_u, double *out_b, double *out_t){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    int boundId;
    //X BOUNDERIES
    boundId = t_i + t_k*nHeight;
    if ( t_j==0 )        out_l[ boundId ] = tid;
    if ( t_j==nWidth-1 ) out_r[ boundId ] = tid;

    //Y BOUNDERIES
    boundId = t_j + t_k*nWidth;
    if  ( t_i==0 )         out_d[ boundId ] = tid;
    if  ( t_i==nHeight-1 ) out_u[ boundId ] = tid;

    //Z BOUNDERIES
    boundId = t_j + t_i*nWidth;
    if ( t_k==0 )         out_b[ boundId ] = tid;
    if ( t_k==nDepth-1 )  out_t[ boundId ] = tid;

  }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  __global__ void setInterFlux_hll( const int coord, const double gamma, const double dx, const double dy, const double dz,
  			 double* cnsv_1, double* cnsv_2, double* cnsv_3, double* cnsv_4, double* cnsv_5,
         double* iFlx_1, double* iFlx_2, double* iFlx_3, double* iFlx_4, double* iFlx_5,
  			 double* times ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    int tid_adj;
    double v2;
    double rho_l, vx_l, vy_l, vz_l, E_l, p_l;
    double rho_c, vx_c, vy_c, vz_c, E_c, p_c;

    //Set adjacent id
    if ( coord == 1 ){
      if ( t_j == 0) tid_adj = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    if ( coord == 2 ){
      if ( t_i == 0) tid_adj = t_j + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    if ( coord == 3 ){
      if ( t_k == 0) tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    //Read adjacent and center conservatives
    rho_l = cnsv_1[ tid_adj ];
    rho_c = cnsv_1[ tid ];

    vx_l = cnsv_2[ tid_adj ] / rho_l;
    vx_c = cnsv_2[ tid ] / rho_c;

    vy_l = cnsv_3[ tid_adj ] / rho_l;
    vy_c = cnsv_3[ tid ] / rho_c;

    vz_l = cnsv_4[ tid_adj ] / rho_l;
    vz_c = cnsv_4[ tid ] / rho_c;

    E_l = cnsv_5[ tid_adj ];
    E_c = cnsv_5[ tid ];

  //   //Boundary bounce condition
  //     if ( t_j == 0 ) vx_l = -vx_c;
  //       //Boundary bounce condition
  //     if ( t_i == 0 ) vy_l = -vy_c;
  //     //Boundary bounce condition
  //     if ( t_k == 0 ) vz_l = -vz_c;

    v2    = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l;
    p_l   = ( E_l - rho_l*v2/2 ) * (gamma-1);

    v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
    p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);


    double cs_l, cs_c, s_l, s_c;
    cs_l = sqrt( p_l * gamma / rho_l );
    cs_c = sqrt( p_c * gamma / rho_c );

    if ( coord == 1 ){
      s_l = min( vx_l - cs_l, vx_c - cs_c );
      s_c = max( vx_l + cs_l, vx_c + cs_c );
      //Use v2 to save time minimum
      v2 = dx / ( abs( vx_c ) + cs_c );
      v2 = min( v2, dy / ( abs( vy_c ) + cs_c ) );
      v2 = min( v2, dz / ( abs( vz_c ) + cs_c ) );
      times[ tid ] = v2;
    }

    else if ( coord == 2 ){
      s_l = min( vy_l - cs_l, vy_c - cs_c );
      s_c = max( vy_l + cs_l, vy_c + cs_c );
    }

    else if ( coord == 3 ){
      s_l = min( vz_l - cs_l, vz_c - cs_c );
      s_c = max( vz_l + cs_l, vz_c + cs_c );
    }

    // Adjacent fluxes from left and center cell
    double F_l, F_c, iFlx;

    //iFlx rho
    if ( coord == 1 ){
      F_l = rho_l * vx_l;
      F_c = rho_c * vx_c;
    }
    else if ( coord == 2 ){
      F_l = rho_l * vy_l;
      F_c = rho_c * vy_c;
    }
    else if ( coord == 3 ){
      F_l = rho_l * vz_l;
      F_c = rho_c * vz_c;
    }
    if ( s_l > 0 ) iFlx = F_l;
    else if ( s_c < 0 ) iFlx = F_c;
    else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c - rho_l ) ) / ( s_c - s_l );
    iFlx_1[tid] = iFlx;

    //iFlx rho * vx
    if ( coord == 1 ){
      F_l = rho_l * vx_l * vx_l + p_l;
      F_c = rho_c * vx_c * vx_c + p_c;
    }
    else if ( coord == 2 ){
      F_l = rho_l * vx_l * vy_l;
      F_c = rho_c * vx_c * vy_c;
    }
    else if ( coord == 3 ){
      F_l = rho_l * vx_l * vz_l;
      F_c = rho_c * vx_c * vz_c;
    }
    if ( s_l > 0 ) iFlx = F_l;
    else if ( s_c < 0 ) iFlx = F_c;
    else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vx_c - rho_l*vx_l ) ) / ( s_c - s_l );
    iFlx_2[tid] = iFlx;

    //iFlx rho * vy
    if ( coord == 1 ){
      F_l = rho_l * vy_l * vx_l ;
      F_c = rho_c * vy_c * vx_c ;
    }
    else if ( coord == 2 ){
      F_l = rho_l * vy_l * vy_l + p_l;
      F_c = rho_c * vy_c * vy_c + p_c;
    }
    else if ( coord == 3 ){
      F_l = rho_l * vy_l * vz_l;
      F_c = rho_c * vy_c * vz_c;
    }
    if ( s_l > 0 ) iFlx = F_l;
    else if ( s_c < 0 ) iFlx = F_c;
    else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vy_c - rho_l*vy_l ) ) / ( s_c - s_l );
    iFlx_3[tid] = iFlx;

    //iFlx rho * vz
    if ( coord == 1 ){
      F_l = rho_l * vz_l * vx_l ;
      F_c = rho_c * vz_c * vx_c ;
    }
    else if ( coord == 2 ){
      F_l = rho_l * vz_l * vy_l ;
      F_c = rho_c * vz_c * vy_c ;
    }
    else if ( coord == 3 ){
      F_l = rho_l * vz_l * vz_l + p_l ;
      F_c = rho_c * vz_c * vz_c + p_c ;
    }
    if ( s_l > 0 ) iFlx = F_l;
    else if ( s_c < 0 ) iFlx = F_c;
    else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vz_c - rho_l*vz_l ) ) / ( s_c - s_l );
    iFlx_4[tid] = iFlx;

    //iFlx E
    if ( coord == 1 ){
      F_l = vx_l * ( E_l + p_l ) ;
      F_c = vx_c * ( E_c + p_c ) ;
    }
    else if ( coord == 2 ){
      F_l = vy_l * ( E_l + p_l ) ;
      F_c = vy_c * ( E_c + p_c ) ;
    }
    else if ( coord == 3 ){
      F_l = vz_l * ( E_l + p_l ) ;
      F_c = vz_c * ( E_c + p_c ) ;
    }
    if ( s_l > 0 ) iFlx = F_l;
    else if ( s_c < 0 ) iFlx = F_c;
    else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( E_c - E_l ) ) / ( s_c - s_l );
    iFlx_5[tid] = iFlx;
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  __global__ void getInterFlux_hll( const int coord, const double dt,  const double gamma,
         const int nWidth, const int nHeight, const int nDepth,
         const double dx, const double dy, const double dz,
         double* cnsv_adv_1, double* cnsv_adv_2, double* cnsv_adv_3, double* cnsv_adv_4, double* cnsv_adv_5,
         double* iFlx_1, double* iFlx_2, double* iFlx_3, double* iFlx_4, double* iFlx_5 ){
  			//  double* gForceX, double* gForceY, double* gForceZ, double* gravWork ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    int tid_adj;
    double iFlx1_l, iFlx2_l, iFlx3_l, iFlx4_l, iFlx5_l;
    double iFlx1_r, iFlx2_r, iFlx3_r, iFlx4_r, iFlx5_r;
    double delta;

    //Set adjacent id
    if ( coord == 1 ){
      if ( t_j == nWidth-1 ) tid_adj = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      delta = dt / dx;
    }
    if ( coord == 2 ){
      if ( t_i == nHeight-1 ) tid_adj = t_j + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      delta = dt / dy;
    }
    if ( coord == 3 ){
      if ( t_k == nDepth-1) tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      delta = dt / dz;
    }

    //Read inter-cell fluxes
    iFlx1_l = iFlx_1[ tid ];
    iFlx1_r = iFlx_1[ tid_adj ];

    iFlx2_l = iFlx_2[ tid ];
    iFlx2_r = iFlx_2[ tid_adj ];

    iFlx3_l = iFlx_3[ tid ];
    iFlx3_r = iFlx_3[ tid_adj ];

    iFlx4_l = iFlx_4[ tid ];
    iFlx4_r = iFlx_4[ tid_adj ];

    iFlx5_l = iFlx_5[ tid ];
    iFlx5_r = iFlx_5[ tid_adj ];

    //Advance the consv values
    // cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
    // cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l ) + dt*gForceX[tid]*50;
    // cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l ) + dt*gForceY[tid]*50;
    // cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l ) + dt*gForceZ[tid]*50;
    // cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l ) + dt*gravWork[tid]*50;

    if ( coord == 1 ){
      cnsv_adv_1[ tid ] = -delta*( iFlx1_r - iFlx1_l );
      cnsv_adv_2[ tid ] = -delta*( iFlx2_r - iFlx2_l );
      cnsv_adv_3[ tid ] = -delta*( iFlx3_r - iFlx3_l );
      cnsv_adv_4[ tid ] = -delta*( iFlx4_r - iFlx4_l );
      cnsv_adv_5[ tid ] = -delta*( iFlx5_r - iFlx5_l );
    }
    else{
      cnsv_adv_1[ tid ] = cnsv_adv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
      cnsv_adv_2[ tid ] = cnsv_adv_2[ tid ] - delta*( iFlx2_r - iFlx2_l );
      cnsv_adv_3[ tid ] = cnsv_adv_3[ tid ] - delta*( iFlx3_r - iFlx3_l );
      cnsv_adv_4[ tid ] = cnsv_adv_4[ tid ] - delta*( iFlx4_r - iFlx4_l );
      cnsv_adv_5[ tid ] = cnsv_adv_5[ tid ] - delta*( iFlx5_r - iFlx5_l );
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  __global__ void reduction_kernel( double *input, double *output ){
    __shared__ double sh_data[256];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x  + threadIdx.x;
    sh_data[tid] = input[i] + input[i + blockDim.x*gridDim.x ] ;
    __syncthreads();

    for( unsigned int s = blockDim.x/2; s>0; s >>= 1){
      if ( tid < s ) sh_data[tid] += sh_data[tid+s];
      __syncthreads();
    }

    if ( tid == 0 ) output[ blockIdx.x ] = sh_data[0];
  }
  __global__ void copyDtoD( double *src, double *dst ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    dst[tid] = src[tid];
  }
  __global__ void addDtoD(
      double *dst_1, double *dst_2, double *dst_3, double *dst_4, double *dst_5,
      double *sum_1, double *sum_2, double *sum_3, double *sum_4, double *sum_5 ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    dst_1[tid] = dst_1[tid] + sum_1[tid];
    dst_2[tid] = dst_2[tid] + sum_2[tid];
    dst_3[tid] = dst_3[tid] + sum_3[tid];
    dst_4[tid] = dst_4[tid] + sum_4[tid];
    dst_5[tid] = dst_5[tid] + sum_5[tid];
  }
}
