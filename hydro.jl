using CUDA
using PyPlot
using HDF5
push!(LOAD_PATH, "/home/bruno/Desktop/Dropbox/Developer/julia/tools")
using tools
# using cudaTools



nPoints = 128

# set simulation volume dimentions
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth
Lx = 1.0
Ly = 1.0
Lz = 1.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
X = linspace( xMin, xMax, nWidth )
Y = linspace( yMin, yMax, nHeight )
Z = linspace( zMin, zMax, nDepth )
grid_coords = [(x,y,z) for x in X, y in Y, z in Z ]
# R = [ sqrt( x*x + y*y + z*z) for x in X, y in Y, z in Z]
sphereR = 0.25
sphereOffCenter = 0.25
R( pos ) = sqrt( pos[1]^2 + pos[2]^2 + pos[3]^2 )
R_left( pos )  = sqrt( (pos[1]+sphereOffCenter)^2 + pos[2]^2 + pos[3]^2 )
R_right( pos ) = sqrt( (pos[1]-sphereOffCenter)^2 + pos[2]^2 + pos[3]^2 )
sphere = find( coord-> R(coord) < sphereR, grid_coords )
sphere_left  = find( coord-> R_left(coord) < sphereR, grid_coords )
sphere_right = find( coord-> R_right(coord) < sphereR, grid_coords )
not_sphere = find( coord-> R(coord) > sphereR, grid_coords )
not_sphere_left  = find( coord-> R_left(coord) > sphereR, grid_coords )
not_sphere_right = find( coord-> R_right(coord) > sphereR, grid_coords )
spheres = [ sphere_left ; sphere_right ]
not_spheres = [ not_sphere_left ; not_sphere_right ]

const gamma = Float64( 7./5. )
const c0 = Float64( 0.4 )

#set time parameters
const nIterations = 100
const nPartialSteps = 10

#Set output file
outDir = "/home/bruno/Desktop/data/"
outFileName = "hydro_data.h5"
outFile = h5open( outDir*outFileName, "w")


#Convert parameters to float64
xMin = Float64(xMin)
yMin = Float64(yMin)
zMin = Float64(zMin)
dx = Float64(dx)
dy = Float64(dy)
dz = Float64(dz)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 16,4,2  #hardcoded, tune to your needs
gridx = divrem(nWidth , block_size_x)[1] + 1 * ( nWidth % block_size_x != 0 )
gridy = divrem(nHeight , block_size_y)[1] + 1 * ( nHeight % block_size_y != 0 )
gridz = divrem(nDepth , block_size_z)[1] + 1 * ( nDepth % block_size_z != 0 )
grid3D = (gridx, gridy, gridz)
block3D = (block_size_x, block_size_y, block_size_z)

# select a CUDA device
# list_devices()
dev = CuDevice(0)
# create a context (like a process in CPU) on the selected device
ctx = create_context(dev)

#Read and compile CUDA code
println( "Compiling CUDA code" )
run(`nvcc -ptx hydro_kernels.cu`)
cudaModule = CuModule("hydro_kernels.ptx")
setInterFlux_hll = CuFunction( cudaModule, "setInterFlux_hll")
getInterFlux_hll = CuFunction( cudaModule, "getInterFlux_hll")
copyDtoD = CuFunction( cudaModule, "copyDtoD")
addDtoD  = CuFunction( cudaModule, "addDtoD")
write_bounderies = CuFunction( cudaModule, "write_bounderies" )
########################################################################
const dt = 0.000995 /10
function timeStepHydro()
  for coord in [ 1, 2, 3]
    CUDA.launch( setInterFlux_hll, grid3D, block3D,
    ( Int32( coord ),  gamma , dx, dy, dz,
      cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d,
      iFlx1_d, iFlx2_d, iFlx3_d, iFlx4_d, iFlx5_d,
      times_d ) )
    # if coord == 1: dt = c0 * gpuarray.min( times_d ).get()

    CUDA.launch( getInterFlux_hll, grid3D, block3D,
    ( Int32( coord ),  Float64(dt), gamma ,
      Int32(nWidth), Int32(nHeight), Int32(nDepth), dx, dy, dz,
      cnsv1_adv_d, cnsv2_adv_d, cnsv3_adv_d, cnsv4_adv_d, cnsv5_adv_d,
      iFlx1_d, iFlx2_d, iFlx3_d, iFlx4_d, iFlx5_d ) )
  end
  CUDA.launch( addDtoD, grid3D, block3D,
              (cnsv1_d, cnsv2_d, cnsv3_d, cnsv4_d, cnsv5_d,
              cnsv1_adv_d, cnsv2_adv_d, cnsv3_adv_d, cnsv4_adv_d, cnsv5_adv_d) )
end
########################################################################
function dynamics( nIter, data_d, data_h; transferEnd=false )
  for i in 1:nIter
    timeStepHydro()
  end
  if transferEnd
    copy!( data_h, data_d )
  end
end

# Initialize data and load it to the GPU
println( "Initialing Data")
rho = 0.7 * ones( Float64, (nWidth, nHeight,nDepth) )
vx  = zeros( Float64, (nWidth, nHeight,nDepth) )
vy  = zeros( Float64, (nWidth, nHeight,nDepth) )
vz  = zeros( Float64, (nWidth, nHeight,nDepth) )
p   = ones( Float64, (nWidth, nHeight,nDepth) )
#####################################################
#Initialize two offset spheres
rho[ sphere ] = 0.95
# rho[ not_spheres ] = 0.1
p[ sphere ] = 10.
# p[ not_sphere_left ] = 1.
v2 = vx .* vx + vy .* vy + vz .* vz
#####################################################
#Initialize conserved values
cnsv1_h = rho
cnsv2_h = rho .* vx
cnsv3_h = rho .* vy
cnsv4_h = rho .* vz
cnsv5_h = ( rho .* v2/2. ) + ( p ./ (gamma-1) )
# Memory for BOUNDERIES
bound_l_h = zeros( Float64, ( nHeight, nDepth ) )
bound_r_h = zeros( Float64, ( nHeight, nDepth ) )
bound_d_h = zeros( Float64, ( nWidth, nDepth ) )
bound_u_h = zeros( Float64, ( nWidth, nDepth ) )
bound_b_h = zeros( Float64, ( nWidth, nHeight ) )
bound_t_h = zeros( Float64, ( nWidth, nHeight ) )
#####################################################
#Initialize device global data
cnsv1_d = CuArray( cnsv1_h )
cnsv2_d = CuArray( cnsv2_h )
cnsv3_d = CuArray( cnsv3_h )
cnsv4_d = CuArray( cnsv4_h )
cnsv5_d = CuArray( cnsv5_h )
cnsv1_adv_d = CuArray( cnsv1_h )
cnsv2_adv_d = CuArray( cnsv2_h )
cnsv3_adv_d = CuArray( cnsv3_h )
cnsv4_adv_d = CuArray( cnsv4_h )
cnsv5_adv_d = CuArray( cnsv5_h )
times_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
iFlx1_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
iFlx2_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
iFlx3_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
iFlx4_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
iFlx5_d = CuArray( zeros( Float64, (nWidth, nHeight,nDepth) ) )
bound_l_d = CuArray( bound_l_h )
bound_r_d = CuArray( bound_r_h )
bound_d_d = CuArray( bound_d_h )
bound_u_d = CuArray( bound_u_h )
bound_b_d = CuArray( bound_b_h )
bound_t_d = CuArray( bound_t_h )


CUDA.launch( write_bounderies, grid3D, block3D,
        ( Int32(nWidth), Int32(nHeight), Int32(nDepth),
          bound_l_d, bound_r_d, bound_d_d, bound_u_d, bound_b_d, bound_t_d))

bound_l_h = to_host( bound_l_d )
bound_r_h = to_host( bound_r_d )
bound_d_h = to_host( bound_d_d )
bound_u_h = to_host( bound_u_d )
bound_b_h = to_host( bound_b_d )
bound_t_h = to_host( bound_t_d )

#
# nIterPerStep, reminderSteps = divrem( nIterations, nPartialSteps )
# totalTime = [ 0.0, 0.0, 0.0 ]
# println( "\nnSnapshots: $nPartialSteps \nOutput: $(outDir*outFileName)\n" )
# println( "Starting $nIterations iterations...\n")
# writeSnapshot( 0, "rho", cnsv1_h, outFile, stride=1 )
# for i in 1:nPartialSteps
#   printProgress( i-1, nPartialSteps, sum(totalTime) )
#   totalTime[1] += @elapsed dynamics( nIterPerStep, cnsv1_d, cnsv1_h, transferEnd=true )
#   totalTime[2] += @elapsed writeSnapshot( i, "rho", cnsv1_h, outFile, stride=1 )
# #
# end
# printProgress( nPartialSteps, nPartialSteps, sum(totalTime) )
# println( "\nTotal Time: $(sum(totalTime)) secs" )
# println( "Compute Time: $(totalTime[1]) secs" )
# println( "Write Time: $(totalTime[2]) secs" )
# println( "Transfer Time: $(totalTime[3]) secs" )


close( outFile )


#
# function dynamics( nIter, data_d, data_h; transferEnd=false )
#   for i in 1:nIter
#     rk4_step()
#   end
#   if transferEnd
#     copy!( data_h, data_d )
#   end
# end
#
#
#


# sumation_h = ones( Float64,  (nWidth, nHeight,nDepth) )
# sumation_d = CuArray( sumation_h )
# #Arrays for reductions
# const blockSize_reduc = 256
# const gridSize_reduc = round( Int, nData / blockSize_reduc  / 2  )
# const last_gridSize = round( Int, gridSize_reduc / blockSize_reduc / 2 )
# prePartialSum_d = CuArray( zeros( Float64, (gridSize_reduc) ) )
# partialSum_h = zeros( Float64, (last_gridSize ) )
# partialSum_d = CuArray( partialSum_h )
#
# function reduction( sumation_d, prePartialSum_d, partialSum_h, partialSum_d )
#   CUDA.launch( reduction_kernel, gridSize_reduc , blockSize_reduc, ( sumation_d, prePartialSum_d ) )
#   CUDA.launch( reduction_kernel, last_gridSize, blockSize_reduc, ( prePartialSum_d, partialSum_d ) )
#   copy!( partialSum_h, partialSum_d )
#   return sum( partialSum_h )
# end
# println( reduction( sumation_d, prePartialSum_d, partialSum_h, partialSum_d ) )
# for i in 1:100
#   totalTime += @elapsed reduction( sumation_d, prePartialSum_d, partialSum_h, partialSum_d )
# end
