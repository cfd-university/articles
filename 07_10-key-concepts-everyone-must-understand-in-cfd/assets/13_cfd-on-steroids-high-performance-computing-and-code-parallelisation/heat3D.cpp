/// this program solves the 3D heat equation on a 3D structured, Cartesian grid using MPI.
/**
 * The equation we want to solve can be expressed in the following way:
 * 
 * T_t = Dx * T_xx + Dy * T_yy + Dz * T_zz,
 * 
 * where we have made use of the subscript notation, where T_xx indicates a partial derivative of second order of T in 
 * the x direction, and similarily for T_yy and T_zz. Dx, Dy and Dz are the thermal diffusivity strengths that determine
 * how fast heat will propage in each direction. We assime isotropy here (which is common for air and metals) where all
 * directions have the same strength. T is the solution vector and represents the temperature that we are solving for.
 * 
 * We use a second order accurate central scheme for the space derivatives, i.e. we have (in 1D):
 * 
 * d^2 T(x) / dx^2 = T_xx ~= (T[i+1] - 2*T[i] + T[i-1]) / (dx^2)
 * 
 * which we can apply in each coordinate direction equivalently. dx is the spacing between to adjacent cells, i.e. the
 * distance from one cell to its neighbors. It can be different for the y and z direction, however, within the same
 * direction it is always constant. For the time derivative, we use a first order Euler time integration scheme like so:
 * 
 * dT(x) / dt = T_t ~= (T[n+1] - T[n]) / dt
 * 
 * Here, n is the timestep from the previous solution and n+1 is the timestep for the next solution. In this way we can
 * integrate our solution in time. Combining the two above approximations, we could write (dor a 1D equation)
 * 
 * T_t = Dx * T_xx =>
 * (T[n+1] - T[n]) / dt = Dx * (T[i+1] - 2*T[i] + T[i-1]) / (dx^2)
 * 
 * We can solve this for T[n+1] to yield:
 * 
 * T[n+1] = T[n] + (dt * Dx / (dx^2)) * (T[i+1] - 2*T[i] + T[i-1])
 * 
 * We have the information of the right hand side available, thus we can calculate T[n+1] for each i.
 * For i=0 or i=iend we need to specify boundary conditions and for all T[n] we need to specify initial conditions.
 * With those information available, we can loop over time and calculate an updated solution until the solution between
 * two consequtive time steps does not change more than a user-defined convergence threshold.
 * 
 * For more information on the heat equation, you may check the following link:
 * https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ws2016-2017/num_methods_i/heat.pdf
 */

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <array>
#include <fstream>
#include <limits>
#include <cmath>
#include <chrono>
#include <cassert>

#if defined(USE_MPI)
  #include "mpi.h"
#endif

// based on compiler flag, use either floats or doubles for floating point operations
#if defined(USE_SINGLE_PRECISION)
  using floatT = float;
  #define MPI_FLOAT_T MPI_FLOAT
#elif defined(USE_DOUBLE_PRECISION)
  using floatT = double;
  #define MPI_FLOAT_T MPI_DOUBLE
#endif

/// enum used to index over the respective coordinate direction
enum COORDINATE { X = 0, Y, Z };

/// enum used to access the respective direction on each local processor
/**
 * Each sub-domain is a cuboid that has neighbors to each side (6 in total). To facilitate easier indexing, this enum
 * holds the names of all the neighbors, which we use to index data, for example when sending and receiving between
 * processors. The below diagram shows how the labels correspond to a cuboid subdomain (note also the coordinate
 * directions X, Y and Z):
 *
 *       Y
 *       |
 *        _________________
 *       /.               /|
 *      / .      3       / |
 *     /_________|______/  |
 *    |   .      |4     |  |
 *    | 0--------/------|-1|
 *    |   ......5|......|..|    -- X
 *    |  /       |      |  /
 *    | /        2      | /
 *    |/________________|/
 *
 *   /
 *  Z
 *
 *  0: LEFT
 *  1: RIGHT
 *  2: BOTTOM
 *  3: TOP
 *  4: BACK
 *  5: FRONT
 */
enum DIRECTION  { LEFT = 0, RIGHT, BOTTOM, TOP, BACK, FRONT };

/// the number of physical dimensions, here 3 as we have a 3D domain
#define NUMBER_OF_DIMENSIONS 3

int main(int argc, char **argv)
{
  /// if USE_MPI is defined (see makefile), execute the following code
  #if defined(USE_MPI)

    /// default ranks and size (number of processors), will be rearranged by Cartesian topology
    int rankDefaultMPICOMM, sizeDefaultMPICOMM;

    /// status and requests for non-blocking communications, i.e. MPI_IAllreduce(...) and MPI_IRecv(...)
    MPI_Status  status     [NUMBER_OF_DIMENSIONS * 2];
    MPI_Status  postStatus [NUMBER_OF_DIMENSIONS    ];
    MPI_Request request    [NUMBER_OF_DIMENSIONS * 2];
    MPI_Request reduceRequest;

    /// buffers into which we write data that we want to send and receive using MPI
    /**
     * sendbuffer will be received into receivebuffer\
     */
    std::array<std::vector<floatT>, NUMBER_OF_DIMENSIONS * 2> sendBuffer;
    std::array<std::vector<floatT>, NUMBER_OF_DIMENSIONS * 2> receiveBuffer;

    /// initialise MPI and get default ranks and size
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankDefaultMPICOMM);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeDefaultMPICOMM);

    /// new MPI communicator for Cartesian topologies
    MPI_Comm MPI_COMM_CART;

    /// new rank and size for Cartesian topology
    int       rank, size;

    /// tag used later during MPI_Send(...)
    int       tagSend       [NUMBER_OF_DIMENSIONS * 2];

    /// tag used later during MPI_IRecv(...)
    int       tagReceive    [NUMBER_OF_DIMENSIONS * 2];

    /// the dimensions are equivalent to how we want our domain to be partitioned.
    /**
     * There are two options here, either we leave the number of dimensions for each coordinate direction to zero,
     * in which case the MPI library will determine the optimum size, or we can specify the size in which case MPI
     * will try to fullfill our request. If the MPI library can't work with the requested size, it will exit with an
     * error status.
     *
     * The number of entries multiplied can not be larger than the number of processors used to execute the code.
     * For example, defining the coordinates with 2, 2, 2 requires at least 8 processors, since 2 * 2 * 2 = 8. Had we
     * run our program using mpirun -n 4 bin/HeatEquation3D, the above would fail, as 2 * 2 * 2 > 4. The entries specify
     * how many subdomains in each (X, Y, Z) direction we want to have. So for example, specifying {2, 1, 3} would give
     * us 2 subdomains in the X direction, 1 in the y direction and 3 in the z direction, totaling 6 subdomain, thus, we
     * need to run MPI with 6 processor exactly.
     *
     * We can, however, also just specify on coordinate direction and leave the rest to zero, in which case MPI is
     * trying to partition our domain only in these directions. Specifying {2, 0, 0}, for example, and then run with
     * mpirun -n 4 bin/HeatEquation3D, will either result in {2, 2, 0} or {2, 0, 2}, which version will be taken depends
     * on the MPI implementation and may differ for different implementation.
     */
    int       dimension3D   [NUMBER_OF_DIMENSIONS    ] = {0, 0, 0};

    /// the coordinate in the current Cartesian topology for the sub processor
    /**
     * for simplicity, assume we have a 2D domain which is partitioned into 4 subdomains like so:
     *  ___________
     * |     |     |
     * |  1  |  3  |
     * |_____|_____|
     * |     |     |
     * |  0  |  2  |
     * |_____|_____|
     *
     * This corresponds to {2, 2} dimensions (as explained above), The following processors have now these coordinates
     * processor 0: {0, 0}
     * processor 1: {0, 1}
     * processor 2: {1, 0}
     * processor 3: {1, 1}
     */
    int       coordinates3D [NUMBER_OF_DIMENSIONS    ];

    /// flags to indicate if we have period boundary conditions
    /**
     * If set to false, we do not send any information on physical boundaries. If set to true, then we send information
     * on the boundary to the processor in the same direction, which also has a physical boundary. Considering the
     * example above, processor 2 and 3 would both send their right boundary information to processor 0 and 1,
     * respectively, which would receive it on their left boundary. Since we have physical boundaries without
     * periodicity, we set all of them to false.
     */
    const int periods3D     [NUMBER_OF_DIMENSIONS    ] = {false, false, false};

    /// neighbors hold the rank of the neighboring processors and are accessed with the DIRECTION enum
    /**
     * In the example above, we have (for example for processor 0):
     * neighbors[DIRECTION::RIGHT ] = 2
     * neighbors[DIRECTION::TOP   ] = 1
     * neighbors[DIRECTION::BOTTOM] = MPI_PROC_NULL
     * neighbors[DIRECTION::LEFT  ] = MPI_PROC_NULL
     *
     * if we don't have a neighbor processor on one side, i.e. if we have physical boundary faces, then we assign the
     * special handle MPI_PROC_NULL, which, whenever it is encountered in a send or receive as a target or source is
     * ignored. In this way we do not need to write special instruction how to handle boundaries.
     */
    int       neighbors     [NUMBER_OF_DIMENSIONS * 2];

    /// MPI tries to find the best possible partition of our domain and stores that in dimension3D
    MPI_Dims_create(sizeDefaultMPICOMM, NUMBER_OF_DIMENSIONS, dimension3D);

    /// based on the partition, we create a new Cartesian topology which simplifies communication
    /**
     * The Cartesian topology is not only good writing simplified communication instructions (as we do not care what
     * happens on the boundary, if we call a send or receive function on the physical boundary, MPI will ignore our
     * request as MPI_PROC_NULL has been assigned, thus we do not need to check for physical boundaries -> simplifying
     * coding), it also offers the potential for optimising communications.
     * The fifth argument is set to true here, which allows MPI to reorder the way processors are assigned to each
     * subdomain, the idea being that processors close together in the domain, are also physically located close
     * in memory to reduce memory access times.
     */
    MPI_Cart_create(MPI_COMM_WORLD, NUMBER_OF_DIMENSIONS, dimension3D, periods3D, true, &MPI_COMM_CART);

    /// These calls will find the direct neighbors for each processors and return MPI_PROC_NULL if no neighbor is found.
    MPI_Cart_shift(MPI_COMM_CART, COORDINATE::X, 1, &neighbors[DIRECTION::LEFT  ], &neighbors[DIRECTION::RIGHT]);
    MPI_Cart_shift(MPI_COMM_CART, COORDINATE::Y, 1, &neighbors[DIRECTION::BOTTOM], &neighbors[DIRECTION::TOP  ]);
    MPI_Cart_shift(MPI_COMM_CART, COORDINATE::Z, 1, &neighbors[DIRECTION::BACK  ], &neighbors[DIRECTION::FRONT]);

    /// get the new rank and size for the Cartesian topology
    MPI_Comm_rank(MPI_COMM_CART, &rank);
    MPI_Comm_size(MPI_COMM_CART, &size);

    /// get the coordinates inside our Cartesian topology
    MPI_Cart_coords(MPI_COMM_CART, rank, NUMBER_OF_DIMENSIONS, coordinates3D);

  /// if USE_SEQUENTIAL is defined (see makefile), execute the following code
  #elif defined(USE_SEQUENTIAL)
    /// no use for the sequential program, but defining it with zero entries is allows for a unified codebase
    int coordinates3D[NUMBER_OF_DIMENSIONS] = {0, 0, 0};
    const int rank = 0;
    const int size = 1;
  #endif

  /// check that we have the right number of input arguments
  /**
   * this is the order in which we need to pass in the command line argument:
   *
   * argv[0]: name of compiled program
   * argv[1]: number of cells in the x direction
   * argv[2]: number of cells in the y direction
   * argv[3]: number of cells in the z direction
   * argv[4]: maximum number of iterations to be used by time loop
   * argv[5]: convergence criterion to be used to check if a solution has converged
   */
  if (rank == 0) {
    if (argc != 6) {
      std::cout << "Incorrect number of command line arguments specified, use the following syntax:\n" << std::endl;
      std::cout << "bin/HeatEquation3D NUM_CELLS_X NUM_CELLS_Y NUM_CELLS_Z ITER_MAX EPS" << std::endl;
      std::cout << "\nor, using MPI, use the following syntax:\n" << std::endl;
      std::cout << "mpirun -n NUM_PROCS bin/HeatEquation3D NUM_CELLS_X NUM_CELLS_Y NUM_CELLS_Z ITER_MAX EPS" << std::endl;
      std::cout << "\nSee source code for additional informations!" << std::endl;
      std::abort();
    } else {
      std::cout << "Runnung HeatEquation3D with the following arguments: " << std::endl;
      std::cout << "executable:               " << argv[0] << std::endl;
      std::cout << "number of cells in x:     " << std::stoi(argv[1]) << std::endl;
      std::cout << "number of cells in y:     " << std::stoi(argv[2]) << std::endl;
      std::cout << "number of cells in z:     " << std::stoi(argv[3]) << std::endl;
      std::cout << "max number of iterations: " << std::stoi(argv[4]) << std::endl;
      #if defined(USE_SINGLE_PRECISION)
        std::cout << "convergence threshold:    " << std::stof(argv[5]) << std::endl;
      #elif defined(USE_DOUBLE_PRECISION)
        std::cout << "convergence threshold:    " << std::stod(argv[5]) << "\n" << std::endl;
      #endif
    }
  }

  /// maximum number of iterations to perform in time loop
  const unsigned iterMax = std::stoi(argv[4]);

  /// convergence criterion, which, once met, will terminate the calculation
  /**
   * for each timestep, we calculate the difference of the current solution to the solution at the previous timestep
   * and compare it against eps. If the difference between the two solution is less than eps, we will terminate the time
   * loop and finish the calculation.
   * Broadly speaking, the lower the value of eps is, the higher the accuracy of the solution while computational time
   * will take longer. If we have not managed to reduce the difference of the previous and current solution below once
   * we have reached the maximum number of iterations (iterMax), the loop will also be terminated.
   */
  #if defined (USE_SINGLE_PRECISION)
    const floatT eps = std::stof(argv[5]);
  #elif defined (USE_DOUBLE_PRECISION)
    const floatT eps = std::stod(argv[5]);
  #endif

  /// both variables are used to calculate the convergence and normalise the result.
  /**
   * We have two normalisation factors as we have to perform a reduction first (if we use MPI) to have a globally
   * available normalisation factor
   */
  floatT globalNorm = 1.0;
  floatT norm = 1.0;

  /// the break conditions used for checking of convergence has been achieved and the simulation should be stopped.
  int breakCondition = false;
  int globalBreakCondition = false;

  /// number of points (in total, not per processor) in x, y and z.
  unsigned numCells[NUMBER_OF_DIMENSIONS];
  numCells[COORDINATE::X] = std::stoi(argv[1]);
  numCells[COORDINATE::Y] = std::stoi(argv[2]);
  numCells[COORDINATE::Z] = std::stoi(argv[3]);

  /// length of the domain in x, y and z.
  floatT domainLength[NUMBER_OF_DIMENSIONS];
  domainLength[COORDINATE::X] = 1.0;
  domainLength[COORDINATE::Y] = 1.0;
  domainLength[COORDINATE::Z] = 1.0;

  /// thermal conductivity parameter.
  /**
   * This parameter is not really of interest here. A higher value just means that heat is conducted faster, however,
   * it is also affecting the time step calculation (making the time step smaller), thus we would not expect to see any
   * change in solution or convergence rate.
   */
  const floatT alpha = 1.0;

  /// The courant fridrich levy number
  /**
   * This number is essential for convergence speed and convergence stability. A higher value will provide faster
   * convergence but will get eventually unstable. A lower value is more stable but takes longer to converge. As we are
   * solving a linear partial differential equation here, this is not really problematic, as we would typically expect
   * to see instability problems with non-linear partial differential equations. However, a theoretical limit of
   * CFL < 0.5 exist and we should not increase the value beyond this point
   */
  const floatT CFL = 0.4;

  /// the distance between cells in the x, y and z direction.
  floatT spacing[NUMBER_OF_DIMENSIONS];
  spacing[COORDINATE::X] = domainLength[COORDINATE::X] / static_cast<floatT>(numCells[COORDINATE::X] - 1.0);
  spacing[COORDINATE::Y] = domainLength[COORDINATE::Y] / static_cast<floatT>(numCells[COORDINATE::Y] - 1.0);
  spacing[COORDINATE::Z] = domainLength[COORDINATE::Z] / static_cast<floatT>(numCells[COORDINATE::Z] - 1.0);

  /// the timestep to be used in the time integration.
  const floatT dt = CFL * 1.0 / (NUMBER_OF_DIMENSIONS * 2) *
    std::pow(std::min({spacing[COORDINATE::X], spacing[COORDINATE::Y], spacing[COORDINATE::Z]}), 2.0) / alpha;

  /// thermal diffusivity strength in the x, y and z direction.
  /**
   * As we assume a constant thermal diffusivity here, we can pre-calculate its strength and apply later to the
   * equations
   */
  const floatT Dx = dt * alpha / (std::pow(spacing[COORDINATE::X], 2.0));
  const floatT Dy = dt * alpha / (std::pow(spacing[COORDINATE::Y], 2.0));
  const floatT Dz = dt * alpha / (std::pow(spacing[COORDINATE::Z], 2.0));

  /// numer of iterations taken to converge solution. will be set once simulation has converged.
  unsigned finalNumIterations = 0;

  #if defined(USE_MPI)
    /// assure that the partition given to use by MPI can be used to partition our domain in each direction
    assert((numCells[COORDINATE::X] - 1) % dimension3D[COORDINATE::X] == 0 &&
      "Can not partition data for given number of processors in x!");
    assert((numCells[COORDINATE::Y] - 1) % dimension3D[COORDINATE::Y] == 0 &&
      "Can not partition data for given number of processors in y!");
    assert((numCells[COORDINATE::Z] - 1) % dimension3D[COORDINATE::Z] == 0 &&
      "Can not partition data for given number of processors in z!");

    /// chunck contains the number of cells in the x, y and z direction for each sub domain.
    const unsigned chunck[NUMBER_OF_DIMENSIONS] = {
      ((numCells[COORDINATE::X] - 1) / dimension3D[COORDINATE::X]) + 1,
      ((numCells[COORDINATE::Y] - 1) / dimension3D[COORDINATE::Y]) + 1,
      ((numCells[COORDINATE::Z] - 1) / dimension3D[COORDINATE::Z]) + 1
    };
  #elif defined(USE_SEQUENTIAL)
    /// chunck contains the number of cells in the x, y and z direction for the whole domain.
    const unsigned chunck[NUMBER_OF_DIMENSIONS] = {numCells[COORDINATE::X], numCells[COORDINATE::Y],
      numCells[COORDINATE::Z]};
  #endif

  /// Create a solution vector
  /**
   * we abbreviate the solution vector by T for Temperature (which is what we are solving for). T contains the current
   * solution while T0 contains the solution fromt he previous time step.
   */
  std::vector<std::vector<std::vector<floatT>>> T, T0;

  /// resize both T and T0 for each sub-domain
  T.resize(chunck[COORDINATE::X]);
  T0.resize(chunck[COORDINATE::X]);
  for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i) {
    T[i].resize(chunck[COORDINATE::Y]);
    T0[i].resize(chunck[COORDINATE::Y]);
    for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j) {
      T[i][j].resize(chunck[COORDINATE::Z]);
      T0[i][j].resize(chunck[COORDINATE::Z]);
    }
  }

  /// initialise each solution vector on each sub-domain with zero everywhere
  for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
    for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        T[i][j][k] = 0.0;

  /// apply boundary conditions on the top of the domain
  /**
   * If we are using MPI, we further need to check that neighbors[DIRECTION::TOP] == MPI_PROC_NULL, which, if true,
   * indicates that the current sub-domain has no neighbors to the top, thus it must be the top-most sub-domain and
   * thus have a physical boundary. If this is the case, we impose our boundary condition with 1 everywhere.
   */
  #if defined(USE_MPI)
    if (neighbors[DIRECTION::TOP] == MPI_PROC_NULL)
  #endif
    for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        T[i][chunck[COORDINATE::Y]-1][k] = 1.0;

  /// apply boundary conditions on the left-side of the domain
  /**
   * this boundary condition may seem odd at first, basically what we are doing is applying the y coordinate as the
   * boundary condition, which will result in a linear distribution between 0 and 1 for the values on the left
   * boundary face. Why do we do that? If we have a linear profile, starting with 0 on the bottom and finishing with 1
   * at the top, and furthermore apply 1 everywhere ont he top and 0 everywhere on the bottom, then the nature of the
   * partial differntial equation is that we would expect the same linear distribution throughout the domain once the
   * solution has converged in time. If we calculate the same solution everywhere in the internal domain as we specified
   * on the boundaries, then we have constructed a sort of analytic solution against which we can compare our final
   * result against. In this case, we simply have established that: T(x, y, z) = f(y) = y. We use that fact later to
   * calculate the error throughout the domain, which we output to the screen as a quick check if the solution is
   * converging towards the right solution (useful for debugging, rather than having to open the file each time with
   * paraview or a similar program).
   */
  #if defined(USE_MPI)
    if (neighbors[DIRECTION::LEFT] == MPI_PROC_NULL)
  #endif
    for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        T[0][j][k] = (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];

  /// apply boundary conditions on the right-side of the domain
  #if defined(USE_MPI)
    if (neighbors[DIRECTION::RIGHT] == MPI_PROC_NULL)
  #endif
    for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        T[chunck[COORDINATE::X] - 1][j][k] = (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];

  /// apply boundary conditions on the back-side of the domain
  #if defined(USE_MPI)
    if (neighbors[DIRECTION::BACK] == MPI_PROC_NULL)
  #endif
    for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
      for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
        T[i][j][0] = (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];

  /// apply boundary conditions on the front-side of the domain
  #if defined(USE_MPI)
    if (neighbors[DIRECTION::FRONT] == MPI_PROC_NULL)
  #endif
    for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
      for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
        T[i][j][chunck[COORDINATE::Z] - 1] = (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];

  /// if we use MPI, make sure that our send and recieve buffers are correctly allocated
  #if defined(USE_MPI)

    /// allocate storage for left-side send- and recievebuffer
    if (neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::LEFT].resize((chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] -1));
      receiveBuffer[DIRECTION::LEFT].resize((chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] -1));
    } else {
      sendBuffer[DIRECTION::LEFT].resize(1);
      receiveBuffer[DIRECTION::LEFT].resize(1);
    }

    /// allocate storage for right-side send- and recievebuffer
    if (neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::RIGHT].resize((chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] -1));
      receiveBuffer[DIRECTION::RIGHT].resize((chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] -1));
    } else {
      sendBuffer[DIRECTION::RIGHT].resize(1);
      receiveBuffer[DIRECTION::RIGHT].resize(1);
    }

    /// allocate storage for bottom-side send- and recievebuffer
    if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::BOTTOM].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] -1));
      receiveBuffer[DIRECTION::BOTTOM].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] -1));
    } else {
      sendBuffer[DIRECTION::BOTTOM].resize(1);
      receiveBuffer[DIRECTION::BOTTOM].resize(1);
    }

    /// allocate storage for top-side send- and recievebuffer
    if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::TOP].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] -1));
      receiveBuffer[DIRECTION::TOP].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] -1));
    } else {
      sendBuffer[DIRECTION::TOP].resize(1);
      receiveBuffer[DIRECTION::TOP].resize(1);

    }

    /// allocate storage for back-side send- and recievebuffer
    if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::BACK].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] -1));
      receiveBuffer[DIRECTION::BACK].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] -1));
    } else {
      sendBuffer[DIRECTION::BACK].resize(1);
      receiveBuffer[DIRECTION::BACK].resize(1);
    }

    /// allocate storage for front-side send- and recievebuffer
    if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL) {
      sendBuffer[DIRECTION::FRONT].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] -1));
      receiveBuffer[DIRECTION::FRONT].resize((chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] -1));
    } else {
      sendBuffer[DIRECTION::FRONT].resize(1);
      receiveBuffer[DIRECTION::FRONT].resize(1);
    }

    /// start timing (we don't want any setup time to be included, thus we start it just before the time loop)
    auto start = MPI_Wtime();
  #elif defined(USE_SEQUENTIAL)
    /// unlike MPI, we use the chrono class foe a high accuracy stop watch
    auto start = std::chrono::system_clock::now();
  #endif

  /// main time loop
  /**
   * this is where we solve the actual partial differential equation and do the communication among processors.
   */
  for (unsigned time = 0; time < iterMax; ++time)
  {
    /// copy the solution from the previous timestep into T, which holds the solution of the last iteration
    for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
      for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
        for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
          T0[i][j][k] = T[i][j][k];

    // HALO communication step
    /**
     * As we want to overlap communication with computation, we initiate the setup of the HALO exchange before we do
     * any commputation. We use non-blocking sends here (MPI_Isend(...)) so that once we have send of our information,
     * each processor can start computing results for which we already have data (i.e. on the internal domain).
     */
    #if defined(USE_MPI)

      /// preparing the send buffer (the data we want to send to the left neighbor), if a neighbor exists
      /**
       * for simplicity, we write the 2D array (the face on the boundary) into a 1D array which we can easily send.
       * It is important that once we receive the it we are aware that the array containing the data is 1D now.
       */
      unsigned counter = 0;
      if (neighbors[DIRECTION::LEFT] != MPI_PROC_NULL)
        for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            sendBuffer[DIRECTION::LEFT][counter++] = T0[1][j][k];

      /// preparing the send buffer (the data we want to send to the right neighbor), if a neighbor exists
      counter = 0;
      if (neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL)
        for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            sendBuffer[DIRECTION::RIGHT][counter++] = T0[chunck[COORDINATE::X] - 2][j][k];

      /// preparing the send buffer (the data we want to send to the bottom neighbor), if a neighbor exists
      counter = 0;
      if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            sendBuffer[DIRECTION::BOTTOM][counter++] = T0[i][1][k];

      /// preparing the send buffer (the data we want to send to the top neighbor), if a neighbor exists
      counter = 0;
      if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            sendBuffer[DIRECTION::TOP][counter++] = T0[i][chunck[COORDINATE::Y] - 2][k];

      /// preparing the send buffer (the data we want to send to the back neighbor), if a neighbor exists
      counter = 0;
      if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            sendBuffer[DIRECTION::BACK][counter++] = T0[i][j][1];

      /// preparing the send buffer (the data we want to send to the front neighbor), if a neighbor exists
      counter = 0;
      if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            sendBuffer[DIRECTION::FRONT][counter++] = T0[i][j][chunck[COORDINATE::Z] - 2];

      /// prepare the tags we need to append to the send message for each send (in each direction) and receive
      for (unsigned index = 0; index < NUMBER_OF_DIMENSIONS * 2; ++index) {
        tagSend[index]    = 100 + neighbors[index];
        tagReceive[index] = 100 + rank;
      }

      /// send the prepared send buffer to the neighbors using non-blocking MPI_Isend(...)
      MPI_Isend(&sendBuffer[DIRECTION::LEFT][0], (chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::LEFT], tagSend[DIRECTION::LEFT  ], MPI_COMM_CART,
        &request[DIRECTION::LEFT]);

      MPI_Isend(&sendBuffer[DIRECTION::RIGHT][0], (chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::RIGHT], tagSend[DIRECTION::RIGHT ], MPI_COMM_CART,
        &request[DIRECTION::RIGHT]);

      MPI_Isend(&sendBuffer[DIRECTION::BOTTOM][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::BOTTOM], tagSend[DIRECTION::BOTTOM], MPI_COMM_CART,
        &request[DIRECTION::BOTTOM]);

      MPI_Isend(&sendBuffer[DIRECTION::TOP][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::TOP], tagSend[DIRECTION::TOP   ], MPI_COMM_CART,
        &request[DIRECTION::TOP]);

      MPI_Isend(&sendBuffer[DIRECTION::BACK][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::BACK], tagSend[DIRECTION::BACK  ], MPI_COMM_CART,
        &request[DIRECTION::BACK]);

      MPI_Isend(&sendBuffer[DIRECTION::FRONT][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::FRONT], tagSend[DIRECTION::FRONT ], MPI_COMM_CART,
        &request[DIRECTION::FRONT]);

    #endif

    // compute internal domain (no halos required)
    /**
     * here we compute the solution of the partial differntial heat equation ont he inside of the domain. as we never
     * touch the boundaries, we do not need any halo data for the exchange. Ideally, we hope that the communication
     * is happening in the background while this computation is executed so that once the computation is done, we can
     * access the received data without having to wait for hit, thus hidding communication overhead.
     */
    for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
      for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
        for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k) {
          T[i][j][k] = T0[i][j][k] +
            Dx * (T0[i+1][j][k] - 2.0*T0[i][j][k] + T0[i-1][j][k]) +
            Dy * (T0[i][j+1][k] - 2.0*T0[i][j][k] + T0[i][j-1][k]) +
            Dz * (T0[i][j][k+1] - 2.0*T0[i][j][k] + T0[i][j][k-1]);
        }

    /// now work on the halo cells
    #if defined(USE_MPI)
      /// receive the halo information from each neighbor, if exists.
      /**
       * similar to the send above, we receive data from each neighbor. If the neighbor does not exists, i.e. if its
       * rank is equal to MPI_PROC_NULL, this function call is a no-op and simply ignored.
       */
      MPI_Recv(&receiveBuffer[DIRECTION::LEFT][0], (chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::LEFT], tagReceive[DIRECTION::LEFT  ], MPI_COMM_CART,
        &status[DIRECTION::LEFT]);

      MPI_Recv(&receiveBuffer[DIRECTION::RIGHT][0], (chunck[COORDINATE::Y] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::RIGHT], tagReceive[DIRECTION::RIGHT ], MPI_COMM_CART,
        &status[DIRECTION::RIGHT]);

      MPI_Recv(&receiveBuffer[DIRECTION::BOTTOM][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::BOTTOM], tagReceive[DIRECTION::BOTTOM], MPI_COMM_CART,
        &status[DIRECTION::BOTTOM]);

      MPI_Recv(&receiveBuffer[DIRECTION::TOP][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Z] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::TOP], tagReceive[DIRECTION::TOP   ], MPI_COMM_CART,
        &status[DIRECTION::TOP]);

      MPI_Recv(&receiveBuffer[DIRECTION::BACK][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::BACK], tagReceive[DIRECTION::BACK  ], MPI_COMM_CART,
        &status[DIRECTION::BACK]);

      MPI_Recv(&receiveBuffer[DIRECTION::FRONT][0], (chunck[COORDINATE::X] - 1) * (chunck[COORDINATE::Y] - 1),
        MPI_FLOAT_T, neighbors[DIRECTION::FRONT], tagReceive[DIRECTION::FRONT ], MPI_COMM_CART,
        &status[DIRECTION::FRONT]);

      /// make sure that all communications have been executed
      /**
       * even though we use a blocking receive here, since we used a non-blocking send, we have to wait for all
       * communications to have finished before continuing.
       */
      MPI_Waitall(NUMBER_OF_DIMENSIONS * 2, request, status);

      /// now that we have the halo cells, we update the boundaries using information from other processors
      /**
       * we do that for each potential boundary by checking if a neighbor exists. If so, that means that we should have
       * data available from the above send/recieve operation so we can operate on it.
       *
       * First we create a reference to the actual received data, called THalo, which holds the temperature information
       * in the halo cells. We set the i, j or k index appropriately, so we can still loop through our solution
       * container as if we were looping over all three dimensions.
       * We use a counter variable which we increase each time we have accessed the halo information, as it is a 1D
       * array. Note that the order of the loop here matters, which is the same order we used to write the halo data.
       * As it is the same order, we can directly access each element in the halo container in sequence.
       * Then we just calculate the solution using the appropriate halo data where necessary.
       */
      if (neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::LEFT];
        unsigned i = 0;
        unsigned counter = 0;

        for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (T0[i+1][j][k] - 2.0*T0[i][j][k] + THalo[counter++]) +
              Dy * (T0[i][j+1][k] - 2.0*T0[i][j][k] + T0[i][j-1][k]   ) +
              Dz * (T0[i][j][k+1] - 2.0*T0[i][j][k] + T0[i][j][k-1]   );
          }
      }

      /// do the same as above, this time for the right neighbor halo data
      if (neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::RIGHT];
        unsigned i = chunck[COORDINATE::X] - 1;
        unsigned counter = 0;

        for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (THalo[counter++] - 2.0*T0[i][j][k] + T0[i-1][j][k]) +
              Dy * (T0[i][j+1][k]    - 2.0*T0[i][j][k] + T0[i][j-1][k]) +
              Dz * (T0[i][j][k+1]    - 2.0*T0[i][j][k] + T0[i][j][k-1]);
          }
      }

      /// do the same as above, this time for the bottom neighbor halo data
      if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::BOTTOM];
        unsigned j = 0;
        unsigned counter = 0;

        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (T0[i+1][j][k] - 2.0*T0[i][j][k] + T0[i-1][j][k]   ) +
              Dy * (T0[i][j+1][k] - 2.0*T0[i][j][k] + THalo[counter++]) +
              Dz * (T0[i][j][k+1] - 2.0*T0[i][j][k] + T0[i][j][k-1]   );
          }
      }

      /// do the same as above, this time for the top neighbor halo data
      if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::TOP];
        unsigned j = chunck[COORDINATE::Y] - 1;
        unsigned counter = 0;

        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (T0[i+1][j][k]    - 2.0*T0[i][j][k] + T0[i-1][j][k]) +
              Dy * (THalo[counter++] - 2.0*T0[i][j][k] + T0[i][j-1][k]) +
              Dz * (T0[i][j][k+1]    - 2.0*T0[i][j][k] + T0[i][j][k-1]);
          }
      }

      /// do the same as above, this time for the back neighbor halo data
      if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::BACK];
        unsigned k = 0;
        unsigned counter = 0;

        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (T0[i+1][j][k] - 2.0*T0[i][j][k] + T0[i-1][j][k]   ) +
              Dy * (T0[i][j+1][k] - 2.0*T0[i][j][k] + T0[i][j-1][k]   ) +
              Dz * (T0[i][j][k+1] - 2.0*T0[i][j][k] + THalo[counter++]);
          }
      }

      /// do the same as above, this time for the front neighbor halo data
      if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL) {
        const auto &THalo = receiveBuffer[DIRECTION::FRONT];
        unsigned k = chunck[COORDINATE::Z] - 1;
        unsigned counter = 0;

        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j) {
            T[i][j][k] = T0[i][j][k] +
              Dx * (T0[i+1][j][k]    - 2.0*T0[i][j][k] + T0[i-1][j][k]) +
              Dy * (T0[i][j+1][k]    - 2.0*T0[i][j][k] + T0[i][j-1][k]) +
              Dz * (THalo[counter++] - 2.0*T0[i][j][k] + T0[i][j][k-1]);
          }
      }

      /// update edges of halo elements
      /**
       * So far, we have only worked on the internal faces of the halo elements. Consider the sub-domain again, as
       * presented aboce:
       *
       *       Y
       *       |
       *        _________________
       *       /.               /|
       *      / .      3       / |
       *     /_________|______/  |
       *    |   .      |4     |  |
       *    | 0--------/------|-1|
       *    |   ......5|......|..|    -- X
       *    |  /       |      |  /
       *    | /        2      | /
       *    |/________________|/
       *
       *   /
       *  Z
       *
       * For the sake of argument, lets assume we are working on the back boundary face (here with index 4), then the 2D
       * representation of that face would look like this:
       *
       * Y
       * |
       *  _ _ _ _ _
       * |_|_|_|_|_|
       * |_|_|_|_|_|
       * |_|_|_|_|_|
       * |_|_|_|_|_|
       * |_|_|_|_|_| __ X
       *
       * Effectively what we have done so far above is working on the internal faces, which can be visualised like so:
       *
       * Y
       * |
       *  _ _ _ _ _
       * |_|_|_|_|_|
       * |_|#|#|#|_|
       * |_|#|#|#|_|
       * |_|#|#|#|_|
       * |_|_|_|_|_| __ X
       *
       * We have not, however, worked ont he edges and corner points, which we do next. The information needed on these
       * edges and corners can, of course, also be obtained from MPI, however, it is cumbersome to index these points
       * in the 1D send/receive buffer representation. Without loosing any accuracy, we can simply extrapolate from the
       * internal domain to the boundary edges to get our values on the edges and then on the corners.
       *
       * For the edges we use a second-order extrapolation. Consider the following 1D line representation:
       *
       * T[i]    T[i+1]  T[i+2]
       * |_______|_______|________... X
       *
       * Assuming equal distance between all three entries, we can write that (T[i] + T[i+2]) / 2 = T[i+1]
       * Solving for T[i] gives: T[i] = 2 * T[i+1] - T[i+2]
       * The T[i]'s are the values on the boundary (which we do not have at the moment) which we can thus extrapolate
       * from the inside of the domain (using T[i+1] and T[i+2], which we have from the internal of the subdomain).
       *
       * Note that this is an approximation, it would be better to calculate the equation at thos points rather than
       * extrapolating the values. However, the very nature of the heat equation is very similar to the extrapolation
       * nature and thus there is very little difference between the two approaches while we can keep the complexity
       * of the MPI communication to a minimum. This is good for performance and readibility of the code.
       */
      if (neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) {
        if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
          unsigned i = 0;
          unsigned j = 0;
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            T[i][j][k] = 2.0 * T[i+1][j][k] - T[i+2][j][k];
        }
        if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
          unsigned i = 0;
          unsigned j = chunck[COORDINATE::Y] - 1;
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            T[i][j][k] = 2.0 * T[i+1][j][k] - T[i+2][j][k];
        }
        if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL) {
          unsigned i = 0;
          unsigned k = 0;
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            T[i][j][k] = 2.0 * T[i+1][j][k] - T[i+2][j][k];
        }
        if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL) {
          unsigned i = 0;
          unsigned k = chunck[COORDINATE::Z] - 1;
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            T[i][j][k] = 2.0 * T[i+1][j][k] - T[i+2][j][k];
        }
      }

      if (neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) {
        if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
          unsigned i = chunck[COORDINATE::X] - 1;
          unsigned j = 0;
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            T[i][j][k] = 2.0 * T[i-1][j][k] - T[i-2][j][k];
        }
        if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
          unsigned i = chunck[COORDINATE::X] - 1;
          unsigned j = chunck[COORDINATE::Y] - 1;
          for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
            T[i][j][k] = 2.0 * T[i-1][j][k] - T[i-2][j][k];
        }
        if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL) {
          unsigned i = chunck[COORDINATE::X] - 1;
          unsigned k = 0;
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            T[i][j][k] = 2.0 * T[i-1][j][k] - T[i-2][j][k];
        }
        if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL) {
          unsigned i = chunck[COORDINATE::X] - 1;
          unsigned k = chunck[COORDINATE::Z] - 1;
          for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
            T[i][j][k] = 2.0 * T[i-1][j][k] - T[i-2][j][k];
        }
      }

      if (neighbors[DIRECTION::BACK] != MPI_PROC_NULL) {
        if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
          unsigned j = 0;
          unsigned k = 0;
          for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
            T[i][j][k] = 2.0 * T[i][j][k+1] - T[i][j][k+2];
        }
        if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
          unsigned j = chunck[COORDINATE::Y] - 1;
          unsigned k = 0;
          for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
            T[i][j][k] = 2.0 * T[i][j][k+1] - T[i][j][k+2];
        }
      }

      if (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL) {
        if (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) {
          unsigned j = 0;
          unsigned k = chunck[COORDINATE::Z] - 1;
          for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
            T[i][j][k] = 2.0 * T[i][j][k-1] - T[i][j][k-2];
        }
        if (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) {
          unsigned j = chunck[COORDINATE::Y] - 1;
          unsigned k = chunck[COORDINATE::Z] - 1;
          for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
            T[i][j][k] = 2.0 * T[i][j][k-1] - T[i][j][k-2];
        }
      }
      /// finished with halo edges extrapolation

      /// at last, we update the boundary points through weighted averages
      /**
       * Consider a corner point, which is on the inside of the domain. This can be depicted like so:
       *
       *            Y
       *            |
       * T[0][1][0] *
       *            |
       *            |      T[1][0][0]
       * T[0][0][0] *------*--- X
       *           /
       *          /
       *         * T[0][0][1]
       *        /
       *       Z
       *
       * The corner point T[0][0][0] is simply obtained through a weighted average of its three immediate neighbor
       * points:
       *
       * T[0][0][0] = 1.0 / 3.0 * (T[1][0][0] + T[0][1][0] + T[0][0][1])
       *
       * This is again a sort of extrapolation and similar to the behavior of the partial differential equation. To
       * illustrate that, check the results after they have been calculated and you will see a linear temperature
       * gradient from the top to the bottom. Considering that we have initialised the top boundary with a value of 1
       * and the bottom boundary with a value of 0, this linear gradient behaviour shows that the partial differential
       * heat equation simply averages between two states. Thus, extrapolating and taking weighted averages is very
       * similar, as we are also trying to average out any differences among neighboring points.
       */
      if ((neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) && (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::BACK] != MPI_PROC_NULL)) {
        unsigned i = 0;
        unsigned j = 0;
        unsigned k = 0;
        T[i][j][k] = 1.0 / 3.0 * (T[i+1][j][k] + T[i][j+1][k] + T[i][j][k+1]);
      }

      if ((neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) && (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL)) {
        unsigned i = 0;
        unsigned j = 0;
        unsigned k = chunck[COORDINATE::Z] - 1;
        T[i][j][k] = 1.0 / 3.0 * (T[i+1][j][k] + T[i][j+1][k] + T[i][j][k-1]);
      }

      if ((neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) && (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::BACK] != MPI_PROC_NULL)) {
        unsigned i = 0;
        unsigned j = chunck[COORDINATE::Y] - 1;
        unsigned k = 0;
        T[i][j][k] = 1.0 / 3.0 * (T[i+1][j][k] + T[i][j-1][k] + T[i][j][k+1]);
      }

      if ((neighbors[DIRECTION::LEFT] != MPI_PROC_NULL) && (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL)) {
        unsigned i = 0;
        unsigned j = chunck[COORDINATE::Y] - 1;
        unsigned k = chunck[COORDINATE::Z] - 1;
        T[i][j][k] = 1.0 / 3.0 * (T[i+1][j][k] + T[i][j-1][k] + T[i][j][k-1]);
      }

      if ((neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) && (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::BACK] != MPI_PROC_NULL)) {
        unsigned i = chunck[COORDINATE::X] - 1;
        unsigned j = 0;
        unsigned k = 0;
        T[i][j][k] = 1.0 / 3.0 * (T[i-1][j][k] + T[i][j+1][k] + T[i][j][k+1]);
      }

      if ((neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) && (neighbors[DIRECTION::BOTTOM] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL)) {
        unsigned i = chunck[COORDINATE::X] - 1;
        unsigned j = 0;
        unsigned k = chunck[COORDINATE::Z] - 1;
        T[i][j][k] = 1.0 / 3.0 * (T[i-1][j][k] + T[i][j+1][k] + T[i][j][k-1]);
      }

      if ((neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) && (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::BACK] != MPI_PROC_NULL)) {
        unsigned i = chunck[COORDINATE::X] - 1;
        unsigned j = chunck[COORDINATE::Y] - 1;
        unsigned k = 0;
        T[i][j][k] = 1.0 / 3.0 * (T[i-1][j][k] + T[i][j-1][k] + T[i][j][k+1]);
      }

      if ((neighbors[DIRECTION::RIGHT] != MPI_PROC_NULL) && (neighbors[DIRECTION::TOP] != MPI_PROC_NULL) &&
        (neighbors[DIRECTION::FRONT] != MPI_PROC_NULL)) {
        unsigned i = chunck[COORDINATE::X] - 1;
        unsigned j = chunck[COORDINATE::Y] - 1;
        unsigned k = chunck[COORDINATE::Z] - 1;
        T[i][j][k] = 1.0 / 3.0 * (T[i-1][j][k] + T[i][j-1][k] + T[i][j][k-1]);
      }
      /// finished with halo corner points
    #endif

    /// calculate the difference between the current and previous (last time step) solution.
    /**
     * here we consider the absolute error, or L_infinity norm. We check if the error is larger between the current
     * and previous solution for each point in the domain (except for boundaries). We store the largest error and use
     * that as and indication if the solution has converged to a final solution. We can define how low we want the
     * solution to reduce in error by changing the "eps" variable.
     *
     * The difference is called residual.
     */
    floatT res = std::numeric_limits<floatT>::min();
    for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
      for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
        for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
          if (std::fabs(T[i][j][k] - T0[i][j][k]) > res)
            res = std::fabs(T[i][j][k] - T0[i][j][k]);

    /// if it is the first time step, store the residual as the normalisation factor
    /**
     * doing this allows us to normalise all of our residuals with respect to the first residual. The effect is that,
     * our first residual will always be 1 and so if we set a convergence threshold of say 1e-5, we are garuanteed that
     * the simulation only stops if we have reached a drop in convergence by 5 orders of magnitutde.
     */
    if (time == 0)
      if (res != 0.0)
        norm = res;

    /// For MPI, we have to communicate the norm by selecting the lowest among all processors
    #if defined(USE_MPI)
      if (time == 0) {
        MPI_Iallreduce(&norm, &globalNorm, 1, MPI_FLOAT_T, MPI_MIN, MPI_COMM_CART, &reduceRequest);
        MPI_Wait(&reduceRequest, MPI_STATUS_IGNORE);
      }
    #endif

    /// if we want to debug, it may be useful to see the residuals. Turned of for release builds for performance.
    #if defined(USE_DEBUG)
      if (rank == 0) {
        std::cout << "time: " << std::setw(10) << time;
        std::cout << std::scientific << std::setw(15) << std::setprecision(5) << ", residual: ";
        std::cout << res / norm << std::endl;
      }
    #endif

    /// check if the current residual has dropped below our defined convergence threshold "eps"
    if (res / norm < eps)
      breakCondition = true;

    /// Again, for MPI we need to among all processors if we can break from the loop
    /**
     * if we were not to do this here, we are almost garuanteed to have a deadlock. Say one processor finishes and
     * breaks from the loop, it will no longer be receiving information during the next halo exchange, thus MPI_Waitall
     * will never finish. Therefore, once one processor has found a converged solution, all of them will break from
     * the loop. We could also use MPI_MIN here, in this case, only if all processors have found convergence we break
     * from the loop (as breakCondition will be true for all processors and true = 1 as an int representation, MPI_MIN
     * will ensure that only once all processors set the condition to true can we break from the loop).
     */
    #if defined(USE_MPI)
      MPI_Iallreduce(&breakCondition, &globalBreakCondition, 1, MPI_INT, MPI_MAX, MPI_COMM_CART, &reduceRequest);
      MPI_Wait(&reduceRequest, MPI_STATUS_IGNORE);
    #elif defined(USE_SEQUENTIAL)
      globalBreakCondition = breakCondition;
    #endif

    /// final check if we can break, the above was just preparation for this check.
    /**
     * set finalNumIterations to the final number of iterations so we can print them to screen after we have left
     * the current scope (after which the time variable will no longer be available).
     */
    if (globalBreakCondition) {
      finalNumIterations = time;
      break;
    }
  }
  /// done with the time loop

  /// output the timing information to screen.
  #if defined(USE_MPI)
    auto end = MPI_Wtime();
    if (rank == 0) {
      std::cout << "Computational time (parallel): " << std::fixed << (end - start) << "\n" << std::endl;
      if (globalBreakCondition) {
        std::cout << "Simulation has converged in " << finalNumIterations << " iterations";
        std::cout << " with a convergence threshold of " << std::scientific << eps << std::endl;
      } else
        std::cout << "Simulation did not converge within " << iterMax << " iterations." << std::endl;
    }
  #elif defined(USE_SEQUENTIAL)
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<floatT> diff = end - start;
    std::cout << "Computational time (serial): " << std::fixed << diff.count() << "\n" << std::endl;
    if (globalBreakCondition) {
      std::cout << "Simulation has converged in " << finalNumIterations << " iterations";
      std::cout << " with a convergence threshold of " << std::scientific << eps << std::endl;
    } else
      std::cout << "Simulation did not converge within " << iterMax << " iterations." << std::endl;
  #endif

  /// calculate the error we have made against the analytic solution
  #if defined(USE_MPI)
    double globalError = 0.0;
    double error = 0.0;
    for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
      for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          error += std::sqrt( std::pow( T[i][j][k] - (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y], 2.0) );
    error /= ( (chunck[COORDINATE::X] - 2) * (chunck[COORDINATE::Y] - 2) * (chunck[COORDINATE::Z] - 2) );
    MPI_Iallreduce(&error, &globalError, 1, MPI_FLOAT_T, MPI_SUM, MPI_COMM_CART, &reduceRequest);
    MPI_Wait(&reduceRequest, MPI_STATUS_IGNORE);
    if (rank == 0)
      std::cout << "L2-norm error: " << std::fixed << std::setprecision(4) << 100 * error << " %" << std::endl;
  #elif defined(USE_SEQUENTIAL)
    double error = 0.0;
    for (unsigned k = 1; k < chunck[COORDINATE::Z] - 1; ++k)
      for (unsigned j = 1; j < chunck[COORDINATE::Y] - 1; ++j)
        for (unsigned i = 1; i < chunck[COORDINATE::X] - 1; ++i)
          error += std::sqrt( std::pow( T[i][j][k] - (spacing[COORDINATE::Y] * j), 2.0) );
    error /= ( (chunck[COORDINATE::X] - 2) * (chunck[COORDINATE::Y] - 2) * (chunck[COORDINATE::Z] - 2) );
    std::cout << "L2-norm error: " << std::fixed << std::setprecision(4) << 100 * error << " %" << std::endl;

  #endif

  /// output the solution in a format readable by a post processor, such as paraview.
  #if defined(USE_MPI)
    std::vector<floatT> receiveBufferPostProcess;
    receiveBufferPostProcess.resize(chunck[COORDINATE::X] * chunck[COORDINATE::Y] * chunck[COORDINATE::Z]);
    if (rank > 0 && size != 1)
    {
      int counter = 0;
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
          for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
            receiveBufferPostProcess[counter++] = T[i][j][k];

      MPI_Send(&receiveBufferPostProcess[0], chunck[COORDINATE::X] * chunck[COORDINATE::Y] * chunck[COORDINATE::Z], MPI_FLOAT_T, 0, 200 + rank, MPI_COMM_CART);
      MPI_Send(&coordinates3D[0], NUMBER_OF_DIMENSIONS, MPI_INT, 0, 300 + rank, MPI_COMM_CART);
    }
    if (rank == 0 && size != 1)
    {
      std::ofstream out("output/out.dat");
      out << "TITLE=\"out\"" << std::endl;
      out << "VARIABLES = \"X\", \"Y\", \"Z\", \"T\", \"rank\"" << std::endl;
      out << "ZONE T = \"" << rank << "\", I=" << chunck[COORDINATE::X] << ", J=" << chunck[COORDINATE::Y] << ", K=" << chunck[COORDINATE::Z] << ", F=POINT" << std::endl;
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
          for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
          {
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::X] * (chunck[COORDINATE::X] - 1) + i) * spacing[COORDINATE::X];
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::Z] * (chunck[COORDINATE::Z] - 1) + k) * spacing[COORDINATE::Z];
            out << std::scientific << std::setprecision(5) << std::setw(15) << T[i][j][k];
            out << std::fixed << std::setw(5) << rank << std::endl;
          }

      for (int recvRank = 1; recvRank < size; ++recvRank)
      {
        int coordinates3DFromReceivedRank[NUMBER_OF_DIMENSIONS];
        MPI_Recv(&receiveBufferPostProcess[0], chunck[COORDINATE::X] * chunck[COORDINATE::Y] * chunck[COORDINATE::Z], MPI_FLOAT_T, recvRank, 200 + recvRank, MPI_COMM_CART, &postStatus[0]);
        MPI_Recv(&coordinates3DFromReceivedRank[0], NUMBER_OF_DIMENSIONS, MPI_INT, recvRank, 300 + recvRank, MPI_COMM_CART, &postStatus[1]);

        out << "ZONE T = \"" << rank << "\", I=" << chunck[COORDINATE::X] << ", J=" << chunck[COORDINATE::Y] << ", K=" << chunck[COORDINATE::Z] << ", F=POINT" << std::endl;
        int counter = 0;
        for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
          for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
            for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
            {
              out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3DFromReceivedRank[COORDINATE::X] * (chunck[COORDINATE::X] - 1) + i) * spacing[COORDINATE::X];
              out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3DFromReceivedRank[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];
              out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3DFromReceivedRank[COORDINATE::Z] * (chunck[COORDINATE::Z] - 1) + k) * spacing[COORDINATE::Z];
              out << std::scientific << std::setprecision(5) << std::setw(15) << receiveBufferPostProcess[counter++];
              out << std::fixed << std::setw(5) << recvRank << std::endl;
            }
      }
      out.close();
    }
    if (size == 1)
    {
      std::ofstream out("output/out.dat");
      out << "TITLE=\"out\"" << std::endl;
      out << "VARIABLES = \"X\", \"Y\", \"Z\", \"T\"" << std::endl;
      out << "ZONE T = \"" << rank << "\", I=" << chunck[COORDINATE::X] << ", J=" << chunck[COORDINATE::Y] << ", K=" << chunck[COORDINATE::Z] << ", F=POINT" << std::endl;
      for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
        for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
          for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
          {
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::X] * (chunck[COORDINATE::X] - 1) + i) * spacing[COORDINATE::X];
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::Y] * (chunck[COORDINATE::Y] - 1) + j) * spacing[COORDINATE::Y];
            out << std::scientific << std::setprecision(5) << std::setw(15) << (coordinates3D[COORDINATE::Z] * (chunck[COORDINATE::Z] - 1) + k) * spacing[COORDINATE::Z];
            out << std::scientific << std::setprecision(5) << std::setw(15) << T[i][j][k] << std::endl;
          }
      out.close();
    }
  #elif defined(USE_SEQUENTIAL)
    std::ofstream out("output/out.dat");
    out << "TITLE=\"out\"" << std::endl;
    out << "VARIABLES = \"X\", \"Y\", \"Z\", \"T\"" << std::endl;
    out << "ZONE T = \"" << rank << "\", I=" << chunck[COORDINATE::X] << ", J=" << chunck[COORDINATE::X] << ", K=" << chunck[COORDINATE::Z] << ", F=POINT" << std::endl;
    for (unsigned k = 0; k < chunck[COORDINATE::Z]; ++k)
      for (unsigned j = 0; j < chunck[COORDINATE::Y]; ++j)
        for (unsigned i = 0; i < chunck[COORDINATE::X]; ++i)
        {
          out << std::scientific << std::setprecision(5) << std::setw(15) << spacing[COORDINATE::X] * ((chunck[COORDINATE::X] - 1) * rank + i);
          out << std::scientific << std::setprecision(5) << std::setw(15) << spacing[COORDINATE::Y] * ((chunck[COORDINATE::Y] - 1) * rank + j);
          out << std::scientific << std::setprecision(5) << std::setw(15) << spacing[COORDINATE::Z] * ((chunck[COORDINATE::Z] - 1) * rank + k);
          out << std::scientific << std::setprecision(5) << std::setw(15) << T[i][j][k] << std::endl;
        }
    out.close();
  #endif

  #if defined(USE_MPI)
    MPI_Finalize();
  #endif
  return 0;
}