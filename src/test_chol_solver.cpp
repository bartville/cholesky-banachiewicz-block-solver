#include <Eigen/Core>
#include <Eigen/Geometry>
#include "chol_solver.h"

#include <iostream>
#include <chrono>

using namespace bo_mary;

template<typename MatrixBlock>
void copyMatrixValues(BoMatrix<MatrixBlock>& dest,
                      const Eigen::MatrixXf& src,
                      const int size,
                      const int block_size) {
  for(int r = 0; r < size; ++r)
    for(int c = 0; c < size; ++c)
      dest.at(r,c) = src.block(r*block_size, c*block_size, block_size, block_size);
}


template<typename VectorBlock>
void copyVectorValues(BoVector<VectorBlock>& dest,
                      const Eigen::MatrixXf& src,
                      const int size,
                      const int block_size) {
  for(int r = 0; r < size; ++r)
      dest.at(r) = src.block(r*block_size, 0, block_size, 1);
}

int main(int argc, char** argv){

  const int size = 200;
  const int block_size = 24;
  const int test = 1;

  typedef Eigen::Matrix<float, block_size, 1> VectorBlock;
  typedef Eigen::Matrix<float, block_size, block_size> MatrixBlock;



  Eigen::MatrixXf eH;
  eH = Eigen::MatrixXf::Random(size*block_size, size*block_size);
  eH = eH.transpose() * eH;

  Eigen::VectorXf eb;
  eb = Eigen::VectorXf::Random(size*block_size);

  // std::cerr << "Input Mat:\n" <<  eH << std::endl;
  // std::cerr << "Input Vector:\n" << eb.transpose() << std::endl;
  Eigen::VectorXf edx;
  auto eigen_start = std::chrono::high_resolution_clock::now();
  for(int i=0; i < test; ++i)
    edx = eH.ldlt().solve(eb);
  auto eigen_end = std::chrono::high_resolution_clock::now();
  double eigen_time = std::chrono::duration_cast<std::chrono::microseconds>(eigen_end- eigen_start).count();

  BoMatrix<MatrixBlock> maryH;
  maryH.allocate(size, size);
  copyMatrixValues(maryH, eH, size, block_size);

  BoVector<VectorBlock> maryB;
  maryB.allocate(size);
  copyVectorValues(maryB, eb, size, block_size);

  BoVector<VectorBlock> maryDx;
  maryDx.allocate(size);


  // std::cerr << "Input Mary H:\n" << maryH << std::endl;
  // std::cerr << "Input Mary B:\n" << maryB << std::endl;

  CholSolver<VectorBlock, MatrixBlock> chol_solver;
  chol_solver.allocate(size);

  CholSolver<VectorBlock, MatrixBlock>::Timings time_stats;

  auto mary_start = std::chrono::high_resolution_clock::now();
  for(int i=0; i < test; ++i)
    chol_solver.solve(maryDx, maryH, maryB, time_stats);
  auto mary_end = std::chrono::high_resolution_clock::now();
  double mary_time = std::chrono::duration_cast<std::chrono::microseconds>(mary_end- mary_start).count();

  float solution_error = 0.f;
  for(int i=0; i<size; ++i) {
    solution_error += (edx.block(i*block_size, 0, block_size, 1) - maryDx.at(i)).norm();
  }

  std::cerr << "\n\n";
  std::cerr << "********* MatrixXf Solution ***********" << std::endl;
  std::cerr << " "<<test<<" iter took: " << eigen_time << " us" << std::endl;
  std::cerr << "\n";
  std::cerr << "********* BoMary   Solution ***********" << std::endl;
  std::cerr << " "<<test<<" iter took: " << mary_time << " us" << std::endl;
  std::cerr << " of which: " << std::endl;
  std::cerr << " Cholesky : " << time_stats.cholesky / test << " us"  << std::endl;
  std::cerr << " Forward  : " << time_stats.forward / test  << " us" << std::endl;
  std::cerr << " Backward : " << time_stats.backward / test << " us"  << std::endl;  
  std::cerr << "\n***************************************\n" << std::endl;
  std::cerr << " Solution Error: " << solution_error / size << std::endl;
  std::cerr << "\n***************************************\n" << std::endl;

  return 0;
}
