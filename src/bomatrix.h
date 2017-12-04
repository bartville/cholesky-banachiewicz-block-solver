#pragma once
#include <iostream>
#include <map>
#include "botriangular_mat.h"
#include "bovector.h"


namespace bo_mary {
  template<typename Block> class BoMatrix {
    
    typedef std::map<int, Block*, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Block*> > > RowBlock;
    typedef std::map<int, RowBlock*, std::less<int>, Eigen::aligned_allocator<std::pair<const int, RowBlock*> > > RowBlockMap;
    
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    BoMatrix() {
      _rows = 0;
      _cols = 0;
      _rows_allocated = 0;
      _cols_allocated = 0;
    }
    
    ~BoMatrix() {
      for(int r = 0; r < _rows_allocated; ++r) {
        RowBlock* row_block = _bomatrix.at(r);
        for(int c = 0; c < _cols_allocated; ++c) {
          delete row_block->at(c);
        }
        delete row_block;
      }      
      std::cerr << "[BoMatrix]: deleted!" << std::endl; 
    }

    inline void allocate(const int Nr, const int Nc);

    inline Block& at(const int r, const int c) {
      if(r >= _rows || c >= _cols)
        throw std::runtime_error("accessing index out of bomatrix size!");      
      return *(_bomatrix.at(r)->at(c));
    }

    inline const Block& at(const int r, const int c) const {
      Block& b = at(r,c);
      return b;
    }

    inline const int rows() const {
      return _rows;
    }
     
    inline const int cols() const {
      return _cols;
    }
    
    inline const int size() const {
      return _rows*_cols;
    }

    inline void computeCholesky(BoTriangMatrix<Block>& L);

    friend std::ostream& operator<<(std::ostream& stream, BoMatrix<Block>& mat) {
      const int block_rows = mat.at(0,0).rows();  
      const int rows = mat.rows();
      const int cols = mat.cols();
      for(int r = 0; r < rows; ++r) {
        for(int block_r = 0; block_r < block_rows; ++block_r) {
          for(int c = 0; c < cols; ++c)
            stream << mat.at(r,c).row(block_r) << " ";      
          stream << std::endl;
        }
      }
      return stream;
    }
    
  protected:
    inline void allocateColBlocks(RowBlock& row_block,
                                  const int Nc,
                                  const int starting_id = 0);   
    
  private:
    int _rows;
    int _cols;

    int _rows_allocated;
    int _cols_allocated;

    RowBlockMap _bomatrix;
  
  };

}
 
#include "bomatrix.hpp"
