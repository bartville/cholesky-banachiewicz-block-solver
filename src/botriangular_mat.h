#pragma once
#include <iostream>
#include <map>

namespace bo_mary {
  template<typename Block> class BoTriangMatrix {

    typedef std::map<int, Block*, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Block*> > > RowBlock;
    typedef std::map<int, RowBlock*, std::less<int>, Eigen::aligned_allocator<std::pair<const int, RowBlock*> > > RowBlockMap;
  
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    BoTriangMatrix() {
      _rows_cols = 0;
      _rows_cols_allocated = 0;
    }
    
    ~BoTriangMatrix() {
      for(int r = 0; r < _rows_cols_allocated; ++r) {
        RowBlock* row_block = _bomatrix.at(r);
        for(int c = 0; c <= r; ++c) {
          delete row_block->at(c);
        }
        delete row_block;
      }      
      std::cerr << "[BoTriangMatrix]: deleted!" << std::endl; 
    }

    inline void allocate(const int Nr);

    inline Block& at(const int r, const int c) {
      if(c > r || r >= _rows_cols)
        throw std::runtime_error("accessing index out of boTriangMatrix size!");      
      return *(_bomatrix.at(r)->at(c));
    }

    inline const Block& at(const int r, const int c) const {
      Block& b = at(r,c);
      return b;
    }

    inline const int rows() const {
      return _rows_cols;
    }
     
    inline const int cols() const {
      return _rows_cols;
    }
    
    inline const int size() const {
      return _rows_cols*_rows_cols / 2;
    }

    friend std::ostream& operator<<(std::ostream& stream, BoTriangMatrix<Block>& mat) {
      const int block_rows = mat.at(0,0).rows();  
      const int rows = mat.rows();
      for(int r = 0; r < rows; ++r) {
        for(int block_r = 0; block_r < block_rows; ++block_r) {
          for(int c = 0; c < r+1; ++c)
            stream << mat.at(r,c).row(block_r) << " ";      
          stream << std::endl;
        }
      }
      return stream;
    }


  protected:
    inline void allocateColBlocks(RowBlock& row_block,
                           const int Nc);   
    
  private:
    int _rows_cols;
    int _rows_cols_allocated;

    RowBlockMap _bomatrix;
  
  };


}
 
#include "botriangular_mat.hpp"
