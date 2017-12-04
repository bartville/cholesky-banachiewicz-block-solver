namespace bo_mary {

  template <typename Block>  
  void BoMatrix<Block>::allocateColBlocks(RowBlock& row_block,
                                          const int Nc,
                                          const int starting_id){
    for(int c = 0; c < Nc; ++c) {
      Block* block = new Block();
      row_block.insert(std::pair<int, Block*>(c+starting_id, block));
    }
  }
  
  template <typename Block>
  void BoMatrix<Block>::allocate(const int Nr, const int Nc) {
    // check if the allocated space is enough
    if(_rows_allocated >= Nr) {
      if(_cols_allocated >= Nc) {
        _rows = Nr;
        _cols = Nc;        
        return;
      }
    }    
    // if not, allocate first needed rows
    if(_rows_allocated < Nr){
      for(int r = _rows_allocated; r < Nr; ++r){
        RowBlock* row_block = new RowBlock();
        allocateColBlocks(*row_block, _cols_allocated);     
        _bomatrix.insert(std::pair<int, RowBlock*>(r, row_block));      
      }
      _rows_allocated = Nr;
    }
    
    // then, for each row, allocate the needed cols
    if(_cols_allocated < Nc){
      for(int r = 0; r < _rows_allocated; ++r) {
        RowBlock& row_block = *_bomatrix.at(r);
        allocateColBlocks(row_block, Nc-_cols_allocated, _cols_allocated);
      }
      _cols_allocated = Nc;
    }            
    _rows = Nr;
    _cols = Nc;
  }


  template<typename Block>
  void BoMatrix<Block>::computeCholesky(BoTriangMatrix<Block>& L) {
    if( L.rows() != _rows || L.rows() != _cols)
      throw std::runtime_error("[BoMatrix]: Cholesky computation not allowed!");

    const int num_rows = L.rows();
    Block accumulator;
    
    for (int r = 0; r < num_rows; ++r)
      for (int c = 0; c < r+1 ; ++c) {

        accumulator.setZero();
        for (int k = 0; k < c; ++k) {
          accumulator += L.at(r,k) * L.at(c,k).transpose();
        }
        
        if(r != c) {
          L.at(r,c) = (this->at(r,c) - accumulator) * L.at(c,c).inverse().transpose();
        }
        else  {          
          L.at(c,c) = (this->at(r,c) - accumulator).llt().matrixL();
        }
      }
  }

  
}

