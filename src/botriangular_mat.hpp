namespace bo_mary {

  template <typename Block>  
  void BoTriangMatrix<Block>::allocateColBlocks(RowBlock& row_block,
                                                const int Nc) {
    for(int c = 0; c < Nc; ++c) {
      Block* block = new Block();
      row_block.insert(std::pair<int, Block*>(c, block));
    }
  }

  
  template<typename Block>
  void BoTriangMatrix<Block>::allocate(const int Nr) {
    if(_rows_cols_allocated >= Nr) {
      _rows_cols = Nr;
      return;
    } else {
      for(int r = _rows_cols_allocated; r < Nr; ++r) {
        RowBlock* row_block = new RowBlock();
        allocateColBlocks(*row_block, r+1);
        _bomatrix.insert(std::pair<int, RowBlock*>(r, row_block));
      }
    }
    _rows_cols_allocated = Nr;
    _rows_cols = Nr;
    
  }
  
  

  
}
