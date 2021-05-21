* subclass the `TableReader` into `EasyOCRReader` and `PytesseractReader` 
 with each having engine-specific args for `__init__`.
* make a 2-step recog: 
    1. recog the table
    2. OCR the cells
* setup some way for each columns/rows's recog to have
  a custom engine/policy