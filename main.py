import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Dict, List, Union, Optional, TYPE_CHECKING, Any, Tuple, DefaultDict

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from typing_extensions import Literal

try:
    from PIL import Image
except ImportError:
    import Image

if TYPE_CHECKING:
    try:
        from easyocr import Reader
    except ImportError:
        pass

CropParam = Union[float, Tuple[float, float], Tuple[float, float, float, float]]


class BaseOCREngine:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    def __init__(self, cropping: CropParam = 0):
        self.cropping: CropParam = cropping

    def preprocess(self, cell_img: np.ndarray) -> np.ndarray:
        border = cv2.copyMakeBorder(cell_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dilation = cv2.dilate(resizing, self.kernel, iterations=1)
        erosion = cv2.erode(dilation, self.kernel, iterations=2)
        return erosion

    def read_cell(self, cell: np.ndarray) -> str:
        raise NotImplemented()


class EasyOCREngine(BaseOCREngine):

    def __init__(self, reader: Optional[Reader], text_read_args: Dict[str, Any], cropping: CropParam = 0):
        super().__init__(cropping)
        if reader is None:
            from easyocr import Reader
            self.reader = Reader(lang_list=["en"])
        else:
            self.reader = reader
        self.read_args = text_read_args

    def read_cell(self, cell: np.ndarray) -> str:
        return self.reader.readtext(cell, **self.read_args)


class PyTesseractEngine(BaseOCREngine):

    def __init__(self, text_read_args: Dict[str, Any], cropping: CropParam = 0):
        super().__init__(cropping)
        self.read_args = text_read_args
        import pytesseract
        self.pytesseract = pytesseract

    def read_cell(self, cell: np.ndarray) -> str:
        return self.pytesseract.image_to_string(cell, **self.read_args)


@dataclass(order=True)
class Cell:
    """Represents a contoured cell. Instances can be sorted, in the attribute's
    order"""
    x: int
    y: int
    w: int
    h: int
    text: Optional[str] = field(init=False, compare=False)
    ocr_engine: Optional[BaseOCREngine] = field(init=False, compare=False)

    def read_text(self, img: np.ndarray, ocr_engine: BaseOCREngine):
        # cropping the original image to the cell's position.
        # Note: by default, y is the first index
        cell_img = img[self.y:self.y + self.h, self.x:self.x + self.w]
        if self.ocr_engine is not None:
            ocr_engine = self.ocr_engine

        cell_img = ocr_engine.preprocess(cell_img)
        self.text = ocr_engine.read_cell(cell_img)

    def __or__(self, other: 'Cell'):
        return Cell(min(self.x, other.x),
                    min(self.y, other.y),
                    max(self.w, other.w),
                    max(self.h, other.h))


@dataclass
class CellSet:
    cells: List[Optional[Cell]] = field(default=list)

    @property
    def supercell(self) -> Cell:
        return reduce(lambda x, y: x | y, self.cells)


class LinesDetector:

    def __init__(self):
        self.intermediate_stages: Dict[str, np.ndarray] = {}
        self.log_intermediate: bool = False

    def init_kernels(self, img: np.ndarray):
        # countcol(width) of kernel as 100th of total width
        self.kernel_len = np.array(img).shape[1] // 100
        # Defining a vertical kernel to detect all vertical lines of image
        self.ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        # Defining a horizontal kernel to detect all horizontal lines of image
        self.hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        # A kernel of 2x2
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    def detect_vertical_lines(self, img: np.ndarray) -> np.ndarray:
        # Use vertical kernel to detect vertical lines
        eroded = cv2.erode(img, self.ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(eroded, self.ver_kernel, iterations=3)
        if self.log_intermediate:
            self.intermediate_stages["vertical_lines"] = vertical_lines
        return vertical_lines

    def detect_horizontal_lines(self, img: np.ndarray) -> np.ndarray:
        # Use vertical kernel to detect vertical lines
        eroded = cv2.erode(img, self.hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(eroded, self.hor_kernel, iterations=3)
        if self.log_intermediate:
            self.intermediate_stages["horizontal_lines"] = horizontal_lines
        return horizontal_lines


class CellDetector:
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    def __init__(self, lines_detector: LinesDetector):
        self.lines_detector = lines_detector
        self.intermediate_stages: Dict[str, np.ndarray] = {}
        self.log_intermediate: bool = False

    def detect_cells_contours(self,
                              img: np.ndarray,
                              vertical_lines: np.ndarray,
                              horizontal_lines: np.ndarray):
        # Combine horizontal and vertical lines in a new third image, with both having same weight.
        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # Eroding and thesholding the image
        img_vh = cv2.erode(~img_vh, self.kernel, iterations=2)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if self.log_intermediate:
            self.intermediate_stages["table_outline"] = img_vh
        bitxor = cv2.bitwise_xor(img, img_vh)
        bitnot = cv2.bitwise_not(bitxor)

        # Detect contours for following box detection
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return bitnot, contours, hierarchy

    def detect_cells(self, img_bin: np.ndarray, img: np.ndarray) -> List[Cell]:
        vertical_lines = self.lines_detector.detect_vertical_lines(img_bin)
        horizontal_lines = self.lines_detector.detect_horizontal_lines(img_bin)

        bitnot, contours, hierarchy = self.detect_cells_contours(vertical_lines,
                                                                 horizontal_lines)
        all_cells: List[Cell] = []
        if self.log_intermediate:
            contoured_cells = img.copy()
        # Get position (x,y), width and height for every contour and show the contour on image
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # cell/box is kept if not too large
            # TODO : cell size limit to be set as a parameter
            if (w < 1000 and h < 500):
                if self.log_intermediate:
                    contoured_cells = cv2.rectangle(contoured_cells,
                                                    (x, y),
                                                    (x + w, y + h),
                                                    (0, 255, 0),
                                                    10)
                all_cells.append(Cell(x=x, y=x, w=w, h=h))

        if self.log_intermediate:
            self.intermediate_stages["contoured_cells"] = contoured_cells

        return all_cells


class BaseCellAllocator:

    def place_cells(self, img: np.ndarray, cells: List[Cell]) -> Tuple[List[CellSet], List[CellSet]]:
        raise NotImplemented()


class GMMCellAllocator(BaseCellAllocator):

    def __init__(self, lines_detector: LinesDetector):
        self.lines_detector = lines_detector

    def gmm_allocation(self, lines_histogram: np.ndarray,
                       cells_pos: np.ndarray,
                       cells: List[Cell]) -> List[CellSet]:
        lines_peaks, _ = find_peaks(lines_histogram, prominence=lines_histogram.max() / 2)
        lines_centers = (lines_peaks[1:] + lines_peaks[:-1]) / 2
        lines_spacing = (lines_peaks[1:] - lines_peaks[:-1])
        gmm_means = np.vstack((lines_centers, lines_spacing))
        gmm_means = gmm_means.swapaxes(0, 1)

        column_gmm = GaussianMixture(n_components=len(lines_centers),
                                     means_init=gmm_means)
        column_gmm.fit(cells_pos)
        predictions = column_gmm.predict(cells_pos)

        cell_sets: DefaultDict[int, CellSet] = defaultdict(CellSet)
        for cell, set_id in zip(cells, predictions):
            cell_sets[set_id].cells.append(cell)
        return list(cell_sets.values())

    def place_cells(self, img: np.ndarray, cells: List[Cell]) -> Tuple[List[CellSet], List[CellSet]]:
        vertical_lines = self.lines_detector.detect_vertical_lines(img)
        vertical_cells_pos = np.array([(c.x + (c.w / 2), c.w) for c in cells])
        columns = self.gmm_allocation(vertical_lines.sum(axis=0), vertical_cells_pos, cells)
        for col in columns:
            col.cells.sort(key=lambda c: c.y)

        horizontal_lines = self.lines_detector.detect_horizontal_lines(img)
        horizontal_cells_pos = np.array([(c.y + (c.h / 2), c.h) for c in cells])
        rows = self.gmm_allocation(horizontal_lines.sum(axis=1), horizontal_cells_pos, cells)
        for row in rows:
            row.cells.sort(key=lambda c: c.x)
        return columns, rows


class ClassicCellAllocator(BaseCellAllocator):

    def place_cells(self, img: np.ndarray, cells: List[Cell]) -> Tuple[List[CellSet], List[CellSet]]:

        y_sorted_cells = sorted(cells, key=lambda c: c.y)
        heights_mean = np.array([cell.h for cell in cells]).mean()
        # Creating two lists to define row and column in which cell is located
        rows = []
        current_row = []
        previous = None
        # Sorting the boxes to their respective row
        for cell in cells:

            if previous is None:
                current_row.append(cell)
                previous = cell

            else:
                # if current cell isn't "under" the former, add it to current row
                if cell.y <= previous.y + heights_mean / 2:
                    current_row.append(cell)
                    previous = cell

                # else, create new row for the current cell
                else:
                    rows.append(current_row)
                    current_row = []
                    previous = cell
                    current_row.append(cell)
        else:
            if current_row:
                # append the last column to list of columns
                rows.append(current_row)

        # inside each row, sort by x
        for row in rows:
            row.sort(key=lambda c: c.x)

        # calculating maximum number of cells per row, i.e., max number of columns
        column_count = max(len(r) for r in rows)
        row_max_columns = np.argmax([len(r) for r in rows])

        # Retrieving the center of each column (using first row)
        center = [int(cell.x + (cell.w / 2)) for cell in rows[row_max_columns]]

        center = np.array(center)
        center.sort()

        # Assigning the cell in each row to the column it's closest to
        final_columns = [CellSet([None for _ in range(len(rows))])]
        final_rows = [CellSet([None for _ in range(len(rows))])]
        for row_id, row in enumerate(rows):
            for cell in row:
                diff = np.abs(center - (cell.x + cell.w / 4))
                closest_column_idx = np.argmin(diff)
                final_columns[closest_column_idx].cells[row_id] = cell
                final_rows[row_id].cells[closest_column_idx] = cell

        return final_rows, final_columns


@dataclass
class Table:
    rows: List[CellSet]
    columns: List[CellSet]
    img: np.ndarray

    def set_header_row(self, row: Union[CellSet, int] = 0):
        pass

    def set_index_col(self, column: Union[CellSet, int] = 0):
        pass

    def set_ocr_engine(self, cell_set: Union[CellSet, int],
                       ocr_engine: BaseOCREngine,
                       set_type: Literal["row", "column"] = "column"):
        if isinstance(cell_set, int):
            cell_set = self.rows[cell_set] if set_type == "row" else self.columns[i]

        for cell in cell_set.cells:
            cell.ocr_engine = ocr_engine

    def read_cells(self, ocr_engine: BaseOCREngine):
        for row in self.rows:
            for cell in row.cells:
                if cell is None:
                    continue
                cell.read_text(self.img, ocr_engine)
d

def detect_table(img: np.ndarray,
                 cell_allocation: Literal["gmm", "classic"] = "gmm") -> Table:
    # thresholding the image to a binary image and inverting
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    lines_detector = LinesDetector()
    cell_detector = CellDetector(lines_detector)
    cells = cell_detector.detect_cells(img_bin, img)
    if cell_allocation == "gmm":
        cell_allocator = GMMCellAllocator(lines_detector)
    else:
        cell_allocator = ClassicCellAllocator()
    rows, columns = cell_allocator.place_cells(img_bin, cells)
    table = Table(rows, columns, img_bin)
    return table


def read_table(img: np.ndarray,
               cell_allocation: Literal["gmm", "classic"] = "gmm",
               ocr_engine: Optional[BaseOCREngine] = None) -> pd.DataFrame:
    table = detect_table(img, cell_allocation)
    table.read_cells(ocr_engine)
    # TODO: convert to pd dataframe


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=Path)

    args = argparser.parse_args()

    # read your file
    img = cv2.imread(args.file, 0)
    print("Img shape is ", img.shape)
