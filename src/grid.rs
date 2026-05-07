/// Minimal Grid type for Stage 0.
///
/// Copied from b-app cerebellum, trimmed to what the experiment needs.
/// A 2D colored grid — the universal state representation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<u8>>,
}

impl Grid {
    /// Create a grid filled with a single color.
    pub fn filled(width: usize, height: usize, color: u8) -> Self {
        Self {
            width,
            height,
            cells: vec![vec![color; width]; height],
        }
    }

    /// Create from row data.
    pub fn from_rows(rows: Vec<Vec<u8>>) -> Self {
        let height = rows.len();
        let width = if height > 0 { rows[0].len() } else { 0 };
        Self {
            width,
            height,
            cells: rows,
        }
    }

    /// Get cell value (None if out of bounds).
    pub fn get(&self, row: usize, col: usize) -> Option<u8> {
        self.cells.get(row).and_then(|r| r.get(col)).copied()
    }

    /// Set cell value (no-op if out of bounds).
    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        if row < self.height && col < self.width {
            self.cells[row][col] = val;
        }
    }

    /// Hamming distance — count of cells that differ.
    /// Returns usize::MAX if grids have different dimensions.
    pub fn hamming_distance(&self, other: &Grid) -> usize {
        if self.width != other.width || self.height != other.height {
            return usize::MAX;
        }
        let mut dist = 0;
        for r in 0..self.height {
            for c in 0..self.width {
                if self.cells[r][c] != other.cells[r][c] {
                    dist += 1;
                }
            }
        }
        dist
    }

    /// Total number of cells.
    pub fn size(&self) -> usize {
        self.width * self.height
    }
}

impl std::fmt::Display for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.cells {
            for (i, &cell) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", cell)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filled() {
        let g = Grid::filled(3, 3, 0);
        assert_eq!(g.width, 3);
        assert_eq!(g.height, 3);
        assert_eq!(g.cells[1][1], 0);
    }

    #[test]
    fn test_hamming_identical() {
        let a = Grid::filled(3, 3, 0);
        let b = Grid::filled(3, 3, 0);
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_hamming_one_diff() {
        let a = Grid::filled(3, 3, 0);
        let mut b = Grid::filled(3, 3, 0);
        b.set(1, 1, 1);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_hamming_different_dims() {
        let a = Grid::filled(3, 3, 0);
        let b = Grid::filled(4, 4, 0);
        assert_eq!(a.hamming_distance(&b), usize::MAX);
    }

    #[test]
    fn test_equality() {
        let a = Grid::from_rows(vec![vec![0, 1], vec![1, 0]]);
        let b = Grid::from_rows(vec![vec![0, 1], vec![1, 0]]);
        let c = Grid::from_rows(vec![vec![1, 0], vec![0, 1]]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
