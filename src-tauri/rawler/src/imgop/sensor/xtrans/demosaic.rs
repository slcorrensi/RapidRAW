use crate::{
  cfa::{PlaneColor, CFA, CFA_COLOR_B, CFA_COLOR_G, CFA_COLOR_R},
  imgop::{
    sensor::bayer::Demosaic,
    Dim2, Point, // Import Point
    Rect,
  },
  pixarray::{Color2D, PixF32},
};
use rayon::prelude::*;
use std::cmp::{min, max};

#[derive(Default)]
pub struct XTransDemosaic {}

impl XTransDemosaic {
    pub fn new() -> Self {
        Self {}
    }

    // Helper function to interpolate borders
    fn border_interpolate(border: usize, image: &mut Color2D<f32, 3>, cfa: &CFA) {
        let colors = 3; // Number of color components (RGB)
        let mut sum = [0.0; 8]; // Using f32 for summing; adjust size as needed

        for row in 0..image.height {
            for col in 0..image.width {
                if col == border && row >= border && row < image.height - border {
                    continue; // Skip to next iteration for border columns
                }

                // Reset sum array to zero
                for item in &mut sum {
                    *item = 0.0;
                }

                // Check neighboring pixels
                let y_start = if row > 0 { row - 1 } else { 0 };
                let y_end = if row + 1 < image.height {
                    row + 1
                } else {
                    image.height - 1
                };
                let x_start = if col > 0 { col - 1 } else { 0 };
                let x_end = if col + 1 < image.width {
                    col + 1
                } else {
                    image.width - 1
                };

                for y in y_start..=y_end {
                    for x in x_start..=x_end {
                        let color = cfa.color_at(y, x);
                        for c in 0..colors {
                            sum[color] += image.data[y * image.width + x][color];
                            sum[c + 4] += 1.0; // Count occurrences
                        }
                    }
                }

                for c in 0..colors {
                    if sum[c + colors] > 0.0 && sum[c + 4] != 0.0 {
                        image.data[row * image.width + col][c] = sum[c] / sum[c + colors];
                    }
                }
            }
        }
    }

    fn clip(x: f32) -> f32 {
        Self::lim(x, 0.0f32, 65535f32)
    }

    fn lim(x: f32, min_val: f32, max_val: f32) -> f32 {
        max(min_val, min(x, max_val)) as f32
    }

    fn get_pix(pixels: &PixF32, row: usize, col: usize, offset: i16) -> &f32 {
        let index = row as i16 * pixels.width as i16 + col as i16 + offset;
        &pixels.data[index as usize]
    }

    fn get_rgb(rgb_image: &Color2D<f32, 3>, row: usize, col: usize, offset: i16) -> &[f32; 3] {
        let index = row as i16 * rgb_image.width as i16 + col as i16 + offset;
        &rgb_image.data[index as usize]
    }

    fn write_pix(rgb_image: &mut Color2D<f32, 3>, color_channel: usize, row: usize, col: usize, offset: i16, value: f32) {
        let index = row as i16 * rgb_image.width as i16 + col as i16 + offset;
        rgb_image.data[index as usize][color_channel] = value;
    }   

}

impl Demosaic<f32, 3> for XTransDemosaic {

    /// Frank Markesteijn's algorithm for Fuji X-Trans sensors
    fn demosaic(&self, pixels: &PixF32, cfa: &CFA, _colors: &PlaneColor, roi: Rect) -> Color2D<f32, 3> {
        
        // Fixed number of passes for high-quality demosaicing
        let passes = 3; 

        // Define ROI dimensions
        let width = roi.width();
        let height = roi.height();

        // Initialize the output color image
        let mut out = Color2D::<f32, 3>::new(width, height);

        // Core variables for the interpolation process
        let ndir = 4;
        let mut allhex = [[[[0; 8]; 2]; 3]; 3];

        // Define static arrays used in interpolation
        const TS: usize = 114; // tile size
        const TSH: usize = TS / 2; // half tile size
        let orth = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1];
        let patt = [
            [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
            [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
        ];
        let dir = [1, TS as i16, (TS + 1) as i16, (TS - 1) as i16];

        // Offset in the sensor matrix of the solitary green pixels
        let mut sgrow: u16 = 0;
        let mut sgcol: u16 = 0;

        // Map a green hexagon around each non-green pixel and vice versa
        for row in 0..3 {
            for col in 0..3 {
                for d_val in (0..10).step_by(2) {
                    let mut ng = 0; // Number of non green pixels adjacent to the current pixel
                    let g = if cfa.color_at(row, col) == CFA_COLOR_G { 1 } else { 0 };

                    // Check neighboring pixels
                    let row_check = (row as i32 + orth[d_val] as i32) as usize;
                    let col_check = (col as i32 + orth[d_val + 2] as i32) as usize;
                    if cfa.color_at((row_check + 48) % 48, (col_check + 48) % 48) == CFA_COLOR_G {
                        ng = 0;
                    } else {
                        ng += 1;
                    }

                    // If there are four non-green pixels adjacent in cardinal directions, this is the solitary green pixel
                    if ng == 4 {
                        sgrow = row as u16;
                        sgcol = col as u16;
                    }

                    if ng == g + 1 {
                        for c in 0..8 {
                            let orth_d = orth[d_val];
                            let orth_d_2 = orth[d_val + 2];
                            let orth_d_3 = orth[d_val + 1];
                            let orth_d_4 = orth[d_val + 3];

                            let v = orth_d as i32 * patt[g as usize][c * 2] as i32
                                + orth_d_3 as i32 * patt[g as usize][c * 2 + 1] as i32;
                            let h = orth_d_2 as i32 * patt[g as usize][c * 2] as i32
                                + orth_d_4 as i32 * patt[g as usize][c * 2 + 1] as i32;

                            let index = c ^ ((g * 2) & d_val as usize);
                            allhex[row][col][0][index] = (h + v * roi.d.w as i32) as i16;
                            allhex[row][col][1][index] = (h + v * TS as i32) as i16;
                        }
                    }
                }
            }
        }

        let mut right_shift = vec![3];
        for row in 0..3 {
            // Count number of green pixels in 3 cols
            let mut green_count = 0;
            for col in 0..3 {
                if cfa.color_at(row, col) == CFA_COLOR_G {
                    green_count += 1;
                }
                right_shift[row] = green_count; 
            }
        }

        #[derive(Clone, Copy)]
        struct MinMaxGreen {
            min: f32,
            max: f32,
        }

        // Allocate separate arrays
        let mut lab = vec![vec![0.0f32; (TS - 8) as usize]; (TS - 8) as usize];
        let mut drv = vec![vec![0.0f32; (TS - 10) as usize]; (TS - 10) as usize];
        let mut homo = vec![vec![0u8; TS as usize]; TS as usize];
        let mut green_min_max_tile = vec![MinMaxGreen { min: 0.0, max: 0.0 }; TSH as usize];
        let mut homo_sum = vec![vec![0u8; TS as usize]; TS as usize];
        let mut homo_sum_max = vec![0u8; TS as usize];

        // Tile processing
        let mut top = 3;
        while top < height - 19 {
            let mut left = 3;
            while left < width - 19 {
                let mrow = min(top + TS, height - 3);
                let mcol = min(left + TS, width - 3);

                // Set greenmin and greenmax to the minimum and maximum allowed values
                for row in top..mrow {
                    // Find first non-green pixel
                    let mut left_start = left;

                    for _ in left_start..mcol {
                        if cfa.color_at(row, left_start) == CFA_COLOR_G {
                            break;
                        } else {
                            left_start += 1;
                        }
                    }

                    let mut col_offset = if right_shift[row % 3] == 1 {
                        3
                    } else {
                        1 + (cfa.color_at(row, left_start + 1) & 1)
                    };

                    if col_offset == 3 {
                        let hex = allhex[row % 3][left_start % 3][0];
                        let mut col = left_start;
                        while col < mcol {
                            let mut minval: f32 = f32::MAX;
                            let mut maxval: f32 = 0.0;

                            for c in 0..6 {
                                let val = pixels.data[(row as i16 * width as i16 + col as i16 + hex[c]) as usize];
                                minval = minval.min(val);
                                maxval = maxval.max(val);
                            }

                            let index = (row - top) * TSH + (col - left) / 2;
                            green_min_max_tile[index].min = minval;
                            green_min_max_tile[index].max = maxval;

                            col += col_offset;
                        }
                    } else {
                        let mut minval: f32 = f32::MAX;
                        let mut maxval: f32 = 0.0;
                        let mut col = left_start;

                        if col_offset == 2 {
                            // Get the hex pattern for this position
                            let hex = &allhex[0][row % 3][col % 3];

                            // Process the 6 neighboring pixels
                            for c in 0..6 {
                                let val = pixels.data[(row as i16 * width as i16 + col as i16 + hex[c]) as usize];
                                minval = minval.min(val);
                                maxval = maxval.max(val);
                            }

                            // Calculate the index for green_min_max_tile
                            let tile_row = row - top;
                            let tile_col = (col - left) / 2;
                            let index = tile_row * TSH + tile_col;

                            // Update the min/max values
                            green_min_max_tile[index].min = minval;
                            green_min_max_tile[index].max = maxval;
                            col += 2;
                        }

                        let hex = allhex[0][row % 3][left_start % 3];

                        while col < mcol.saturating_sub(1) {
                            let mut minval: f32 = f32::MAX;
                            let mut maxval: f32 = 0f32;

                            for c in 0..6 {
                                let val = pixels.data[(row as i16 * width as i16 + col as i16 + hex[c]) as usize];
                                minval = minval.min(val);
                                maxval = maxval.max(val);
                            }

                            let index1 = (row - top) * TSH + (col - left) / 2;
                            let index2 = (row - top) * TSH + (col + 1 - left) / 2;
                            green_min_max_tile[index1].min = minval;
                            green_min_max_tile[index1].max = maxval;
                            green_min_max_tile[index2].min = minval;
                            green_min_max_tile[index2].max = maxval;

                            col += 3;
                        }

                        if col < mcol {
                            let mut minval: f32 = f32::MAX;
                            let mut maxval: f32 = 0f32;

                            for c in 0..6 {
                                let val = pixels.data[(row as i16 * width as i16 + col as i16 + hex[c]) as usize];
                                minval = minval.min(val);
                                maxval = maxval.max(val);
                            }

                            let index = (row - top) * TSH + (col - left) / 2;
                            green_min_max_tile[index].min = minval;
                            green_min_max_tile[index].max = maxval;
                        } 
                    }
                }

                for row in top..mrow {
                    for col in left..mcol {
                        let pixel_value = pixels.data[row * width + col];
                        out.data[row * width + col] = [pixel_value, pixel_value, pixel_value];
                    }
                }

                // Interpolate green horizontally, vertically and along both diagonals
                for row in top..mrow {

                    // Find first non green pixel
                    let mut left_start = left;
                    while left_start < mcol {
                        if cfa.color_at(row, left_start) != 1 {
                            break;
                        } else {
                            left_start += 1;
                        }
                    }

                    let col_offset = if right_shift[row % 3] == 1 {
                        3
                    } else {
                        1 + (cfa.color_at(row, left_start + 1) & 1)
                    };

                    if col_offset == 3 {
                        let hex = &allhex[row % 3][left_start % 3][0];
                        let mut col = left_start;
                        while col < mcol {
                            let pix_base_idx = col;
                            let mut color = [0.0f32; 4];
                            
                            color[0] = 0.6796875f32 * (Self::get_pix(&pixels, row, col, hex[1]) + Self::get_pix(&pixels, row, col,hex[0])) 
                                    - 0.1796875f32 * (Self::get_pix(&pixels, row, col,2 * hex[1]) + Self::get_pix(&pixels, row, col,2 * hex[0]));
                            
                            color[1] = 0.87109375f32 * Self::get_pix(&pixels, row, col,hex[3])
                                    + Self::get_pix(&pixels, row, col,hex[2]) * 0.12890625f32 
                                    + 0.359375f32 * (Self::get_pix(&pixels, row, col,0) - Self::get_pix(&pixels, row, col,-hex[2]));
                            
                            for c in 0..2 {
                                color[2 + c] = 0.640625f32 * Self::get_pix(&pixels, row, col,hex[4 + c]) 
                                            + 0.359375f32 * Self::get_pix(&pixels, row, col,-2 * hex[4 + c]) 
                                            + 0.12890625f32 * (2.0f32 * Self::get_pix(&pixels, row, col,0) 
                                                            - Self::get_pix(&pixels, row, col,3 * hex[4 + c]) 
                                                            - Self::get_pix(&pixels, row, col,-3 * hex[4 + c]));
                            }
                            
                            for c in 0..4 {
                                let index = (row - top) * TSH + (col - left) / 2;
                                let min_val = green_min_max_tile[index].min;
                                let max_val = green_min_max_tile[index].max;
                                out.data[(row - top) * width + (col - left) + c][1] = color[c].max(min_val).min(max_val);
                            }
                            
                            col += col_offset;
                        }
                    } else {
                        let mut hexmod = [
                            &allhex[row % 3][left_start % 3][0],
                            &allhex[row % 3][(left_start + col_offset) % 3][0],
                        ];

                        let mut col = left_start;
                        let mut hex_index = 0;
                        while col < mcol {
                            let pix = &pixels.data[row * width + col];
                            let hex = hexmod[hex_index];
                            let mut color = [0.0f32; 4];

                            color[0] =   0.6796875f32 * (Self::get_pix(&pixels, row, col, hex[1]) + Self::get_pix(&pixels, row, col, hex[0])) 
                                       - 0.1796875f32 * (Self::get_pix(&pixels, row, col, 2 * hex[1]) + Self::get_pix(&pixels, row, col, 2 * hex[0]));
                            
                            color[1] =   0.87109375f32 * Self::get_pix(&pixels, row, col, hex[3])
                                       + 0.12890625f32 * Self::get_pix(&pixels, row, col, hex[2])
                                       + 0.359375f32 * (Self::get_pix(&pixels, row, col, 0) - Self::get_pix(&pixels, row, col, -hex[2]));
                            
                            for c in 0..2 {
                                color[2 + c] =   0.640625f32 * Self::get_pix(pixels, row, col, hex[4 + c])
                                               + 0.359375f32 * Self::get_pix(pixels, row, col, -2 * hex[4 + c])
                                               + 0.12890625f32 * (2.0f32 * Self::get_pix(pixels, row, col, 0) - Self::get_pix(pixels, row, col, 3 * hex[4 + c]) - Self::get_pix(pixels, row, col, -3 * hex[4 + c]));
                            }

                            for c in 0..4 {
                                let index = (row - top) * TSH + (col - left) / 2;
                                let value = Self::lim(color[c], green_min_max_tile[index].min, green_min_max_tile[index].max);
                                Self::write_pix(&mut out, 1, row - top, col - left, (c ^ 1) as i16, value)
                            }

                            col += col_offset;
                            col_offset ^= 3;
                            hex_index ^= 1;
                        }
                    }
                }
            
                for pass in 0..passes {
                    if pass == 1 {
                        // TRANSLATE !
                        todo!("memcpy")
                    }

                    // Recalculate green from interpolated values of closer pixels
                    if pass != 0 {
                        for row in (top + 2)..(mrow - 2) {
                            
                            let mut left_start = left + 2;
                            while left_start < mcol {
                                if cfa.color_at(row, left_start) != 1 {
                                    break;
                                } else {
                                    left_start += 1;
                                }
                            }

                            let col_offset = if right_shift[row % 3] == 1 {
                                3
                            } else {
                                1 + (cfa.color_at(row, left_start + 1) & 1)
                            };

                            if col_offset == 3 {
                                let f = cfa.color_at(row, left_start);
                                let mut col = left_start;
                                let hex = &allhex[row % 3][left_start % 3][1];
                                while col < mcol - 2 {
                                    for d in 3..6 {
                                        let val = 
                                              0.33333333f32 * Self::get_pix(&out, row, col, -2 * hex[d], 1)
                                            + 2 * (Self::get_pix(&out, row, col, hex[d], 1) - Self::get_pix(&out, row, col, hex[d], f))
                                            - Self::get_pix(&out, row, col, -2 * hex[d], 1) 
                                            + Self::get_pix(&out, row, col, 0, f);
                                        let index = (row - top) * TSH + (col - left) / 2;
                                        let value = Self::lim(val, green_min_max_tile.min, green_min_max_tile.max);
                                        Self::write_pix(out, row, col, 0, 1, value);
                                    }

                                    f ^= 2;
                                    col += col_offset;
                                }
                            } else {
                                let f = cfa.color_at(row, left_start);
                                let mut hex_mod = [
                                    &allhex[row % 3][left_start % 3][1],
                                    &allhex[row % 3][(left_start + col_offset) % 3][1],
                                ];

                                let mut col = left_start;
                                let hex_index = 0;
                                while col < mcol - 2 {

                                    for d in 3..6 {
                                        let base_offset = (d - 2) ^ 1;
                                        let hex = hex_mod[hex_index];
                                        let val =   0.33333333f32 * Self::get_rgb(&out, row - top, col - left, base_offset - 2 * hex[d])[1]
                                                  + 2 * (Self::get_rgb(&out, row - top, col - left, base_offset + hex[d])[1]
                                                       - Self::get_rgb(&out, row - top, col - left, base_offset + hex[d])[f])
                                                  - Self::get_rgb(&out, row - top, col - left, base_offset - 2 * hex[d])[f]
                                                  + Self::get_rgb(&out, row - top, col - left, base_offset)[f];
                                        let index = (row - top) * TSH + (col - left) / 2;
                                        let value = Self::lim(val, green_min_max_tile.min, green_min_max_tile.max);
                                        Self::write_pix(&mut out, 1, row, col, 0, value)
                                    }

                                    col += col_offset;
                                    col_offset ^= 3;
                                    f = f ^(col_offset & 2);
                                    hex_index ^= 1; 
                                }
                                
                            }
                        


                        }
                    }

                    // Interpolate red and blue values for solitary green pixels
                    let sg_start_col = (left - sgrow + 4) / 3 * 3 + sgcol;
                    let mut color = vec![vec![0f32; 6]; 3];

                    for row in ( ((top - sgrow + 4) / 3 * 3)..(mrow - 2) ).step_by(3) {
                        let mut col = sg_start_col;
                        let h = cfa.color_at(row, col + 1);
                        while col < mcol - 2 {
                            let mut diff = vec![0f32; 6];
                            let mut base_offset = 0i16;
                            let i = 1;
                            for d in 0..6 {
                                for c in 0..2 {
                                    let g = 2.0f32 * Self::get_rgb(&mut out, row - top, col - left, base_offset)[1]
                                            - Self::get_rgb(&mut out, row - top, col - left,   base_offset + i << c)[1]
                                            - Self::get_rgb(&mut out, row - top, col - left, base_offset + - i << c)[1];
                                    color[h][d] = g + Self::get_rgb(&mut out, row - top, col - left,   base_offset + i << c)[h]
                                                    + Self::get_rgb(&mut out, row - top, col - left, base_offset + - i << c)[h];
                                    
                                    if d > 1 {
                                        diff[d] += (  Self::get_rgb(&mut out, row - top, col - left,   base_offset + i << c)[1]
                                                    - Self::get_rgb(&mut out, row - top, col - left, base_offset + - i << c)[1]
                                                    - Self::get_rgb(&mut out, row - top, col - left,   base_offset + i << c)[h]
                                                    + Self::get_rgb(&mut out, row - top, col - left, base_offset + - i << c)[h] ).powi(2i32)
                                                   + g.powi(2i32);
                                    }
                                    h ^= 2;
                                }

                                if d > 2 && (d & 1) != 0 {
                                    if diff[d - 1] < diff[d] {
                                        for c in 0..2 {
                                            color[c * 2][d] = color[c * 2][d - 1]; 
                                        }
                                    }
                                }

                                if (d & 1) != 0 || d < 2 {
                                    for c in 0..2 {
                                        let value = Self::clip(0.5f32 * color[c * 2][d]);
                                        Self::write_pix(&mut out, c * 2, row - top, col - left, base_offset, value);
                                    }
                                    base_offset += (TS * TS) as i16;
                                }

                                i ^= (TS ^ 1) as i16;
                                h ^= 2;
                            }

                            col += 3;
                            h ^= 2;
                        }
                    }

                    // Interpolate red for blue pixels and vice versa
                    for row in (top + 3)..(mrow - 3) {
                        
                        let mut left_start = left + 3;
                        while left_start < mcol - 1 {
                            if cfa.color_at(row, left_start) != 1 {
                                break;
                            } else {
                                left_start += 1;
                            }
                        }

                        let col_offset = if right_shift[row % 3] == 1 {
                            3
                        } else {
                            1
                        };

                        let c = if ((row - sgrow) % 3) != 0 { TS } else { 1 };
                        let h = 3 * (c ^ TS ^ 1);

                        if col_offset == 3 {
                            let f = 2 - cfa.color_at(row, left_start);

                            let mut col = left_start;
                            while col < mcol - 3 {

                                let mut base_offset = 0;
                                let g = Self::get_rgb(&out, row - top, col - left, base_offset)[1];
                                let g_pc = Self::get_rgb(&out, row - top, col - left, base_offset + c)[1];
                                let g_mc = Self::get_rgb(&out, row - top, col - left, base_offset - c)[1];
                                let g_ph = Self::get_rgb(&out, row - top, col - left, base_offset + h)[1];
                                let g_mh = Self::get_rgb(&out, row - top, col - left, base_offset - h)[1];
                                for d in 0..4 {
                                    let i = if d > 1 || (d ^ c) & 1 != 0 || (g - g_pc).abs() + (g - g_mc).abs() < 2f32 * (g - g_ph).abs() + (g - g_mh).abs() {
                                        c
                                    } else {
                                        h
                                    };
                                    let c1 = Self::get_rgb(&out, row - top, col - left, base_offset + i)[f];
                                    let c2 = Self::get_rgb(&out, row - top, col - left, base_offset - i)[f];
                                    let c3 = Self::get_rgb(&out, row - top, col - left, base_offset + i)[1];
                                    let c4 = Self::get_rgb(&out, row - top, col - left, base_offset - i)[1];

                                    let value = Self::clip(g + 0.5f32 * (c1 + c2 - c3 - c4));
                                    Self::write_pix(&mut out, f, row, col, base_offset, value);

                                    base_offset += (TS * TS) as i16;
                                }

                                col += col_offset;
                                f ^= 2;
                            }

                        } else {
                            col_offset = if cfa.color_at(row, left_start + 1) == CFA_COLOR_G {
                                2
                            } else {
                                1
                            };

                            let f = 2 - cfa.color_at(row, left_start);

                            let mut col = left_start;
                            let mut base_offset = 0;
                            while col < mcol - 3 {
                                for d in 0..4 {
                                    let i = if d > 1 || (d ^ c) & 1 != 0 || (g - g_pc).abs() + (g - g_mc).abs() < 2f32 * (g - g_ph).abs() + (g - g_mh).abs() {
                                        c
                                    } else {
                                        h
                                    };
                                    let c1 = Self::get_rgb(&out, row - top, col - left, base_offset + i)[f];
                                    let c2 = Self::get_rgb(&out, row - top, col - left, base_offset - i)[f];
                                    let c3 = Self::get_rgb(&out, row - top, col - left, base_offset + i)[1];
                                    let c4 = Self::get_rgb(&out, row - top, col - left, base_offset - i)[1];

                                    let value = Self::clip(g + 0.5f32 * (c1 + c2 - c3 - c4));
                                    Self::write_pix(&mut out, f, row, col, base_offset, value);

                                    base_offset += (TS * TS) as i16;
                                }
                                

                                col += col_offset;
                                col_offset ^= 3;
                                f = f ^ (col_offset & 2);
                            }
                        }
                    }

                    // Fill in red and blue for 2x2 blocks of green
                    
                    // Find first row of 2x2 green
                    let mut top_start = top + 2;
                    while top_start < mrow - 2 {
                        if (top_start - sgrow) % 3 != 0 {
                            break;
                        } else {
                            top_start += 1;
                        }
                    }

                    let mut left_start = left + 2;
                    while left_start < mcol - 2 {
                        if (left_start - sgcol) % 3 != 0 {
                            break;
                        } else {
                            left_start += 1;
                        }
                    }

                    let col_offset_start = 2 - (cfa.color_at(top_start, left_start + 1) & 1);

                    for row in top_start..(mrow - 2) {
                        if (row - sgrow) % 3 != 0 {
                            let mut hexmod = [
                                &allhex[row % 3][left_start % 3][1],
                                &allhex[row % 3][(left_start + col_offset_start) % 3][1],
                            ];

                            let mut col = left_start;
                            let mut col_offset = col_offset_start;
                            let mut hex_index = 0;
                            while col < mcol - 2 {
                                let base_offset = 0;
                                let hex = hexmod[hex_index];
                                
                                for d in (0..ndir).step_by(2) {
                                    if hex[d] + hex[d + 1] != 0 {
                                        let g = 3f32 * Self::get_rgb(&out, row - top , col - left, base_offset)[1]
                                                - 2f32 * Self::get_rgb(&out, row - top , col - left, base_offset + hex[d])[1]
                                                -     Self::get_rgb(&out, row - top , col - left, base_offset + hex[d + 1])[1];
                                        
                                        for c in (0..4).step_by(2) {
                                            let value = Self::clip(g
                                                + 2f32 * Self::get_rgb(&out, row - top , col - left, base_offset + hex[d])[c]
                                                + 0.33333333f32 * Self::get_rgb(&out, row - top , col - left, base_offset + hex[d + 1])[c]);
                                            Self::write_pix(&mut out, c, row - top, col - left, base_offset, value);
                                        }
                                    } else {
                                        let g = 
                                            2f32 * Self::get_rgb(&out, row - top , col - left, base_offset)[1]
                                            - Self::get_rgb(&out, row - top , col - left, base_offset + hex[d])[1]
                                            - Self::get_rgb(&out, row - top , col - left, base_offset + hex[d + 1])[1];
                                        
                                        for c in (0..4).step_by(2) {
                                            let value = Self::clip(g
                                                + Self::get_rgb(&out, row - top , col - left, base_offset + hex[d])[c]
                                                + 0.5f32 * Self::get_rgb(&out, row - top , col - left, base_offset + hex[d + 1])[c]);
                                            Self::write_pix(&mut out, c, row - top, col - left, base_offset, value);
                                        }
                                    }
                                    base_offset += TS * TS;
                                }
                                col += col_offset;
                                col_offset ^= 3;
                                hex_index ^= 1;
                            }
                        }
                    }
                } // end of multipass part

                mrow -= top;
                mcol -= left;






                left += TS - 16;
            }
            top += TS - 16;
        }
        
        todo!("Complete the translation up to the specified comment and beyond.");

        // Return placeholder color image
        out
    }
}












/////////////////////////////////////////////////

// impl Demosaic<f32, 3> for XTransDemosaic {
//   /// Demosaics an X-Trans image to full resolution using a gradient-based method.
//   /// This version is corrected to handle the non-uniform X-Trans CFA pattern,
//   /// avoiding artifacts by correctly identifying and averaging available neighbor pixels.
//   fn demosaic(&self, pixels: &PixF32, cfa: &CFA, _colors: &PlaneColor, roi: Rect) -> Color2D<f32, 3> {
//     let mut out = Color2D::<f32, 3>::new(roi.width(), roi.height());
//     let cfa = cfa.shift(roi.p.x, roi.p.y);

//     // Pass 1: Copy known sensor values into the correct channels of the output buffer.
//     for y in 0..roi.height() {
//       for x in 0..roi.width() {
//         let color_idx = cfa.color_at(y, x);
//         if color_idx < 3 {
//           // Ensure we only handle R, G, B
//           out.at_mut(y, x)[color_idx] = *pixels.at(roi.p.y + y, roi.p.x + x);
//         }
//       }
//     }

//     // Create a padded copy for easier border handling during interpolation
//     let mut padded = out.make_padded(2);

//     // Pass 2: Interpolate Green channel at R and B locations using gradient detection.
//     for y in 2..padded.height - 2 {
//       for x in 2..padded.width - 2 {
//         let roi_y = y - 2;
//         let roi_x = x - 2;
//         let color_idx = cfa.color_at(roi_y, roi_x);

//         if color_idx == CFA_COLOR_R || color_idx == CFA_COLOR_B {
//           // Gradients of the current color (R or B). In X-Trans, same-colored
//           // neighbors are typically 2 pixels away in cardinal directions.
//           let h_grad = (padded.at(y, x - 2)[color_idx] - padded.at(y, x + 2)[color_idx]).abs();
//           let v_grad = (padded.at(y - 2, x)[color_idx] - padded.at(y + 2, x)[color_idx]).abs();

//           let mut g_h_sum = 0.0;
//           let mut g_h_count = 0;
//           let mut g_v_sum = 0.0;
//           let mut g_v_count = 0;

//           // Find and sum horizontal Green neighbors
//           if roi_x > 0 && cfa.color_at(roi_y, roi_x - 1) == CFA_COLOR_G {
//             g_h_sum += padded.at(y, x - 1)[CFA_COLOR_G];
//             g_h_count += 1;
//           }
//           if cfa.color_at(roi_y, roi_x + 1) == CFA_COLOR_G {
//             g_h_sum += padded.at(y, x + 1)[CFA_COLOR_G];
//             g_h_count += 1;
//           }

//           // Find and sum vertical Green neighbors
//           if roi_y > 0 && cfa.color_at(roi_y - 1, roi_x) == CFA_COLOR_G {
//             g_v_sum += padded.at(y - 1, x)[CFA_COLOR_G];
//             g_v_count += 1;
//           }
//           if cfa.color_at(roi_y + 1, roi_x) == CFA_COLOR_G {
//             g_v_sum += padded.at(y + 1, x)[CFA_COLOR_G];
//             g_v_count += 1;
//           }

//           let g_h = if g_h_count > 0 { g_h_sum / g_h_count as f32 } else { 0.0 };
//           let g_v = if g_v_count > 0 { g_v_sum / g_v_count as f32 } else { 0.0 };

//           let g = if g_h_count == 0 && g_v_count == 0 {
//             0.0 // Fallback, should not happen for R/B in X-Trans
//           } else if g_h_count == 0 {
//             g_v // Only vertical Gs available
//           } else if g_v_count == 0 {
//             g_h // Only horizontal Gs available
//           } else {
//             // Both horizontal and vertical Gs exist, use gradient to decide.
//             if (h_grad - v_grad).abs() < 0.001 { // Gradients are similar
//               (g_h_sum + g_v_sum) / (g_h_count + g_v_count) as f32
//             } else if h_grad < v_grad { // Horizontal edge
//               g_h
//             } else { // Vertical edge
//               g_v
//             }
//           };
//           padded.at_mut(y, x)[CFA_COLOR_G] = g;
//         }
//       }
//     }

//     // Pass 3: Interpolate R/B at G locations
//     for y in 2..padded.height - 2 {
//       for x in 2..padded.width - 2 {
//         let roi_y = y - 2;
//         let roi_x = x - 2;
//         if cfa.color_at(roi_y, roi_x) == CFA_COLOR_G {
//           let mut r_sum = 0.0;
//           let mut r_count = 0;
//           let mut b_sum = 0.0;
//           let mut b_count = 0;

//           // Check cardinal neighbors with boundary checks
//           // Left
//           if roi_x > 0 {
//             match cfa.color_at(roi_y, roi_x - 1) {
//               CFA_COLOR_R => { r_sum += padded.at(y, x - 1)[CFA_COLOR_R]; r_count += 1; }
//               CFA_COLOR_B => { b_sum += padded.at(y, x - 1)[CFA_COLOR_B]; b_count += 1; }
//               _ => {}
//             }
//           }
//           // Right
//           match cfa.color_at(roi_y, roi_x + 1) {
//             CFA_COLOR_R => { r_sum += padded.at(y, x + 1)[CFA_COLOR_R]; r_count += 1; }
//             CFA_COLOR_B => { b_sum += padded.at(y, x + 1)[CFA_COLOR_B]; b_count += 1; }
//             _ => {}
//           }
//           // Top
//           if roi_y > 0 {
//             match cfa.color_at(roi_y - 1, roi_x) {
//               CFA_COLOR_R => { r_sum += padded.at(y - 1, x)[CFA_COLOR_R]; r_count += 1; }
//               CFA_COLOR_B => { b_sum += padded.at(y - 1, x)[CFA_COLOR_B]; b_count += 1; }
//               _ => {}
//             }
//           }
//           // Bottom
//           match cfa.color_at(roi_y + 1, roi_x) {
//             CFA_COLOR_R => { r_sum += padded.at(y + 1, x)[CFA_COLOR_R]; r_count += 1; }
//             CFA_COLOR_B => { b_sum += padded.at(y + 1, x)[CFA_COLOR_B]; b_count += 1; }
//             _ => {}
//           }

//           if r_count > 0 {
//             padded.at_mut(y, x)[CFA_COLOR_R] = r_sum / r_count as f32;
//           }
//           if b_count > 0 {
//             padded.at_mut(y, x)[CFA_COLOR_B] = b_sum / b_count as f32;
//           }
//         }
//       }
//     }

//     // Pass 4: Interpolate R at B and B at R
//     for y in 2..padded.height - 2 {
//       for x in 2..padded.width - 2 {
//         let roi_y = y - 2;
//         let roi_x = x - 2;
//         let color_idx = cfa.color_at(roi_y, roi_x);

//         if color_idx == CFA_COLOR_R || color_idx == CFA_COLOR_B {
//           let mut r_sum = 0.0;
//           let mut b_sum = 0.0;
//           let mut g_neighbor_count = 0;

//           // Interpolate using diagonal neighbors with boundary checks.
//           // We only use G neighbors, which now have interpolated R and B values from Pass 3.
          
//           // Top-Left
//           if roi_y > 0 && roi_x > 0 && cfa.color_at(roi_y - 1, roi_x - 1) == CFA_COLOR_G {
//             r_sum += padded.at(y - 1, x - 1)[CFA_COLOR_R];
//             b_sum += padded.at(y - 1, x - 1)[CFA_COLOR_B];
//             g_neighbor_count += 1;
//           }
//           // Top-Right
//           if roi_y > 0 && cfa.color_at(roi_y - 1, roi_x + 1) == CFA_COLOR_G {
//             r_sum += padded.at(y - 1, x + 1)[CFA_COLOR_R];
//             b_sum += padded.at(y - 1, x + 1)[CFA_COLOR_B];
//             g_neighbor_count += 1;
//           }
//           // Bottom-Left
//           if roi_x > 0 && cfa.color_at(roi_y + 1, roi_x - 1) == CFA_COLOR_G {
//             r_sum += padded.at(y + 1, x - 1)[CFA_COLOR_R];
//             b_sum += padded.at(y + 1, x - 1)[CFA_COLOR_B];
//             g_neighbor_count += 1;
//           }
//           // Bottom-Right
//           if cfa.color_at(roi_y + 1, roi_x + 1) == CFA_COLOR_G {
//             r_sum += padded.at(y + 1, x + 1)[CFA_COLOR_R];
//             b_sum += padded.at(y + 1, x + 1)[CFA_COLOR_B];
//             g_neighbor_count += 1;
//           }

//           if g_neighbor_count > 0 {
//             if color_idx == CFA_COLOR_B {
//               // Interpolate R at B locations
//               padded.at_mut(y, x)[CFA_COLOR_R] = r_sum / g_neighbor_count as f32;
//             } else { // color_idx == CFA_COLOR_R
//               // Interpolate B at R locations
//               padded.at_mut(y, x)[CFA_COLOR_B] = b_sum / g_neighbor_count as f32;
//             }
//           }
//         }
//       }
//     }

//     // Crop the padding off to return the final image
//     padded.crop(Rect::new_with_points(
//       Point::new(2, 2),
//       Point::new(padded.width - 2, padded.height - 2),
//     ))
//   }
// }

#[derive(Default)]
pub struct XTransSuperpixelDemosaic {}

impl XTransSuperpixelDemosaic {
  pub fn new() -> Self {
    Self {}
  }
}

impl Demosaic<f32, 3> for XTransSuperpixelDemosaic {
  /// Debayer an X-Trans image using a 6x6 superpixel method.
  /// Each output RGB pixel is the average of all R, G, and B pixels
  /// within a 6x6 block of the original sensor data.
  /// The resulting image is 1/36th the size (1/6 width, 1/6 height).
  fn demosaic(&self, pixels: &PixF32, cfa: &CFA, colors: &PlaneColor, roi: Rect) -> Color2D<f32, 3> {
    // ROI width/height must be a multiple of 6 for this algorithm.
    let roi = Rect::new(roi.p, Dim2::new(roi.width() / 6 * 6, roi.height() / 6 * 6));
    let dim = pixels.dim();

    // The CFA pattern must be shifted according to the ROI's top-left corner.
    let cfa = cfa.shift(roi.p.x, roi.p.y);

    // This lookup table maps a CFAColor (R,G,B) to its correct output channel index (0,1,2).
    let plane_map = colors.plane_lookup_table();

    // Get a slice of the image corresponding to the ROI's starting row.
    let window = &pixels[roi.y() * dim.w..];

    let out_data: Vec<[f32; 3]> = window
      .par_chunks_exact(dim.w * 6) // Process 6 rows at a time
      .take(roi.height() / 6) // Process roi.height() / 6 blocks of rows
      .flat_map(|six_rows_slice| {
        // six_rows_slice contains 6 full rows of the original image.
        // We process them in 6-pixel wide chunks.
        let rows: Vec<_> = (0..6)
          .map(|i| &six_rows_slice[i * dim.w + roi.x()..i * dim.w + roi.x() + roi.width()])
          .collect();

        (0..roi.width() / 6)
          .map(|block_x| {
            let mut sums = [0.0f32; 3];
            let mut counts = [0u32; 3];

            for y_offset in 0..6 {
              for x_offset in 0..6 {
                // Get the color (R, G, or B) at this position from the shifted CFA.
                let cfa_color_val = cfa.color_at(y_offset, x_offset);
                if cfa_color_val < 3 {
                  // Ensure it's R, G, or B
                  // Use the plane_map to find the correct output channel for this color.
                  let plane_index = plane_map[cfa_color_val];
                  if plane_index < 3 {
                    let pixel_value = rows[y_offset][block_x * 6 + x_offset];
                    sums[plane_index] += pixel_value;
                    counts[plane_index] += 1;
                  }
                }
              }
            }

            // The sums are now in the correct R, G, B order thanks to plane_map.
            let r = if counts[0] > 0 { sums[0] / counts[0] as f32 } else { 0.0 };
            let g = if counts[1] > 0 { sums[1] / counts[1] as f32 } else { 0.0 };
            let b = if counts[2] > 0 { sums[2] / counts[2] as f32 } else { 0.0 };

            [r, g, b]
          })
          .collect::<Vec<_>>()
      })
      .collect();

    Color2D::new_with(out_data, roi.width() / 6, roi.height() / 6)
  }
}
