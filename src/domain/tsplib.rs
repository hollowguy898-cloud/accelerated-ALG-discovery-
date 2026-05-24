// src/domain/tsplib.rs
// TSPLIB File Parser for Real Benchmark Instances
//
// Parses the standard TSPLIB95 format (.tsp files) including:
// - EUC_2D: 2D Euclidean distance (rounded to nearest integer)
// - CEIL_2D: 2D Euclidean distance (ceiling)
// - GEO: Geographical distance (TSPLIB geographic formula)
// - ATT: ATT distance (pseudo-Euclidean with ceiling)
// - EXPLICIT: Full distance matrix given in the file
// - LOWER_DIAG_ROW, UPPER_ROW, UPPER_DIAG_ROW, FULL_MATRIX subsections
//
// Reference: https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf

use crate::domain::City;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

/// A parsed TSPLIB instance with all metadata.
#[derive(Clone, Debug)]
pub struct TsplibInstance {
    /// Instance name (from NAME field)
    pub name: String,
    /// Problem type (always "TSP" for our purposes)
    pub problem_type: String,
    /// Number of cities/nodes
    pub dimension: usize,
    /// Edge weight type (EUC_2D, GEO, ATT, CEIL_2D, EXPLICIT)
    pub edge_weight_type: String,
    /// Edge weight format (for EXPLICIT type)
    pub edge_weight_format: Option<String>,
    /// City coordinates (only for coordinate-based types)
    pub cities: Vec<City>,
    /// Distance matrix (computed or read directly)
    pub matrix: Vec<Vec<f64>>,
    /// Known optimal tour length (if provided in .opt.tour file)
    pub optimal: Option<f64>,
}

impl TsplibInstance {
    /// Parse a TSPLIB .tsp file from disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path.as_ref(), e))?;
        Self::from_str(&content)
    }

    /// Parse a TSPLIB instance from a string.
    pub fn from_str(content: &str) -> Result<Self, String> {
        let mut name = String::new();
        let mut problem_type = String::from("TSP");
        let mut dimension = 0usize;
        let mut edge_weight_type = String::new();
        let mut edge_weight_format = None;
        let mut cities = Vec::new();
        let mut explicit_matrix = Vec::new();

        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.is_empty() {
                i += 1;
                continue;
            }

            // Parse keyword: value pairs (header section)
            if let Some(colon_pos) = line.find(':') {
                let key = line[..colon_pos].trim().to_uppercase();
                let value = line[colon_pos + 1..].trim();

                match key.as_str() {
                    "NAME" => name = value.to_string(),
                    "TYPE" => problem_type = value.to_uppercase(),
                    "DIMENSION" => {
                        dimension = value.parse::<usize>()
                            .map_err(|e| format!("Invalid DIMENSION '{}': {}", value, e))?;
                    }
                    "EDGE_WEIGHT_TYPE" => edge_weight_type = value.to_uppercase(),
                    "EDGE_WEIGHT_FORMAT" => edge_weight_format = Some(value.to_uppercase()),
                    _ => {} // Ignore COMMENT, DISPLAY_DATA_TYPE, etc.
                }
                i += 1;
                continue;
            }

            // Parse coordinate section
            if line.to_uppercase().starts_with("NODE_COORD_SECTION") {
                if dimension == 0 {
                    return Err("DIMENSION must be specified before NODE_COORD_SECTION".into());
                }
                i += 1;
                cities = Vec::with_capacity(dimension);
                while i < lines.len() && cities.len() < dimension {
                    let coord_line = lines[i].trim();
                    if coord_line.is_empty() || coord_line.to_uppercase().starts_with("EOF")
                        || coord_line.to_uppercase().contains("SECTION") {
                        break;
                    }
                    let parts: Vec<&str> = coord_line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let x = parts[1].parse::<f64>()
                            .map_err(|e| format!("Invalid x coordinate: {}", e))?;
                        let y = parts[2].parse::<f64>()
                            .map_err(|e| format!("Invalid y coordinate: {}", e))?;
                        cities.push(City { x, y });
                    }
                    i += 1;
                }
                continue;
            }

            // Parse explicit matrix section
            if line.to_uppercase().starts_with("EDGE_WEIGHT_SECTION") {
                if dimension == 0 {
                    return Err("DIMENSION must be specified before EDGE_WEIGHT_SECTION".into());
                }
                i += 1;
                let mut all_values: Vec<f64> = Vec::with_capacity(dimension * dimension);
                while i < lines.len() && all_values.len() < dimension * dimension {
                    let row_line = lines[i].trim();
                    if row_line.is_empty() || row_line.to_uppercase().starts_with("EOF")
                        || row_line.to_uppercase().contains("SECTION") {
                        break;
                    }
                    for val_str in row_line.split_whitespace() {
                        if let Ok(v) = val_str.parse::<f64>() {
                            all_values.push(v);
                        }
                    }
                    i += 1;
                }

                explicit_matrix = match edge_weight_format.as_deref() {
                    Some("FULL_MATRIX") => {
                        if all_values.len() != dimension * dimension {
                            return Err(format!(
                                "FULL_MATRIX: expected {} values, got {}",
                                dimension * dimension, all_values.len()
                            ));
                        }
                        let mut m = vec![vec![0.0; dimension]; dimension];
                        for r in 0..dimension {
                            for c in 0..dimension {
                                m[r][c] = all_values[r * dimension + c];
                            }
                        }
                        m
                    }
                    Some("UPPER_ROW") => {
                        // Upper triangular row by row, excluding diagonal
                        // Row 0: elements (0,1), (0,2), ..., (0,n-1)
                        // Row 1: elements (1,2), ..., (1,n-1)
                        // etc.
                        let mut m = vec![vec![0.0; dimension]; dimension];
                        let mut idx = 0;
                        for r in 0..dimension {
                            for c in (r + 1)..dimension {
                                if idx < all_values.len() {
                                    m[r][c] = all_values[idx];
                                    m[c][r] = all_values[idx];
                                    idx += 1;
                                }
                            }
                        }
                        m
                    }
                    Some("UPPER_DIAG_ROW") => {
                        // Upper triangular including diagonal
                        let mut m = vec![vec![0.0; dimension]; dimension];
                        let mut idx = 0;
                        for r in 0..dimension {
                            for c in r..dimension {
                                if idx < all_values.len() {
                                    m[r][c] = all_values[idx];
                                    m[c][r] = all_values[idx];
                                    idx += 1;
                                }
                            }
                        }
                        m
                    }
                    Some("LOWER_DIAG_ROW") => {
                        // Lower triangular including diagonal
                        let mut m = vec![vec![0.0; dimension]; dimension];
                        let mut idx = 0;
                        for r in 0..dimension {
                            for c in 0..=r {
                                if idx < all_values.len() {
                                    m[r][c] = all_values[idx];
                                    m[c][r] = all_values[idx];
                                    idx += 1;
                                }
                            }
                        }
                        m
                    }
                    Some("LOWER_ROW") => {
                        // Lower triangular excluding diagonal
                        let mut m = vec![vec![0.0; dimension]; dimension];
                        let mut idx = 0;
                        for r in 1..dimension {
                            for c in 0..r {
                                if idx < all_values.len() {
                                    m[r][c] = all_values[idx];
                                    m[c][r] = all_values[idx];
                                    idx += 1;
                                }
                            }
                        }
                        m
                    }
                    _ => {
                        return Err(format!(
                            "Unsupported EDGE_WEIGHT_FORMAT: {:?}",
                            edge_weight_format
                        ));
                    }
                };
                continue;
            }

            // Skip other sections we don't need
            if line.to_uppercase().starts_with("DISPLAY_DATA_SECTION")
                || line.to_uppercase().starts_with("DEPOT_SECTION")
                || line.to_uppercase().starts_with("DEMAND_SECTION")
                || line.to_uppercase().starts_with("FIXED_EDGES_SECTION")
                || line.to_uppercase().starts_with("TOUR_SECTION") {
                i += 1;
                // Skip until next section or EOF
                while i < lines.len() {
                    let skip = lines[i].trim();
                    if skip.is_empty() || skip.to_uppercase().starts_with("EOF") {
                        break;
                    }
                    // Check for next section keyword
                    let upper = skip.to_uppercase();
                    if upper.contains("SECTION") || upper.contains(':') && !upper.starts_with(|c: char| c.is_ascii_digit()) {
                        break;
                    }
                    i += 1;
                }
                continue;
            }

            i += 1;
        }

        if dimension == 0 {
            return Err("No DIMENSION specified in TSPLIB file".into());
        }
        if edge_weight_type.is_empty() && explicit_matrix.is_empty() {
            return Err("No EDGE_WEIGHT_TYPE or EDGE_WEIGHT_SECTION specified".into());
        }

        // Compute distance matrix from coordinates if needed
        let matrix = if !explicit_matrix.is_empty() {
            explicit_matrix
        } else {
            compute_distance_matrix(&cities, &edge_weight_type, dimension)?
        };

        Ok(TsplibInstance {
            name,
            problem_type,
            dimension,
            edge_weight_type,
            edge_weight_format,
            cities,
            matrix,
            optimal: None,
        })
    }

    /// Load the optimal tour length from a .opt.tour or .tsp.opt.tour file.
    pub fn load_optimal_tour_length(&mut self, tour_path: &str) -> Result<f64, String> {
        let content = fs::read_to_string(tour_path)
            .map_err(|e| format!("Failed to read tour file: {}", e))?;

        let mut tour: Vec<usize> = Vec::new();
        let mut in_tour_section = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.to_uppercase() == "TOUR_SECTION" {
                in_tour_section = true;
                continue;
            }
            if trimmed.to_uppercase() == "EOF" || trimmed.to_uppercase().starts_with("-1") {
                break;
            }
            if in_tour_section {
                if let Ok(city) = trimmed.parse::<usize>() {
                    if city > 0 {
                        tour.push(city - 1); // TSPLIB is 1-indexed
                    }
                }
            }
        }

        if tour.is_empty() {
            return Err("No tour found in the optimal tour file".into());
        }

        // Compute tour length
        let mut total = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total += self.matrix[from][to];
        }

        self.optimal = Some(total);
        Ok(total)
    }

    /// Try to find and load an optimal tour file adjacent to the .tsp file.
    pub fn try_load_optimal(&mut self, tsp_path: &str) -> Option<f64> {
        let candidates = [
            tsp_path.replace(".tsp", ".opt.tour"),
            format!("{}.opt.tour", tsp_path),
            tsp_path.replace(".tsp", ".opt.tour.tsp"),
        ];

        for candidate in &candidates {
            if Path::new(candidate).exists() {
                if let Ok(opt) = self.load_optimal_tour_length(candidate) {
                    return Some(opt);
                }
            }
        }
        None
    }
}

/// Compute the distance matrix based on the edge weight type.
fn compute_distance_matrix(
    cities: &[City],
    edge_weight_type: &str,
    dimension: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if cities.len() != dimension {
        return Err(format!(
            "Expected {} cities, got {} coordinates",
            dimension,
            cities.len()
        ));
    }

    let mut matrix = vec![vec![0.0; dimension]; dimension];

    for i in 0..dimension {
        for j in 0..dimension {
            if i == j {
                matrix[i][j] = 0.0;
            } else {
                matrix[i][j] = compute_distance(&cities[i], &cities[j], edge_weight_type)?;
            }
        }
    }

    Ok(matrix)
}

/// Compute distance between two cities according to TSPLIB edge weight type.
fn compute_distance(a: &City, b: &City, edge_weight_type: &str) -> Result<f64, String> {
    match edge_weight_type {
        "EUC_2D" => {
            // Euclidean distance rounded to nearest integer
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            Ok((dx * dx + dy * dy).sqrt().round())
        }
        "CEIL_2D" => {
            // Euclidean distance with ceiling
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            Ok((dx * dx + dy * dy).sqrt().ceil())
        }
        "GEO" => {
            // Geographical distance using TSPLIB formula
            Ok(geo_distance(a, b))
        }
        "ATT" => {
            // Pseudo-Euclidean (ATT) distance
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let rij = ((dx * dx + dy * dy) / 10.0).sqrt();
            let tij = rij.round();
            Ok(if tij < rij { tij + 1.0 } else { tij })
        }
        "MAN_2D" => {
            // Manhattan distance
            Ok((a.x - b.x).abs() + (a.y - b.y).abs())
        }
        _ => Err(format!("Unsupported EDGE_WEIGHT_TYPE: {}", edge_weight_type)),
    }
}

/// Compute geographical distance between two cities using the TSPLIB GEO formula.
///
/// This uses the latitude/longitude interpretation where coordinates are
/// in DDD.MM format (degrees.minutes). The formula computes great-circle
/// distance using the TSPLIB-specific constants.
fn geo_distance(a: &City, b: &City) -> f64 {
    let pi = std::f64::consts::PI;
    let deg_to_min = |x: f64| {
        let deg = x.trunc();
        let min = x - deg;
        deg + min / 60.0
    };

    let lat_a = deg_to_min(a.y) * pi / 180.0;
    let lon_a = deg_to_min(a.x) * pi / 180.0;
    let lat_b = deg_to_min(b.y) * pi / 180.0;
    let lon_b = deg_to_min(b.x) * pi / 180.0;

    let q1 = (lon_a - lon_b).cos();
    let q2 = (lat_a - lat_b).cos();
    let q3 = (lat_a + lat_b).cos();

    let radius = 6378.388; // TSPLIB earth radius in km
    let dist = radius * (((1.0 - q1) * (1.0 - q1) * (1.0 - q2 + (1.0 + q2) / 2.0))
        + (1.0 + q1) * (1.0 + q1) * (1.0 - q3 + (1.0 + q3) / 2.0))
        .abs()
        .sqrt()
        / 2.0;

    dist.round()
}

/// Known optimal tour lengths for common TSPLIB instances.
/// Source: TSPLIB95 documentation and published results.
pub fn known_optimal(instance_name: &str) -> Option<f64> {
    let opt: f64 = match instance_name.to_uppercase().as_str() {
        // Small symmetric TSP instances
        "BERLIN52" => 7542.0,
        "BAYG29" => 1610.0,
        "BAYS29" => 2020.0,
        "BURMA14" => 3323.0,
        "DANTZIG42" => 699.0,
        "FRI26" => 937.0,
        "GR17" => 2085.0,
        "GR21" => 2707.0,
        "GR24" => 1272.0,
        "GR48" => 5046.0,
        "GR96" => 55209.0,
        "HK48" => 11461.0,
        "PR76" => 108159.0,
        "RD100" => 7910.0,
        "ST70" => 675.0,
        "OLIVER30" => 423.740552,
        "ATT48" => 10628.0,
        "ATT532" => 27686.0,
        // Medium instances
        "KROA100" => 21282.0,
        "KROB100" => 22141.0,
        "KROC100" => 20749.0,
        "KROD100" => 21294.0,
        "KROE100" => 22068.0,
        "CH130" => 6110.0,
        "CH150" => 6528.0,
        "EIL51" => 426.0,
        "EIL76" => 538.0,
        "EIL101" => 629.0,
        "LIN105" => 14379.0,
        "LIN318" => 42029.0,
        "PCB442" => 50778.0,
        "PR107" => 44303.0,
        "PR124" => 59030.0,
        "PR136" => 96772.0,
        "PR144" => 58537.0,
        "PR152" => 73682.0,
        "PR226" => 80369.0,
        "PR264" => 49135.0,
        "PR299" => 48191.0,
        "PR439" => 107217.0,
        "QUAN150" => 1131.0,
        "RAT99" => 1211.0,
        "RAT195" => 2323.0,
        "U159" => 42080.0,
        // Large instances
        "PR1002" => 259045.0,
        "PR2392" => 378032.0,
        "FL1577" => 22249.0,
        "FL3797" => 28772.0,
        "FNL4461" => 182566.0,
        "RL5934" => 565530.0,
        "RL11849" => 923288.0,
        "PLA7397" => 23260728.0,
        "PLA85900" => 142382641.0,
        "USA13509" => 19982859.0,
        // VLSI instances
        "XQF131" => 566.0,
        "XQG237" => 1019.0,
        _ => return None,
    };
    Some(opt)
}

/// Download a TSPLIB instance from the standard repository.
///
/// Attempts to download from multiple mirror sources.
pub fn download_instance(name: &str, target_dir: &str) -> Result<String, String> {
    let name_upper = name.to_uppercase();
    let filename = format!("{}.tsp", name_upper);
    let target_path = format!("{}/{}", target_dir, filename);

    // Check if already downloaded
    if Path::new(&target_path).exists() {
        return Ok(target_path);
    }

    // Try downloading from university mirrors
    let urls = [
        format!("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{}.tsp", name_upper),
        format!("https://raw.githubusercontent.com/mastqe/tsplib/master/{}.tsp", name_upper),
    ];

    for url in &urls {
        // Use curl to download
        let output = std::process::Command::new("curl")
            .args(&["-sL", "-o", &target_path, url])
            .output();

        match output {
            Ok(status) if status.status.success() => {
                // Verify the file is valid TSPLIB
                if let Ok(content) = fs::read_to_string(&target_path) {
                    if content.contains("NAME") && content.contains("DIMENSION") {
                        return Ok(target_path);
                    }
                }
                // Invalid file, remove it
                let _ = fs::remove_file(&target_path);
            }
            _ => {
                let _ = fs::remove_file(&target_path);
            }
        }
    }

    Err(format!("Failed to download TSPLIB instance '{}' from any mirror", name_upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_berlin52_format() {
        // Minimal valid TSPLIB file
        let content = r#"
NAME : berlin52
COMMENT : 52 locations in Berlin (Groetschel)
TYPE : TSP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
4 945.0 685.0
5 845.0 655.0
EOF
"#;
        let instance = TsplibInstance::from_str(content).unwrap();
        assert_eq!(instance.name, "berlin52");
        assert_eq!(instance.dimension, 5);
        assert_eq!(instance.edge_weight_type, "EUC_2D");
        assert_eq!(instance.cities.len(), 5);
        assert_eq!(instance.matrix.len(), 5);

        // Check symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert!((instance.matrix[i][j] - instance.matrix[j][i]).abs() < 0.01);
            }
        }
        // Check diagonal is zero
        for i in 0..5 {
            assert!(instance.matrix[i][i].abs() < 0.01);
        }
    }

    #[test]
    fn test_euc_2d_rounding() {
        // EUC_2D should round to nearest integer
        let cities = vec![
            City { x: 0.0, y: 0.0 },
            City { x: 3.0, y: 4.0 },  // distance = 5.0
            City { x: 1.0, y: 1.0 },  // distance = sqrt(2) ≈ 1.414, rounds to 1
        ];
        let matrix = compute_distance_matrix(&cities, "EUC_2D", 3).unwrap();
        assert_eq!(matrix[0][1], 5.0);
        assert_eq!(matrix[0][2], 1.0); // sqrt(2) rounds to 1
    }

    #[test]
    fn test_att_distance() {
        let cities = vec![
            City { x: 0.0, y: 0.0 },
            City { x: 30.0, y: 40.0 },
        ];
        let matrix = compute_distance_matrix(&cities, "ATT", 2).unwrap();
        // ATT: rij = sqrt((30^2 + 40^2)/10) = sqrt(250) ≈ 15.81
        // tij = round(15.81) = 16
        // 16 > 15.81, so distance = 16
        assert_eq!(matrix[0][1], 16.0);
    }

    #[test]
    fn test_known_optimals() {
        assert_eq!(known_optimal("berlin52"), Some(7542.0));
        assert_eq!(known_optimal("kroA100"), Some(21282.0));
        assert_eq!(known_optimal("pr2392"), Some(378032.0));
        assert_eq!(known_optimal("nonexistent"), None);
    }
}
