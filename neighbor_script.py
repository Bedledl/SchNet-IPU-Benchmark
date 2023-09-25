import sys
from ase.io.proteindatabank import read_proteindatabank
from ase.neighborlist import build_neighbor_list
from torch import full as torch_full
from torch import tensor, max, median, mean

def main(filename):
    # Read .pdb file using ASE
    atoms = read_proteindatabank(filename, index=0)
    
    # Set cutoff distance to 5.0 for each atom
    cutoffs = torch_full((len(atoms),), 0.5)
    
    # Get neighbor list
    neighbor_list = build_neighbor_list(atoms, cutoffs=cutoffs)
    neighbor_count = tensor([len(neighbor_list.get_neighbors(i)[0]) for i in range(len(atoms))])

    # Calculate average, median, and maximum numbers of neighbors
    avg_neighbors = mean(neighbor_count.double())
    median_neighbors = median(neighbor_count)
    max_neighbors = max(neighbor_count)
    
    print(f"Average number of neighbors: {avg_neighbors}")
    print(f"Median number of neighbors: {median_neighbors}")
    print(f"Maximum number of neighbors: {max_neighbors}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdb_file>")
        sys.exit(1)
        
    filename = sys.argv[1]
    main(filename)

