import torch


def rowswap(A, source, target):
    # Copy the matrix and swap the selected rows
    result = A.clone()
    result[[source, target]] = result[[target, source]]
    return result


def rowscale(A, source, factor):
    # Use floating-point values to allow fractional scaling
    result = A.clone().to(dtype=torch.float64)

    # Multiply the selected row by the factor
    result[source] = factor * result[source]
    return result


def rowreplacement(A, first, second, j, k):
    # Replace the second row with j*first row + k*second row
    result = A.clone().to(dtype=torch.float64)
    result[second] = j * result[first] + k * result[second]
    return result


def rref(A, tolerance=1e-10):
    # Return row echelon form with pivots equal to 1
    # The name rref is retained as requested in the assignment
    result = A.clone().to(dtype=torch.float64)
    number_of_rows, number_of_columns = result.shape
    pivot_row = 0

    for column in range(number_of_columns):
        # Stop when every available row has a pivot
        if pivot_row == number_of_rows:
            break

        # Find a nonzero entry at or below the pivot row
        selected_row = None

        for row in range(pivot_row, number_of_rows):
            if abs(result[row, column].item()) > tolerance:
                selected_row = row
                break

        # Skip this column if no nonzero entry is found
        if selected_row is None:
            continue

        # Move the selected row to the pivot position
        if selected_row != pivot_row:
            result = rowswap(result, pivot_row, selected_row)

        # Make the pivot equal to 1
        factor = 1.0 / result[pivot_row, column].item()
        result = rowscale(result, pivot_row, factor)

        # Make all entries below the pivot zero
        for row in range(pivot_row + 1, number_of_rows):
            factor = -result[row, column].item()
            result = rowreplacement(
                result, pivot_row, row, factor, 1.0
            )

        # Move to the next pivot row
        pivot_row += 1

    # Replace tiny numerical residues with zero
    result[torch.abs(result) <= tolerance] = 0.0
    return result


if __name__ == "__main__":
    # Display tensors with four decimal places
    torch.set_printoptions(precision=4, sci_mode=False)

    # Define the matrix given in the question
    A = torch.tensor([
        [1, 3, 0, 0, 3],
        [0, 0, 1, 0, 9],
        [0, 0, 0, 1, -4]
    ], dtype=torch.float64)

    print("Original matrix:")
    print(A)

    # Swap R1 and R2; Python indices start at 0
    A1 = rowswap(A, 0, 1)
    print("\nAfter swapping R1 and R2:")
    print(A1)

    # Scale R1 of the resulting matrix by 1/3
    A2 = rowscale(A1, 0, 1 / 3)
    print("\nAfter scaling R1 by 1/3:")
    print(A2)

    # Replace R3 with -3R1 + R3
    A3 = rowreplacement(A2, 0, 2, -3, 1)
    print("\nAfter replacing R3 with -3R1 + R3:")
    print(A3)

    # Demonstrate the row echelon function
    print("\nRow echelon form:")
    print(rref(A3))
