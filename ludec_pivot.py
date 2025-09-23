# LU decomposition with partial pivoting
def ludec_pivot(A_in):
    A = np.copy(A_in).astype(float)
    n = A.shape[0]
    L = np.identity(n)
    U = np.copy(A)
    P = np.identity(n)

    for j in range(n-1):
        pivot_row = j + np.argmax(np.abs(U[j:, j]))
        if np.isclose(U[pivot_row, j], 0):
            raise ZeroDivisionError(f"No nonzero pivot found at column {j}.")
        if pivot_row != j:
            U[[j, pivot_row]] = U[[pivot_row, j]]
            P[[j, pivot_row]] = P[[pivot_row, j]]
            if j > 0:
                L[[j, pivot_row], :j] = L[[pivot_row, j], :j]
        for i in range(j+1, n):
            coeff = U[i, j] / U[j, j]
            U[i, j:] -= coeff * U[j, j:]
            L[i, j] = coeff
    return P, L, U

def lusolve_pivot(A, b):
    P, L, U = ludec_pivot(A)
    b_permuted = P @ b
    y = forsub(L, b_permuted)
    x = backsub(U, y)
    return x