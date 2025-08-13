def solve():
    # Read input
    n = int(input())
    grid = [input().strip() for _ in range(n)]
    
    # DP table: 0 for right, 1 for down, -1 for uncomputed
    dp = [[-1] * n for _ in range(n)]
    
    # Function to decide the next move at (i,j)
    def decide_move(i, j):
        # Boundary cases: only one move possible
        if i == n-1:
            return 0  # Must move right
        if j == n-1:
            return 1  # Must move down
        
        # Compare immediate next cells
        right = grid[i][j+1]  # Next cell if moving right
        down = grid[i+1][j]   # Next cell if moving down
        
        if right < down:
            return 0
        elif down < right:
            return 1
        else:
            # If equal, look ahead until paths differ
            k = 1
            while True:
                # If one path ends, prefer the shorter or equal path
                if j + k >= n:
                    return 1  # Down is shorter or equal
                if i + k >= n:
                    return 0  # Right is shorter or equal
                # Compare characters along the paths
                r_char = grid[i][j + k]    # Right path
                d_char = grid[i + k][j]    # Down path
                if r_char != d_char:
                    return 0 if r_char < d_char else 1
                k += 1
    
    # Fill DP table from bottom-right to top-left
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            if i == n-1 and j == n-1:
                continue  # No decision needed at end
            dp[i][j] = decide_move(i, j)
    
    # Reconstruct the path and build the string
    path = []
    i, j = 0, 0
    while i < n-1 or j < n-1:
        path.append(grid[i][j])
        if dp[i][j] == 0:
            j += 1  # Move right
        else:
            i += 1  # Move down
    path.append(grid[n-1][n-1])  # Add final cell
    
    # Output the result
    print(''.join(path))

# Execute the solution
solve()