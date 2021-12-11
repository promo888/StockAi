l = [
    ['o', 'o', 'x', 'o', 'o'],
    ['o', 'x', 'o', 'x', 'o'],
    ['o', 'o', 'x', 'o', 'o'],
    ['o', 'o', 'o', 'x', 'o'],
    ['x', 'o', 'o', 'x', 'x'],
]
shared_x = set() #given we are numbering rows * cols from left to right
all_x = set()
checked_cells = set()


def getCellId(rownum, cellnum, total_cells):
    id = cellnum + 1 + rownum*total_cells
    checked_cells.add(id)
    return id

def addShared(current_cell_id, shared_cell_id):
    #checked_cells.add(current_cell_id)
    #checked_cells.add(shared_cell_id)
    if not current_cell_id in shared_x:
        shared_x.add(current_cell_id) #uniq insert
    if not shared_cell_id in shared_x:
        shared_x.add(shared_cell_id) #uniq insert for connection of 2 points

check_count = 0
row_count = len(l)
for r in range(row_count):
    #print(f"row{r} {len(l[r])}cols")
    col_count = len(l[r])
    for c in range(col_count):
        #print(f"row:{r} col:{c} val:{l[r][c]}")
        current_cell_id = getCellId(r, c, col_count) #(c + 1) + r*col_count #numbering cols from 1 to 25
        # print(f"Current ID: {current_cell_id}")
        if current_cell_id in checked_cells:
            next
        checked_cells.add(current_cell_id)

        #global check_count
        check_count += 1
        if l[r][c] == 'x':
            all_x.add(current_cell_id)
            if l[r][c] == 'x' and c+1<col_count and l[r][c+1] == 'x':
                addShared(current_cell_id, getCellId(r, c+1, col_count))
            elif l[r][c] == 'x' and c-1>=0 and l[r][c-1] == 'x': #left cell
                addShared(current_cell_id, getCellId(r, c-1, col_count))
            elif l[r][c] == 'x' and c+1<col_count and l[r][c+1] == 'x': #right cell
                addShared(current_cell_id, getCellId(r, c-1, col_count))
            elif l[r][c] == 'x' and r-1>=0 and l[r-1][c] == 'x': #up cell
                addShared(current_cell_id, getCellId(r-1, c, col_count))
            elif l[r][c] == 'x' and r-1>=0 and c-1>=0 and l[r-1][c] == 'x': #up left cell
                addShared(current_cell_id, getCellId(r-1, c-1, col_count))
            elif l[r][c] == 'x' and r-1>=0 and c+1<col_count and l[r-1][c+1] == 'x': #up right cell
                addShared(current_cell_id, getCellId(r-1, c+1, col_count))
            elif l[r][c] == 'x' and r+1<row_count and l[r+1][c] == 'x': #down cell
                addShared(current_cell_id, getCellId(r+1, c, col_count))
            elif l[r][c] == 'x' and r+1<row_count and c-1>=0 and l[r+1][c-1] == 'x': #down left cell
                addShared(current_cell_id, getCellId(r+1, c-1, col_count))
            elif l[r][c] == 'x' and r+1<row_count and c+1<col_count and l[r + 1][c - 1] == 'x':  # down left cell
                addShared(current_cell_id, getCellId(r+1, c+1, col_count))

print(f"{len(set(shared_x))} cells shared found from total {len(all_x)} Xs\n \
        sharedIds:\n{set(shared_x)}\n \
        {len(set(checked_cells))} cells checked \n"
      f"{check_count} loop checks")


l = [{"id": 0, "connected_vertices": [3, 1]},
    {"id": 0, "connected_vertices": [0, 2]},
    {"id": 2, "connected_vertices": [1, 3]},
    {"id": 3, "connected_vertices": [2, 0]}
]
for i in range(len(l)-1):
      vert_value_left = l[i]["connected_vertices"][0]
      vert_value_right = l[i+1]["connected_vertices"][0]
      if (vert_value_left > vert_value_right):
           l[i], l[i+1] = l[i+1], l[i] #vert_value_right, vert_value_left #swap indexes

print(l)