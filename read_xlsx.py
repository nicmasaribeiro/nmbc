# Load a specific sheet by name
df_sheet1 = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# Load a specific sheet by index (e.g., first sheet)
df_sheet1_by_index = pd.read_excel('file.xlsx', sheet_name=0)

# Load multiple sheets by providing a list of sheet names or indices
dfs = pd.read_excel('file.xlsx', sheet_name=['Sheet1', 'Sheet2'])

# Access data from each sheet in the dictionary
df_sheet1 = dfs['Sheet1']
df_sheet2 = dfs['Sheet2']

# Load all sheets at once (returns a dictionary with sheet names as keys)
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)

# Access individual sheet data
df_sheet1 = all_sheets['Sheet1']