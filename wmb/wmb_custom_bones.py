
def encode_parts_index_no_table(parts_no_index_map):
    # Credits to Skyth
    l1_table = [0xFFFF] * 16
    l2_tables = []
    l3_tables = []

    l2_index_map = {}
    l3_index_map = {}
    l3_data = {}

    for parts_no, parts_index in parts_no_index_map:
        l1 = (parts_no >> 8) & 0xF
        l2 = (parts_no >> 4) & 0xF
        l3 = parts_no & 0xF

        if l1 not in l2_index_map:
            l2_index_map[l1] = len(l2_tables)
            l2_tables.append([0xFFFF] * 16)

        l2_table = l2_tables[l2_index_map[l1]]

        l3_key = (l1, l2)
        if l3_key not in l3_index_map:
            l3_index_map[l3_key] = len(l3_tables)
            l3_data[l3_key] = [0xFFF] * 16
            l3_tables.append(l3_data[l3_key])

        parts_indices = l3_data[l3_key]
        if parts_indices[l3] != 0xFFF:
            raise ValueError("{} was somehow already added to the table!".format(parts_no))

        parts_indices[l3] = parts_index

    l2_offset_base = 16
    l3_offset_base = l2_offset_base + len(l2_tables) * 16

    for l1, l2_idx in l2_index_map.items():
        l1_table[l1] = l2_offset_base + l2_idx * 16

    for (l1, l2), l3_idx in l3_index_map.items():
        l2_table = l2_tables[l2_index_map[l1]]
        l2_table[l2] = l3_offset_base + l3_idx * 16

    table = l1_table
    for l2 in l2_tables:
        table.extend(l2)
    for l3 in l3_tables:
        table.extend(l3)

    return table