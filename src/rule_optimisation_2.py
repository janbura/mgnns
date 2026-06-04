# BEST APPROXIMATION 2
# This uses old code, and can only be done for 2 layers and relu. Essentially it multiplies the products
# of the matrix weights along an `influence path': for example, if we have 2 layers, position 1 in layer 0 affects
# positions 2 and 3 in layer 1, which in turn affect position 4 in layer 2, (1->2->4) and (1->3->4) are two
# differenc influence paths. We sort influence paths by value and add atoms in this order.
if model.num_layers == 2 and model.activation(1) == torch.relu:

    # An input unit for a given node, i, and layer is a triple (node',col,j) where node' is connected to node
    # via a link col (or, if col=-1, node'=node), and it holds that both the j-th feature of node' in layer-1 is
    # positive, max among those features for the col and j, and the (i,j)-weight of the matrix for colour col
    # (matrix A if col=-1) is also positive. Intuitively, it captures all inputs that affect the value of the
    # ith feature of node in layer A, on this dataset.
    def get_input_units(node_as_row, ii, layer):
        return_list = []
        for clr in set(can_encoder_decoder.colours).union({-1}):
            if clr == -1:
                matrix = model.matrix_A(layer)
                nghbrs = {node_as_row}
            else:
                matrix = model.matrix_B(layer=layer, colour=clr)
                mask = gd_edge_colour_list == clr
                gd_clr_edge_list = gd_edge_list[:, mask]
                nghbrs = set(gd_clr_edge_list[:, gd_clr_edge_list[1] == node_as_row][0].tolist())
            for jj in range(model.layer_dimension(layer - 1)):
                if matrix[ii][jj].item() > 0:
                    mx_neighbour = None
                    mx_value = 0
                    for nghbr in nghbrs:
                        feature = gnn_output_gd[layer - 1][nghbr][jj].item()
                        if feature > mx_value:
                            mx_neighbour = nghbr
                            mx_value = feature
                    if mx_neighbour is not None:
                        return_list.append((mx_neighbour, clr, jj))
        return return_list


    r_body_dataset = []
    if not test_gr_dataset(r_body_dataset):
        contributions = []
        for (source2_row, col2, j2) in get_input_units(cd_fact_gd_row, cd_fact_pred_pos, 2):
            source2_node = gd_row_to_node_dict[source2_row]
            if col2 == -1:
                z2 = "X1"
            else:
                variable_counter += 1
                z2 = "X" + str(variable_counter)
                nu_node_to_variable_dict[source2_node] = z2
                nu_variable_to_node_dict[z2] = source2_node
            next_level = get_input_units(source2_row, j2, 1)
            if col2 == -1:
                matrix2 = model.matrix_A(layer=2)
            else:
                matrix2 = model.matrix_B(layer=2, colour=col2)
            if not next_level:
                c_value = matrix2[cd_fact_pred_pos][j2] * gnn_output_gd[1][cd_fact_gd_row][j2]
                contributions.append((c_value, z2, None, col2, None, j2, None))
            for (source1_row, col1, j1) in get_input_units(source2_row, j2, 1):
                source1_node = gd_row_to_node_dict[source1_row]
                if col1 == -1:
                    z1 = z2
                else:
                    variable_counter += 1
                    z1 = "X" + str(variable_counter)
                    nu_node_to_variable_dict[source1_node] = z1
                    nu_variable_to_node_dict[z1] = source1_node
                if col1 == -1:
                    matrix1 = model.matrix_A(layer=1)
                else:
                    matrix1 = model.matrix_B(layer=1, colour=col1)
                contribution_value = matrix2[cd_fact_pred_pos][j2] * matrix1[j2][j1]
                contributions.append((contribution_value, z2, z1, col2, col1, j2, j1))
        if True:  # In contributions where the second colour col2 is -1, there might be multiple influence paths
            # we sum them here
            new_contributions = []
            cont = {}
            for contrib in contributions:
                contribution_value, z2, z1, col2, col1, j2, j1 = contrib
                if col2 == -1:
                    if j1 in cont:
                        cont[(z2, z1, col1, j1)] += contribution_value
                    else:
                        cont[(z2, z1, col1, j1)] = contribution_value
                else:
                    new_contributions.append(contrib)
            for (z2, z1, col1, j1) in cont:
                new_contributions.append((cont[z2, z1, col1, j1], z2, z1, col2, col1, 0, j1))
            contributions = new_contributions

        contributions = sorted(contributions, reverse=True)
        threshold_met = False
        used_contributions = []
        contributions_to_atoms_necessary = {}
        while not threshold_met and contributions:
            contrib = contributions.pop(0)
            contributions_to_atoms_necessary[contrib] = []
            used_contributions.append(contrib)
            contribution_value, z2, z1, col2, col1, _, j1 = contrib
            if col2 == -1:
                if col1 is None:
                    pass
                elif col1 == -1:
                    atom = ("X1", type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                    r_body_dataset.append(atom)
                    contributions_to_atoms_necessary[contrib].append(atom)
                    threshold_met = test_gr_dataset(r_body_dataset)
                else:
                    binary_atom_1 = (z1, can_encoder_decoder.colour_binary_pred_dict[col1], "X1")
                    contributions_to_atoms_necessary[contrib].append(binary_atom_1)
                    if binary_atom_1 not in r_body_dataset:
                        r_body_dataset.append(binary_atom_1)
                        threshold_met = test_gr_dataset(r_body_dataset)
                    if not threshold_met:
                        atom = (z1, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                        contributions_to_atoms_necessary[contrib].append(atom)
                        r_body_dataset.append(atom)
                        threshold_met = test_gr_dataset(r_body_dataset)
            else:
                binary_atom_2 = (z2, can_encoder_decoder.colour_binary_pred_dict[col2], "X1")
                contributions_to_atoms_necessary[contrib].append(binary_atom_2)
                if binary_atom_2 not in r_body_dataset:
                    r_body_dataset.append(binary_atom_2)
                    threshold_met = test_gr_dataset(r_body_dataset)
                if not threshold_met and col1 is not None:
                    if col1 == -1:
                        atom = (z2, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                        contributions_to_atoms_necessary[contrib].append(atom)
                        r_body_dataset.append(atom)
                        threshold_met = test_gr_dataset(r_body_dataset)
                    else:
                        binary_atom_1 = (z1, can_encoder_decoder.colour_binary_pred_dict[col1], z2)
                        contributions_to_atoms_necessary[contrib].append(binary_atom_1)
                        if binary_atom_1 not in r_body_dataset:
                            r_body_dataset.append(binary_atom_1)
                            threshold_met = test_gr_dataset(r_body_dataset)
                        if not threshold_met:
                            atom = (z1, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                            contributions_to_atoms_necessary[contrib].append(atom)
                            r_body_dataset.append(atom)
                            threshold_met = test_gr_dataset(r_body_dataset)
        (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(
            r_body_dataset)
        gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
        gnn_output_gr = model.all_labels(gr_dataset)
        necessary_body_atoms = set()
        while used_contributions:
            contrib = used_contributions.pop()
            (contribution_value, z2, z1, col2, col1, j2, j1) = contrib
            # if gnn_output_gr[1][node_to_gr_row_dict[nodes.const_node_dict[z2]]][j2] != 0:
            for atom in contributions_to_atoms_necessary[contrib]:
                necessary_body_atoms.add(atom)
        short_body_2 = list(necessary_body_atoms)